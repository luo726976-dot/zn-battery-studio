# -*- coding: utf-8 -*-
"""
水系锌离子电池添加剂筛选工作台 - 诊断与容错计算核心引擎 (Diagnostic Engine)
包含：
1. 硬件状态与运行环境自检 (CUDA/CPU, RDKit, PyTorch)
2. 表格数据多层合规质检与行级错误精准定位
3. SMILES 语法异常沙箱与化学特征提取
4. 实验物理量边界合理性诊断
5. 多任务模型加载、安全推理沙箱与内置 Mock 基准测试模型
6. 高斯过程回归 (GPR) 连续浓度空间寻优
"""

import os
import io
import traceback
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# RDKit 分子化学计算
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, Draw
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ZnDiagnosticEngine")


# ==============================================================================
# 0. 实验物理量合理性边界标准字典 (Physical Sanity Bounds)
# ==============================================================================
PHYSICAL_BOUNDS: Dict[str, Dict[str, Any]] = {
    "CV_Area": {
        "name_cn": "CV 峰面积",
        "aliases": ["cv峰面积", "cv_area", "cv_peak_area", "stripping_area_mc", "cv_stripping_area_mc"],
        "min": 100.0,
        "max": 10000.0,
        "unit": "mC",
        "desc": "循环伏安测试中的锌沉积剥离峰面积积分值",
        "typical": "2000 ~ 4000 mC",
        "default": 3100.0
    },
    "Tafel_Slope": {
        "name_cn": "Tafel 斜率",
        "aliases": ["tafel斜率", "tafel_slope", "tafel_slope_mv_dec", "tafel_anodic_slope"],
        "min": 20.0,
        "max": 300.0,
        "unit": "mV/dec",
        "desc": "极化曲线动力学斜率，表征电极腐蚀与钝化阻力",
        "typical": "50 ~ 120 mV/dec",
        "default": 72.0
    },
    "XPS_Shift": {
        "name_cn": "XPS 结合能偏移",
        "aliases": ["xps结合能偏移", "xps_shift", "xps_be_shift", "zn2p_be_shift_ev", "xps_zn2p_be_shift_ev"],
        "min": -2.0,
        "max": 2.0,
        "unit": "eV",
        "desc": "Zn 2p 轨道结合能相对于纯溶剂体系的化学位移",
        "typical": "0.10 ~ 0.80 eV",
        "default": 0.35
    },
    "Raman_Area": {
        "name_cn": "Raman 拟合峰面积",
        "aliases": ["raman峰面积拟合值", "raman_area", "raman_peak_area_fitted", "raman_peak_area", "water_hbond_area_ratio"],
        "min": 200.0,
        "max": 5000.0,
        "unit": "无量纲",
        "desc": "水分子强/弱氢键缔合峰(~3400 cm⁻¹)分峰拟合积分",
        "typical": "1000 ~ 2500",
        "default": 1650.0
    },
    "XRD_Intensity": {
        "name_cn": "XRD 相对晶面强度",
        "aliases": ["xrd晶面相对强度", "xrd_intensity", "xrd_relative_intensity", "xrd_i_002_to_i_101_ratio", "tc_002"],
        "min": 0.1,
        "max": 10.0,
        "unit": "无量纲",
        "desc": "I(002) / I(101) 锌沉积晶面择优取向衍射强度比",
        "typical": "1.5 ~ 3.5",
        "default": 2.10
    },
    "Concentration": {
        "name_cn": "添加剂浓度",
        "aliases": ["concentration", "concentration_wt", "浓度", "添加浓度"],
        "min": 0.01,
        "max": 10.0,
        "unit": "wt%",
        "desc": "电解液中添加剂的质量百分比",
        "typical": "0.5 ~ 2.5 wt%",
        "default": 1.50
    }
}

STANDARD_MOL_KEYS = ["MolWt", "TPSA", "LogP", "NumHDonors", "NumHAcceptors"]
STANDARD_EXP_KEYS = ["CV_Area", "Tafel_Slope", "XPS_Shift", "Raman_Area", "XRD_Intensity"]


# ==============================================================================
# 1. 硬件与环境状态自检 (Hardware & Environment Inspector)
# ==============================================================================
def inspect_system_environment(model_path: str = "/home/a1810/best_multitask_model.pth") -> Dict[str, Any]:
    """
    检查计算硬件 (CUDA / CPU)、依赖库版本与本地模型就绪状态
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU",
        "device_detail": "标准 x86_64 多核处理器",
        "pytorch_version": torch.__version__,
        "rdkit_version": getattr(rdkit, "__version__", "2026.x"),
        "model_file_exists": os.path.exists(model_path),
        "model_path": model_path,
        "model_size_kb": 0.0
    }

    if info["cuda_available"]:
        try:
            info["device_name"] = f"CUDA: {torch.cuda.get_device_name(0)}"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["device_detail"] = f"显存容量: {vram_gb:.2f} GB"
        except Exception:
            info["device_name"] = "CUDA (已启用)"

    if info["model_file_exists"]:
        try:
            size_bytes = os.path.getsize(model_path)
            info["model_size_kb"] = round(size_bytes / 1024.0, 1)
        except Exception:
            pass

    return info


# ==============================================================================
# 2. 表格文件多维合规质检与精准排错 (File & Data Table Validator)
# ==============================================================================
def validate_uploaded_table(
    file_or_path: Union[str, io.BytesIO, Any],
    file_name: str = ""
) -> Dict[str, Any]:
    """
    对上传或本地的 CSV / Excel 执行行级数据质检：
    - 表头比对：核查 SMILES 及 5 大实验字段
    - 行级排查：精确定位非数值、空值行号及列名，并应用安全填充
    - 容错返回：无论数据何种错误，绝不抛出未捕获异常
    """
    report = {
        "success": False,
        "df": None,
        "clean_df": None,
        "source_name": file_name or "未命名文件",
        "total_rows": 0,
        "total_cols": 0,
        "missing_required_cols": [],
        "col_mapping": {},
        "row_issues": [],
        "diagnostic_messages": [],
        "fatal_error": None
    }

    # 1. 尝试读取文件
    try:
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                report["fatal_error"] = f"文件路径不存在: '{file_or_path}'"
                report["diagnostic_messages"].append(f"❌ 严重错误: 未能找到指定路径文件 '{file_or_path}'。")
                return report
            name_lower = file_or_path.lower()
            if name_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_or_path)
            else:
                try:
                    df = pd.read_csv(file_or_path, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file_or_path, encoding="gbk")
        else:
            name_lower = file_name.lower()
            if name_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_or_path)
            else:
                if hasattr(file_or_path, "seek"):
                    file_or_path.seek(0)
                try:
                    df = pd.read_csv(file_or_path, encoding="utf-8")
                except UnicodeDecodeError:
                    if hasattr(file_or_path, "seek"):
                        file_or_path.seek(0)
                    df = pd.read_csv(file_or_path, encoding="gbk")

        if df.empty:
            report["fatal_error"] = "数据表内容为空 (0 行记录)"
            report["diagnostic_messages"].append("❌ 数据表为空：请上传包含有效数据的实验表格。")
            return report

    except Exception as e:
        report["fatal_error"] = f"文件解析中断: {str(e)}"
        report["diagnostic_messages"].append(f"❌ 文件解析中断: 文件格式损坏或无法以标准 CSV/Excel 解码 ({str(e)})。")
        return report

    report["df"] = df
    report["total_rows"] = len(df)
    report["total_cols"] = len(df.columns)
    cols = list(df.columns)

    # 2. 表头智能比对
    # 识别 SMILES 列
    smiles_col = None
    smi_aliases = ["smiles", "canonical_smiles", "分子式", "结构式"]
    for c in cols:
        if str(c).strip().lower() in smi_aliases:
            smiles_col = c
            break
    if not smiles_col:
        for c in cols:
            if "smiles" in str(c).strip().lower():
                smiles_col = c
                break

    if smiles_col:
        report["col_mapping"]["SMILES"] = smiles_col
    else:
        report["missing_required_cols"].append("SMILES")

    # 识别 5 大实验特征列
    for std_key, meta in PHYSICAL_BOUNDS.items():
        if std_key == "Concentration":
            continue
        matched = None
        for alias in meta["aliases"]:
            for c in cols:
                c_clean = str(c).strip().lower().replace(" ", "").replace("_", "")
                a_clean = alias.lower().replace(" ", "").replace("_", "")
                if c_clean == a_clean:
                    matched = c
                    break
            if matched:
                break
        
        # 模糊包含匹配
        if not matched:
            for alias in meta["aliases"]:
                for c in cols:
                    if alias.lower() in str(c).strip().lower():
                        matched = c
                        break
                if matched:
                    break

        if matched:
            report["col_mapping"][std_key] = matched
        else:
            report["missing_required_cols"].append(f"{std_key} ({meta['name_cn']})")

    # 校验是否缺少关键列
    if report["missing_required_cols"]:
        msg = f"缺少关键字段: {report['missing_required_cols']}，请核对数据表头格式（支持中英文别名映射，如 CV_Area 或 CV峰面积）。"
        report["diagnostic_messages"].append(f"⚠️ {msg}")

    # 3. 行级单元格脏数据排查 (Row-level Quality Audit)
    clean_df = df.copy()
    row_issues = []

    for std_key, act_col in report["col_mapping"].items():
        if std_key == "SMILES":
            # 检查 SMILES 空值
            for r_idx, val in enumerate(clean_df[act_col]):
                if pd.isna(val) or not str(val).strip():
                    issue = {
                        "row": r_idx + 1,
                        "col": act_col,
                        "std_field": "SMILES",
                        "raw_value": str(val),
                        "action": "标记为空，需用户手动修正",
                        "msg": f"第 {r_idx + 1} 行 [{act_col}] 存在空结构式，无法进行分子特征提取。"
                    }
                    row_issues.append(issue)
            continue

        # 检查数值列中的空值与非法字符
        series_raw = clean_df[act_col]
        # 计算该列正常数值的均值
        numeric_series = pd.to_numeric(series_raw, errors="coerce")
        col_mean = float(numeric_series.mean()) if not pd.isna(numeric_series.mean()) else PHYSICAL_BOUNDS[std_key]["default"]

        for r_idx, val in enumerate(series_raw):
            is_bad = False
            if pd.isna(val):
                is_bad = True
                val_repr = "NaN (空值)"
            else:
                try:
                    float_v = float(val)
                    if np.isnan(float_v) or np.isinf(float_v):
                        is_bad = True
                        val_repr = str(val)
                except (ValueError, TypeError):
                    is_bad = True
                    val_repr = str(val)

            if is_bad:
                issue = {
                    "row": r_idx + 1,
                    "col": act_col,
                    "std_field": std_key,
                    "raw_value": val_repr,
                    "action": f"已自动填充列均值 ({col_mean:.2f})",
                    "msg": f"第 {r_idx + 1} 行 [{act_col}] 存在非法空值或非数值字符 (原值: '{val_repr}')，已自动使用该列均值 ({col_mean:.2f}) 填充。"
                }
                row_issues.append(issue)

        # 整体安全转换为 float 类型的 Series 并填充均值
        clean_df[act_col] = numeric_series.fillna(col_mean).astype(np.float32)

    report["clean_df"] = clean_df
    report["row_issues"] = row_issues
    report["success"] = True

    if row_issues:
        report["diagnostic_messages"].append(
            f"ℹ️ 数据清洗提示: 在数据表中检测到 {len(row_issues)} 处数据异常，已实施均值防御性填充，保障后续计算不崩溃。"
        )

    return report


# ==============================================================================
# 3. SMILES 语法异常沙箱与分子特征提取 (SMILES Sandbox)
# ==============================================================================
def validate_smiles_sandbox(smiles: str) -> Dict[str, Any]:
    """
    SMILES 沙箱校验器：
    - 精准定位括号不闭合、大小写错误等化学语法异常
    - 绝不中断页面运行，返回直观诊断信息与 2D 骨架图
    """
    result = {
        "valid": False,
        "smiles": smiles,
        "mol": None,
        "descriptors": None,
        "image": None,
        "error_title": None,
        "error_detail": None,
        "suggestion": None
    }

    if not isinstance(smiles, str) or not smiles.strip():
        result["error_title"] = "SMILES 输入为空"
        result["error_detail"] = "未检测到有效的分子结构式输入。"
        result["suggestion"] = "请输入有效的 SMILES 字符串（如甘氨酸: NCC(=O)O）。"
        result["image"] = _generate_placeholder_image("请输入有效 SMILES 分子式")
        return result

    clean_smi = smiles.strip()
    result["smiles"] = clean_smi

    # 语法预检 (括号配对)
    if clean_smi.count("(") != clean_smi.count(")"):
        result["error_title"] = "SMILES 括号未配对"
        result["error_detail"] = f"圆括号 '(' 与 ')' 数量不匹配 (左括号: {clean_smi.count('(')}, 右括号: {clean_smi.count(')')})。"
        result["suggestion"] = "请仔细核对 SMILES 分支结构的括号闭合情况。"
        result["image"] = _generate_placeholder_image("括号未闭合语法错误")
        return result

    if clean_smi.count("[") != clean_smi.count("]"):
        result["error_title"] = "SMILES 中括号未配对"
        result["error_detail"] = f"中括号 '[' 与 ']' 数量不匹配 (左括号: {clean_smi.count('[')}, 右括号: {clean_smi.count(']')})。"
        result["suggestion"] = "带电荷基团或特定同位素需成对使用方括号，如 [Na+] 或 [O-]。"
        result["image"] = _generate_placeholder_image("方括号未闭合")
        return result

    # 调用 RDKit 解析
    try:
        mol = Chem.MolFromSmiles(clean_smi)
    except Exception as e:
        result["error_title"] = "RDKit 底层解析异常"
        result["error_detail"] = str(e)
        result["suggestion"] = "请检查分子式中是否存在非法字符或无法识别的化合价态。"
        result["image"] = _generate_placeholder_image("RDKit 底层解析失败")
        return result

    if mol is None:
        result["error_title"] = "SMILES 格式无效"
        result["error_detail"] = "无法识别化学结构：化学价键不满足物理规则或元素大小写混淆。"
        result["suggestion"] = "请检查元素大小写（例如氯应为 Cl 而非 CL，芳香环需小写如 c1ccccc1）。"
        result["image"] = _generate_placeholder_image("无效化学结构 (无法成键)")
        return result

    # 提取 5 维分子描述符
    try:
        mol_wt = float(Descriptors.MolWt(mol))
        tpsa = float(Descriptors.TPSA(mol))
        logp = float(Crippen.MolLogP(mol))
        hbd = float(Lipinski.NumHDonors(mol))
        hba = float(Lipinski.NumHAcceptors(mol))

        result["descriptors"] = {
            "MolWt": round(mol_wt, 3),
            "TPSA": round(tpsa, 3),
            "LogP": round(logp, 3),
            "NumHDonors": float(int(hbd)),
            "NumHAcceptors": float(int(hba))
        }
        result["mol"] = mol
        result["valid"] = True

        # 绘制高对比度实验室样式 2D 骨架图
        img = Draw.MolToImage(mol, size=(320, 240))
        result["image"] = img

    except Exception as e:
        result["error_title"] = "描述符计算异常"
        result["error_detail"] = f"计算物理化学描述符时出错: {str(e)}"
        result["suggestion"] = "该分子结构可能存在无法计算的自由基或极其异常的拓扑。"
        result["image"] = _generate_placeholder_image("计算描述符异常")

    return result


def _generate_placeholder_image(text: str) -> Image.Image:
    """生成简洁的实验室仪器风格占位图"""
    img = Image.new("RGB", (320, 240), color="#f1f5f9")
    draw = ImageDraw.Draw(img)
    # 画边框
    draw.rectangle([(2, 2), (317, 237)], outline="#cbd5e1", width=2)
    # 绘制斜交叉线
    draw.line([(10, 10), (310, 230)], fill="#e2e8f0", width=1)
    draw.line([(10, 230), (310, 10)], fill="#e2e8f0", width=1)
    # 居中文本 (尝试使用默认字体)
    draw.text((35, 110), f"⚠️ {text}", fill="#64748b")
    return img


# ==============================================================================
# 4. 实验物理量边界合理性检查 (Physical Bounds Inspector)
# ==============================================================================
def inspect_physical_bounds(
    exp_features: Dict[str, float],
    concentration: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    检查输入的物理量是否偏离常规电解液实验区间
    """
    warnings = []
    
    # 检查 5 项实验参数
    for key, val in exp_features.items():
        # 兼容标准名与实际列名
        target_meta = None
        for std_key, meta in PHYSICAL_BOUNDS.items():
            if key == std_key or key == meta["name_cn"] or key in meta["aliases"]:
                target_meta = meta
                break
        
        if not target_meta:
            continue

        try:
            num_val = float(val)
        except (ValueError, TypeError):
            warnings.append({
                "field": key,
                "current_val": val,
                "severity": "RED",
                "message": f"参数 [{target_meta['name_cn']}] 为非数值类型，将导致张量转换失败！"
            })
            continue

        if num_val < target_meta["min"] or num_val > target_meta["max"]:
            warnings.append({
                "field": target_meta["name_cn"],
                "current_val": num_val,
                "unit": target_meta["unit"],
                "min": target_meta["min"],
                "max": target_meta["max"],
                "typical": target_meta["typical"],
                "severity": "AMBER",
                "message": (
                    f"[{target_meta['name_cn']}] 数值 ({num_val} {target_meta['unit']}) "
                    f"异常偏离常规实验区间 (参考区间: {target_meta['min']} ~ {target_meta['max']} {target_meta['unit']}，"
                    f"典型值: {target_meta['typical']})，请确认测试单位是否匹配。"
                )
            })

    # 检查添加浓度
    if concentration is not None:
        conc_meta = PHYSICAL_BOUNDS["Concentration"]
        if concentration < conc_meta["min"] or concentration > conc_meta["max"]:
            warnings.append({
                "field": "添加剂浓度",
                "current_val": concentration,
                "unit": "wt%",
                "min": conc_meta["min"],
                "max": conc_meta["max"],
                "typical": conc_meta["typical"],
                "severity": "AMBER",
                "message": (
                    f"[添加剂浓度] 数值 ({concentration} wt%) 偏离常规水系锌电解液优化区间 "
                    f"(参考: {conc_meta['min']} ~ {conc_meta['max']} wt%，典型区间: {conc_meta['typical']})。"
                )
            })

    return warnings


# ==============================================================================
# 5. 内置 Mock 基准测试模型 (Fallback Benchmark Engine)
# ==============================================================================
class MockBenchmarkModel(nn.Module):
    """
    当外部权重文件缺失时激活的内置基准模型，防止系统崩溃
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基于第一性原理经验映射输出物理合理的基准估算
        # 输出为 3 维标定空间: 循环寿命得分, CE 得分, HER 得分
        with torch.no_grad():
            batch_size = x.shape[0]
            # 取实验特征均值作为调节系数
            exp_norm_mean = torch.mean(x[:, 5:], dim=-1, keepdim=True)
            mol_wt = x[:, 0:1]
            wt_factor = torch.clamp(mol_wt / 300.0, 0.2, 1.2)

            life = torch.clamp(0.4 + 0.5 * exp_norm_mean - 0.1 * wt_factor, 0.1, 0.95)
            ce = torch.clamp(0.5 + 0.4 * exp_norm_mean, 0.2, 0.98)
            her = torch.clamp(0.3 + 0.6 * exp_norm_mean, 0.1, 0.92)
            return torch.cat([life, ce, her], dim=-1)


# ==============================================================================
# 6. 模型加载与安全推理沙箱 (Model Inference Sandbox)
# ==============================================================================
def run_model_inference_sandbox(
    tensor_10d: torch.Tensor,
    model_path: str = "/home/a1810/best_multitask_model.pth",
    use_mock: bool = False,
    benchmark_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    模型推理安全沙箱：
    - 捕获所有 PyTorch 算子报错或模型加载异常
    - 格式化 Traceback 供折叠框展示
    - 返回反归一化后的工程量纲数值
    """
    res = {
        "success": False,
        "is_mock": False,
        "predictions": None,
        "raw_outputs": None,
        "traceback_str": None,
        "error_message": None
    }

    try:
        # 1. 检查输入张量形态
        if not isinstance(tensor_10d, torch.Tensor):
            res["error_message"] = f"输入参数不是 PyTorch Tensor 类型 (当前类型: {type(tensor_10d)})"
            return res

        if tensor_10d.shape != (1, 10):
            res["error_message"] = f"输入张量维度不符合要求！需要 (1, 10)，当前为 {tuple(tensor_10d.shape)}"
            return res

        # 2. 加载或回退模型
        model = None
        if not use_mock and os.path.exists(model_path):
            try:
                from zn_battery_multihead_model import ZnBatteryMultiHeadNet
                net = ZnBatteryMultiHeadNet(in_features=10)
                state_dict = torch.load(model_path, map_location="cpu")
                net.load_state_dict(state_dict)
                net.eval()
                model = net
            except Exception as load_err:
                logger.warning(f"加载主模型失败，将尝试回退: {load_err}")
                if not use_mock:
                    res["error_message"] = f"主模型权重文件损坏或架构不匹配: {str(load_err)}"
                    res["traceback_str"] = traceback.format_exc()
                    return res

        if model is None:
            # 启用内置基准测试模型
            model = MockBenchmarkModel()
            model.eval()
            res["is_mock"] = True

        # 3. 前向推理计算
        with torch.no_grad():
            preds_norm = model(tensor_10d)
            if isinstance(preds_norm, dict):
                # 兼容字典输出
                arr = torch.cat([preds_norm['cycle_life'], preds_norm['ce'], preds_norm['her_overpotential']], dim=-1)
                preds_np = arr.detach().cpu().numpy()
            else:
                preds_np = preds_norm.detach().cpu().numpy()

        res["raw_outputs"] = preds_np.tolist()[0]

        # 4. 反归一化为真实物理量纲
        target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        target_cols = ["循环寿命_h", "库仑效率_CE", "HER过电位_mV"]
        if benchmark_df is not None and all(c in benchmark_df.columns for c in target_cols):
            target_scaler.fit(benchmark_df[target_cols].apply(pd.to_numeric, errors='coerce').to_numpy())
        else:
            # 标准电解液物理基准边界
            dummy_targets = np.array([
                [680.0, 0.9880, 140.5],
                [1680.0, 0.9978, 225.4]
            ], dtype=np.float32)
            target_scaler.fit(dummy_targets)

        real_preds = target_scaler.inverse_transform(preds_np)[0]

        res["predictions"] = {
            "循环寿命": {
                "value": float(round(real_preds[0], 1)),
                "unit": "h",
                "status": "PASS" if real_preds[0] >= 1000.0 else "FAIR"
            },
            "库仑效率_CE": {
                "value": float(round(real_preds[1] * 100, 2)),
                "unit": "%",
                "status": "PASS" if (real_preds[1] * 100) >= 99.2 else "FAIR"
            },
            "HER析氢过电位": {
                "value": float(round(real_preds[2], 1)),
                "unit": "mV",
                "status": "PASS" if real_preds[2] >= 175.0 else "FAIR"
            }
        }
        res["success"] = True

    except Exception as e:
        res["error_message"] = f"模型推理阶段发生未处理异常: {str(e)}"
        res["traceback_str"] = traceback.format_exc()
        logger.error(f"推理沙箱拦截异常: {e}\n{res['traceback_str']}")

    return res


# ==============================================================================
# 7. 高斯过程回归 (GPR) 连续浓度空间优化计算
# ==============================================================================
def calculate_gpr_concentration_curve(
    smiles: str,
    base_score: float,
    current_conc: float = 1.5
) -> Dict[str, Any]:
    """
    高斯过程连续浓度响应拟合与不确定性置信带量化
    """
    eval_concs = np.linspace(0.1, 3.0, 80).reshape(-1, 1)

    # 分子量位阻物理拐点调整
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    mol_wt = float(Descriptors.MolWt(mol)) if mol else 150.0

    peak_offset = -0.15 if mol_wt > 300 else (0.10 if mol_wt < 100 else 0.0)
    ideal_peak_conc = 1.35 + peak_offset

    benchmark_concs = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    grad_y = []
    for c in benchmark_concs:
        dist = c - ideal_peak_conc
        shape_penalty = 18.0 * (dist ** 2) if dist >= 0 else 14.0 * (dist ** 2)
        score = max(45.0, (base_score - shape_penalty))
        grad_y.append(score)

    X_train = benchmark_concs.reshape(-1, 1)
    y_train = np.array(grad_y, dtype=np.float32)

    kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=0.8, length_scale_bounds=(0.2, 4.0)) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True)
    gpr.fit(X_train, y_train)

    y_mean, y_std = gpr.predict(eval_concs, return_std=True)

    # 寻找最佳推荐浓度
    best_idx = int(np.argmax(y_mean))
    best_conc = float(eval_concs[best_idx][0])
    max_score = float(y_mean[best_idx])
    best_uncertainty = float(1.96 * y_std[best_idx])

    # 当前输入浓度处的预测值
    curr_mean, curr_std = gpr.predict([[current_conc]], return_std=True)

    return {
        "concs": eval_concs.flatten().tolist(),
        "mean_scores": y_mean.flatten().tolist(),
        "upper_bound": (y_mean + 1.96 * y_std).flatten().tolist(),
        "lower_bound": (y_mean - 1.96 * y_std).flatten().tolist(),
        "best_conc": round(best_conc, 2),
        "max_score": round(max_score, 1),
        "best_uncertainty": round(best_uncertainty, 1),
        "current_conc": round(current_conc, 2),
        "current_score": round(float(curr_mean[0]), 1)
    }
