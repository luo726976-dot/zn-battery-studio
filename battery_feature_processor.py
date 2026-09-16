# -*- coding: utf-8 -*-
"""
水系锌离子电池添加剂特征采集、解析、对齐与张量打包核心模块
(Aqueous Zinc-ion Battery Additive Feature Acquisition & Tensor Processing Engine)

模块划分：
1. DatabaseManager: 本地数据库与上传文件 (CSV/Excel) 解析、特征列智能匹配、MinMaxScaler 归一化标定
2. MolecularPropertyEngine: 分子 SMILES 智能检索 (内置字典 + PubChemPy) 及 RDKit 5 维化学描述符计算
3. FeatureAlignmentPipeline: 特征对齐融合、归一化转换、PyTorch [1, 10] 张量打包与画像构建
"""

import os
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# RDKit 分子化学计算模块
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, Draw

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ZnBatteryProcessor")


# ==============================================================================
# 0. 预置典型锌电池添加剂常用字典 (内置高速匹配库)
# ==============================================================================
PRESET_ADDITIVE_SMILES_DICT: Dict[str, str] = {
    # 核心目标添加剂
    "2-氨基-4-溴蒽醌-2-磺酸钠": "Nc1c(S(=O)(=O)[O-])cc(Br)c2c1C(=O)c1ccccc1C2=O.[Na+]",
    "溴氨酸钠": "Nc1c(S(=O)(=O)[O-])cc(Br)c2c1C(=O)c1ccccc1C2=O.[Na+]",
    "甘氨酸": "NCC(=O)O",
    "柠檬酸": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    "硫脲": "NC(=S)N",
    "D-葡萄糖": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "葡萄糖": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "L-抗坏血酸": "C1=C(C(=O)OC1C(CO)O)O",
    "抗坏血酸": "C1=C(C(=O)OC1C(CO)O)O",
    "维生素C": "C1=C(C(=O)OC1C(CO)O)O",
    "聚乙二醇二甲醚单体": "COCCOC",
    "无水烟酰胺": "NC(=O)c1cccnc1",
    "烟酰胺": "NC(=O)c1cccnc1",
    "聚乙烯吡咯烷酮单体": "C1CC(=O)N(C1)C=C",
    "尿素": "NC(=O)N",
    "甜菜碱": "C[N+](C)(C)CC(=O)[O-]",
    "丁二酸": "C(CC(=O)O)C(=O)O",
    "琥珀酸": "C(CC(=O)O)C(=O)O",
    "乙酰丙酮": "CC(=O)CC(=O)C",
    "季戊四醇": "C(C(CO)(CO)CO)O",
    "谷氨酸": "C(CC(=O)O)C(C(=O)O)N",
    "丙氨酸": "CC(C(=O)O)N",
    "麦芽糖": "C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O",
    "邻菲罗啉": "c1cnc2c(c1)ccc3c2nccc3",
    "1,10-菲罗啉": "c1cnc2c(c1)ccc3c2nccc3",
    "三氟甲磺酸根单体": "C(F)(F)(F)S(=O)(=O)[O-]",
    "三氟甲磺酸钠": "C(F)(F)(F)S(=O)(=O)[O-].[Na+]",
    "苯甲酸钠": "c1ccccc1C(=O)[O-].[Na+]",
    "苯甲酸": "c1ccccc1C(=O)O",
    "乙二醇": "OCCO",
    "丙烯酰胺": "C=CC(=O)N",
    "咖啡因": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "丝氨酸": "C(C(C(=O)O)N)O",
    "脯氨酸": "C1CC(NC1)C(=O)O",
    "甲酸钠": "C(=O)[O-].[Na+]",
    "醋酸锌": "CC(=O)[O-].CC(=O)[O-].[Zn+2]",
    "十二烷基硫酸钠": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
}

# 5 大 Origin 实验指标的标准名称及其在 CSV/Excel 中的常见同义词列表
EXP_FEATURE_ALIASES: Dict[str, List[str]] = {
    "CV峰面积": [
        "CV峰面积", "cv峰面积", "cv_peak_area", "cv_area", "CV_peak_area",
        "cv_stripping_area_mC", "stripping_area_mC", "cv_area_mC", "CV Peak Area"
    ],
    "Tafel斜率": [
        "Tafel斜率", "tafel斜率", "tafel_slope", "Tafel_slope", "tafel_slope_mV_dec",
        "tafel_anodic_slope", "tafel_cathodic_slope", "Tafel Slope"
    ],
    "XPS结合能偏移": [
        "XPS结合能偏移", "xps结合能偏移", "xps_be_shift", "XPS_BE_shift",
        "xps_Zn2p_BE_shift_eV", "Zn2p_BE_shift_eV", "xps_binding_energy_shift", "XPS Shift"
    ],
    "Raman峰面积拟合值": [
        "Raman峰面积拟合值", "raman峰面积拟合值", "raman_peak_area_fitted",
        "raman_peak_area", "raman_fitted_area", "water_hbond_area_ratio", "Raman Peak Area"
    ],
    "XRD晶面相对强度": [
        "XRD晶面相对强度", "xrd晶面相对强度", "xrd_relative_intensity", "xrd_intensity_ratio",
        "xrd_I_002_to_I_101_ratio", "I_002_to_I_101_ratio", "TC_002", "XRD Ratio"
    ]
}

# 标准 10 维联合特征名称定义 (严格与之前模型对齐)
MOL_FEATURE_NAMES: List[str] = ["MolWt", "TPSA", "LogP", "NumHDonors", "NumHAcceptors"]
EXP_FEATURE_NAMES: List[str] = ["CV峰面积", "Tafel斜率", "XPS结合能偏移", "Raman峰面积拟合值", "XRD晶面相对强度"]
ALL_FEATURE_NAMES: List[str] = [f"mol_{k}" for k in MOL_FEATURE_NAMES] + [f"exp_{k}" for k in EXP_FEATURE_NAMES]


# ==============================================================================
# 1. 模块一：数据库管理与文件读取解析器 (DatabaseManager)
# ==============================================================================
class DatabaseManager:
    """
    负责本地 CSV / Excel 数据库的读取、表头智能解析、候选添加剂提取与归一化器构建
    """
    def __init__(self, preset_path: str = "/home/a1810/zn_battery_experiment_database.csv"):
        self.preset_path = preset_path
        self.df: Optional[pd.DataFrame] = None
        self.source_desc: str = "未加载"
        self.resolved_exp_cols: Dict[str, str] = {}
        self.additive_col: Optional[str] = None
        self.smiles_col: Optional[str] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_medians: Dict[str, float] = {}

        # 默认加载预设本地数据库
        self.load_preset_database()

    def load_preset_database(self) -> bool:
        """加载本地预设数据库文件"""
        if os.path.exists(self.preset_path):
            try:
                df = pd.read_csv(self.preset_path)
                self._parse_and_set_dataframe(df, f"本地预设数据库 ({os.path.basename(self.preset_path)})")
                return True
            except Exception as e:
                logger.error(f"加载预设数据库失败: {e}")
                return False
        else:
            logger.warning(f"预设数据库文件不存在: {self.preset_path}")
            return False

    def load_from_file_or_buffer(self, file_or_buffer: Union[str, io.BytesIO, Any], file_name: str = "") -> Tuple[bool, str]:
        """
        从用户上传的文件流或路径加载 CSV / Excel
        """
        try:
            name_lower = file_name.lower()
            if name_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_or_buffer)
            else:
                # 默认尝试 CSV 读取
                if isinstance(file_or_buffer, (io.BytesIO, io.StringIO)):
                    file_or_buffer.seek(0)
                try:
                    df = pd.read_csv(file_or_buffer, encoding="utf-8")
                except UnicodeDecodeError:
                    if hasattr(file_or_buffer, "seek"):
                        file_or_buffer.seek(0)
                    df = pd.read_csv(file_or_buffer, encoding="gbk")

            msg = self._parse_and_set_dataframe(df, f"用户上传文件: {file_name}")
            return True, msg
        except Exception as e:
            err_msg = f"解析上传文件失败: {str(e)}"
            logger.error(err_msg)
            return False, err_msg

    def _parse_and_set_dataframe(self, df: pd.DataFrame, source_desc: str) -> str:
        """解析 DataFrame 表头，识别添加剂名、SMILES及 5 维实验指标"""
        if df.empty:
            raise ValueError("数据表内容为空")

        df = df.copy()
        cols = list(df.columns)

        # 1. 识别添加剂名称列
        name_cands = ["additive_name", "additive", "添加剂名称", "添加剂", "name", "compound", "物质名称"]
        self.additive_col = None
        for cand in name_cands:
            for c in cols:
                if str(c).strip().lower() == cand.lower():
                    self.additive_col = c
                    break
            if self.additive_col:
                break
        if not self.additive_col:
            for c in cols:
                if df[c].dtype == object and c.lower() not in ["smiles", "id", "sample_id"]:
                    self.additive_col = c
                    break
        if not self.additive_col:
            self.additive_col = cols[0]

        # 2. 识别 SMILES 列
        smi_cands = ["smiles", "SMILES", "Smiles", "结构式", "分子式", "canonical_smiles"]
        self.smiles_col = None
        for cand in smi_cands:
            for c in cols:
                if str(c).strip().lower() == cand.lower():
                    self.smiles_col = c
                    break
            if self.smiles_col:
                break

        # 3. 识别 5 大实验特征列
        self.resolved_exp_cols = {}
        missing_exp = []
        for std_name, aliases in EXP_FEATURE_ALIASES.items():
            matched = None
            for cand in aliases:
                for c in cols:
                    if str(c).strip().lower() == cand.lower():
                        matched = c
                        break
                if matched:
                    break
            if matched is None:
                for cand in aliases:
                    for c in cols:
                        if cand.lower() in str(c).strip().lower():
                            matched = c
                            break
                    if matched:
                        break
            if matched:
                self.resolved_exp_cols[std_name] = matched
            else:
                missing_exp.append(std_name)

        self.df = df
        self.source_desc = source_desc

        # 4. 构建归一化器与统计中位数
        self._fit_scaler_and_calc_medians()

        report = (
            f"成功加载数据源: {source_desc} (共 {len(df)} 行，{len(cols)} 列)。\n"
            f"识别添加剂列: '{self.additive_col}'，SMILES列: '{self.smiles_col or '未自动识别'}'。\n"
            f"已对齐实验特征列数: {len(self.resolved_exp_cols)}/5。"
        )
        if missing_exp:
            report += f"\n注：以下实验指标未在表中找到对应列: {missing_exp} (将使用系统默认中位数补全)。"
        logger.info(report)
        return report

    def _fit_scaler_and_calc_medians(self) -> None:
        """拟合实验特征的 MinMaxScaler 并记录中位数作为安全填充底限"""
        if self.df is None or self.df.empty:
            return

        default_baseline = {
            "CV峰面积": 3100.0,
            "Tafel斜率": 72.0,
            "XPS结合能偏移": 0.35,
            "Raman峰面积拟合值": 1650.0,
            "XRD晶面相对强度": 2.10
        }

        exp_data = []
        for std_name in EXP_FEATURE_NAMES:
            if std_name in self.resolved_exp_cols:
                col = self.resolved_exp_cols[std_name]
                s = pd.to_numeric(self.df[col], errors="coerce")
                val_med = float(s.median()) if not pd.isna(s.median()) else default_baseline[std_name]
                self.feature_medians[std_name] = val_med
                s_filled = s.fillna(val_med).to_numpy(dtype=np.float32)
                exp_data.append(s_filled)
            else:
                val_med = default_baseline[std_name]
                self.feature_medians[std_name] = val_med
                exp_data.append(np.full(len(self.df), val_med, dtype=np.float32))

        exp_matrix = np.column_stack(exp_data).astype(np.float32)
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.scaler.fit(exp_matrix)
        logger.info("MinMaxScaler 已针对当前数据库实验特征完成拟合标定。")

    def get_additive_names(self) -> List[str]:
        """获取所有候选添加剂名称列表"""
        if self.df is None or self.additive_col is None:
            return list(PRESET_ADDITIVE_SMILES_DICT.keys())
        names = self.df[self.additive_col].dropna().astype(str).tolist()
        seen = set()
        unique_names = []
        for n in names:
            n_clean = n.strip()
            if n_clean and n_clean not in seen:
                seen.add(n_clean)
                unique_names.append(n_clean)
        return unique_names

    def get_additive_details(self, additive_name: str) -> Dict[str, Any]:
        """根据添加剂名称提取已知特征信息"""
        result: Dict[str, Any] = {
            "name": additive_name,
            "smiles": "",
            "exp_features": self.feature_medians.copy(),
            "found_in_db": False
        }

        if self.df is not None and self.additive_col is not None:
            match = self.df[self.df[self.additive_col].astype(str).str.strip() == additive_name.strip()]
            if not match.empty:
                row = match.iloc[0]
                result["found_in_db"] = True
                if self.smiles_col and self.smiles_col in row and pd.notna(row[self.smiles_col]):
                    result["smiles"] = str(row[self.smiles_col]).strip()
                for std_name in EXP_FEATURE_NAMES:
                    if std_name in self.resolved_exp_cols:
                        act_col = self.resolved_exp_cols[std_name]
                        val = row.get(act_col, None)
                        if pd.notna(val):
                            try:
                                result["exp_features"][std_name] = float(val)
                            except (ValueError, TypeError):
                                pass

        if not result["smiles"] and additive_name in PRESET_ADDITIVE_SMILES_DICT:
            result["smiles"] = PRESET_ADDITIVE_SMILES_DICT[additive_name]

        return result


# ==============================================================================
# 2. 模块二：分子化学性质与自动检索补全引擎 (MolecularPropertyEngine)
# ==============================================================================
class MolecularPropertyEngine:
    """
    负责分子 SMILES 校验、RDKit 5 维描述符计算、PubChem 在线检索与化学结构绘制
    """
    @staticmethod
    def query_smiles(name: str) -> Tuple[Optional[str], str, Optional[str]]:
        """
        根据分子名称自动检索/补全 SMILES：
        优先匹配内置高频电池添加剂字典；若未命中则调用 PubChemPy 在线 API。
        返回: (smiles, 检索来源, 错误信息)
        """
        clean_name = name.strip()
        if not clean_name:
            return None, "", "请输入有效的添加剂名称"

        # 1. 优先查内置字典
        if clean_name in PRESET_ADDITIVE_SMILES_DICT:
            return PRESET_ADDITIVE_SMILES_DICT[clean_name], "预置高频添加剂字典", None

        # 模糊部分匹配内置字典
        for k, v in PRESET_ADDITIVE_SMILES_DICT.items():
            if clean_name.lower() in k.lower() or k.lower() in clean_name.lower():
                return v, f"预置字典模糊匹配 ({k})", None

        # 2. 调用 PubChemPy 在线检索
        try:
            import pubchempy as pcp
            logger.info(f"正在通过 PubChem 在线检索分子: '{clean_name}'...")
            compounds = pcp.get_compounds(clean_name, "name")
            if compounds and len(compounds) > 0:
                smi = compounds[0].smiles
                if smi:
                    return smi, "PubChem 在线数据库", None
            return None, "", f"PubChem 未检索到物质 '{clean_name}'，建议直接输入其 SMILES 结构式"
        except Exception as e:
            err_msg = f"PubChem 联网检索异常或网络超时 ({str(e)})，建议手动输入 SMILES"
            logger.warning(err_msg)
            return None, "", err_msg

    @staticmethod
    def calculate_descriptors(smiles: str) -> Tuple[bool, Optional[Dict[str, float]], Optional[str]]:
        """
        使用 RDKit 计算 5 维关键化学描述符：
        - MolWt: 分子量 (g/mol)
        - TPSA: 极性表面积 (Å²)
        - LogP: 脂水分配系数
        - NumHDonors: 氢键供体数
        - NumHAcceptors: 氢键受体数
        返回: (是否成功, 描述符字典, 异常提示信息)
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return False, None, "SMILES 字符串不能为空"

        clean_smi = smiles.strip()
        try:
            mol = Chem.MolFromSmiles(clean_smi)
        except Exception as e:
            return False, None, f"RDKit 解析出现底层语法错误: {str(e)}"

        if mol is None:
            return False, None, f"无效的 SMILES 分子式 '{clean_smi}'，无法被 RDKit 解析生成分子图"

        try:
            mol_wt = float(Descriptors.MolWt(mol))
            tpsa = float(Descriptors.TPSA(mol))
            logp = float(Crippen.MolLogP(mol))
            hbd = float(Lipinski.NumHDonors(mol))
            hba = float(Lipinski.NumHAcceptors(mol))

            feats = {
                "MolWt": round(mol_wt, 3),
                "TPSA": round(tpsa, 3),
                "LogP": round(logp, 3),
                "NumHDonors": float(int(hbd)),
                "NumHAcceptors": float(int(hba))
            }
            return True, feats, None
        except Exception as e:
            return False, None, f"计算分子描述符时发生异常: {str(e)}"

    @staticmethod
    def draw_molecule_image(smiles: str, img_size: Tuple[int, int] = (300, 240)):
        """绘制 2D 分子化学结构图，返回 PIL Image 对象；若失败返回 None"""
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is not None:
                return Draw.MolToImage(mol, size=img_size)
        except Exception as e:
            logger.warning(f"RDKit 绘制分子结构图失败: {e}")
        return None


# ==============================================================================
# 3. 模块三：特征对齐与张量打包流水线 (FeatureAlignmentPipeline)
# ==============================================================================
class FeatureAlignmentPipeline:
    """
    负责将分子 5 维描述符与实验 5 维特征进行标准化对齐、归一化缩放并封装为 PyTorch Tensor [1, 10]
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def pack_sample_features(
        self,
        sample_name: str,
        smiles: str,
        mol_features: Dict[str, float],
        exp_features_raw: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        打包单个样本的特征：
        1. 验证分子特征与实验特征完备性，缺失值安全填补
        2. 实验特征通过 MinMaxScaler 归一化至 [0, 1]
        3. 横向拼接为 10 维 Float32 向量并转为 PyTorch Tensor [1, 10]
        4. 输出待分析样本的多维度画像
        """
        # 1. 整理 5 维分子特征 (未归一化，与训练一致)
        mol_vec = []
        for k in MOL_FEATURE_NAMES:
            val = mol_features.get(k, 0.0)
            mol_vec.append(float(val))
        mol_np = np.array(mol_vec, dtype=np.float32)

        # 2. 整理 5 维实验特征并填充默认值
        exp_raw_vec = []
        for k in EXP_FEATURE_NAMES:
            val = exp_features_raw.get(k, None)
            if val is None or pd.isna(val):
                val = self.db_manager.feature_medians.get(k, 0.0)
            exp_raw_vec.append(float(val))
        exp_raw_np = np.array(exp_raw_vec, dtype=np.float32).reshape(1, -1)

        # 3. 归一化实验特征 (MinMaxScaler)
        if self.db_manager.scaler is not None:
            exp_scaled_np = self.db_manager.scaler.transform(exp_raw_np).astype(np.float32).flatten()
        else:
            exp_scaled_np = exp_raw_np.flatten()

        # 4. 横向拼接为 10 维联合向量
        joint_np = np.concatenate([mol_np, exp_scaled_np], axis=0).astype(np.float32)
        
        # 5. 打包为 PyTorch FloatTensor [1, 10]
        joint_tensor = torch.tensor(joint_np.reshape(1, 10), dtype=torch.float32)

        # 6. 构建画像字典与对齐 DataFrame
        portrait_raw = {}
        for k in MOL_FEATURE_NAMES:
            portrait_raw[f"mol_{k}"] = mol_features[k]
        for k, v in zip(EXP_FEATURE_NAMES, exp_raw_vec):
            portrait_raw[f"exp_{k}"] = v

        portrait_scaled = {}
        for k in MOL_FEATURE_NAMES:
            portrait_scaled[f"mol_{k}"] = mol_features[k]
        for k, v in zip(EXP_FEATURE_NAMES, exp_scaled_np):
            portrait_scaled[f"exp_{k} (归一化)"] = round(float(v), 4)

        df_portrait = pd.DataFrame([
            {
                "特征大类": "分子描述符 (RDKit 原值)",
                "特征指标": k,
                "原始数值": mol_features[k],
                "打包张量输入值": mol_features[k],
                "量纲/说明": self._get_feature_unit(k)
            } for k in MOL_FEATURE_NAMES
        ] + [
            {
                "特征大类": "电化学/物相实验特征",
                "特征指标": k,
                "原始数值": raw_v,
                "打包张量输入值": round(float(scale_v), 4),
                "量纲/说明": f"归一化至[0,1] | 原始单位: {self._get_feature_unit(k)}"
            } for k, raw_v, scale_v in zip(EXP_FEATURE_NAMES, exp_raw_vec, exp_scaled_np)
        ])

        return {
            "sample_name": sample_name,
            "smiles": smiles,
            "raw_portrait": portrait_raw,
            "scaled_portrait": portrait_scaled,
            "feature_table": df_portrait,
            "numpy_vector": joint_np,
            "torch_tensor": joint_tensor,
            "tensor_shape": list(joint_tensor.shape),
            "tensor_dtype": str(joint_tensor.dtype),
            "feature_names": ALL_FEATURE_NAMES
        }

    @staticmethod
    def _get_feature_unit(feature_name: str) -> str:
        """获取特征单位说明"""
        units = {
            "MolWt": "g/mol (分子量)",
            "TPSA": "Å² (极性表面积)",
            "LogP": "无量纲 (脂水分配)",
            "NumHDonors": "个 (氢键供体)",
            "NumHAcceptors": "个 (氢键受体)",
            "CV峰面积": "mC (沉积剥离峰面积)",
            "Tafel斜率": "mV/dec (腐蚀动力学斜率)",
            "XPS结合能偏移": "eV (Zn 2p 偏移量)",
            "Raman峰面积拟合值": "无量纲 (水分子氢键网络结构拟合面积)",
            "XRD晶面相对强度": "无量纲 (I(002)/I(101) 晶面择优取向比)"
        }
        return units.get(feature_name, "-")

    def run_optional_model_prediction(
        self,
        tensor_10d: torch.Tensor,
        model_path: str = "/home/a1810/best_multitask_model.pth"
    ) -> Optional[Dict[str, float]]:
        """
        若本地存在已训练的 best_multitask_model.pth，执行前向推理预测：
        1. 对称电池循环寿命 (h)
        2. 库仑效率 CE (%)
        3. HER 析氢过电位 (mV)
        """
        if not os.path.exists(model_path):
            return None

        try:
            from zn_battery_multihead_model import ZnBatteryMultiHeadNet
            
            # 读取目标反归一化标定基准
            target_cols = ["循环寿命_h", "库仑效率_CE", "HER过电位_mV"]
            target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            if self.db_manager.df is not None and all(c in self.db_manager.df.columns for c in target_cols):
                target_scaler.fit(self.db_manager.df[target_cols].apply(pd.to_numeric, errors='coerce').to_numpy())
            else:
                dummy_targets = np.array([
                    [680.0, 0.9880, 140.5],
                    [1680.0, 0.9978, 225.4]
                ], dtype=np.float32)
                target_scaler.fit(dummy_targets)

            model = ZnBatteryMultiHeadNet(in_features=10)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            with torch.no_grad():
                out = model(tensor_10d)
                out_np = out.detach().cpu().numpy()
                pred_real = target_scaler.inverse_transform(out_np)[0]

            return {
                "循环寿命 (h)": float(round(pred_real[0], 1)),
                "库仑效率 CE (%)": float(round(pred_real[1] * 100, 2)),
                "HER析氢过电位 (mV)": float(round(pred_real[2], 1))
            }
        except Exception as e:
            logger.warning(f"调用预训练多任务网络预测失败: {e}")
            return None
