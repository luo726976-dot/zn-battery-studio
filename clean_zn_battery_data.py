"""
2 mol/L 硫酸锌 (ZnSO4) 电池多模态特征工程与清洗融合流水线 (第二阶段)
核心任务：
1. 基于 RDKit 的候选电解液添加剂分子化学描述符提取 (MolWt, TPSA, LogP, 氢键供体数, 氢键受体数)
2. 行级 (Row-wise/Sample-wise) 纵深对齐与特征维度横向无缝融合 (分子特征 + Origin 实验特征)
3. 构建标准 PyTorch 联合特征张量 (Joint Feature Tensor)，直通后续 4 层前馈神经网络 (FCNN/MLP)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# RDKit 分子化学计算库
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ZnBatteryFeatureEngineering")


# ==============================================================================
# 1. 典型锌电池候选添加剂 SMILES 预置库 (含 2-氨基-4-溴蒽醌-2-磺酸钠 与 甘氨酸)
# ==============================================================================
CANDIDATE_ADDITIVES: Dict[str, str] = {
    # 核心目标添加剂 1: 2-氨基-4-溴蒽醌-2-磺酸钠 (溴氨酸钠)
    "2-氨基-4-溴蒽醌-2-磺酸钠": "Nc1c(S(=O)(=O)[O-])cc(Br)c2c1C(=O)c1ccccc1C2=O.[Na+]",
    # 核心目标添加剂 2: 甘氨酸 (双极性两性离子型氨基酸添加剂)
    "甘氨酸": "NCC(=O)O",
    # 对比/候选添加剂
    "柠檬酸": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    "硫脲": "NC(=S)N",
    "D-葡萄糖": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "L-抗坏血酸": "C1=C(C(=O)OC1C(CO)O)O",
    "聚乙二醇二甲醚单体": "COCCOC",
    "无水烟酰胺": "NC(=O)c1cccnc1"
}


# ==============================================================================
# 2. RDKit 分子化学描述符提取函数 (MolWt, TPSA, LogP, HBD, HBA)
# ==============================================================================
def extract_rdkit_molecular_features(mol_input: Union[str, Chem.Mol]) -> Dict[str, float]:
    """
    提取单个分子的 5 项关键物理化学描述符：
    1. MolWt: 分子量 (Molecular Weight, g/mol)
    2. TPSA: 拓扑极性表面积 (Topological Polar Surface Area, Å²)
    3. LogP: 脂水分配系数 (Wildman-Crippen LogP)
    4. NumHDonors: 氢键供体数量 (HBD)
    5. NumHAcceptors: 氢键受体数量 (HBA)

    参数:
        mol_input: 分子中文名、SMILES 字符串或 RDKit Chem.Mol 实例

    返回:
        Dict[str, float]: 包含 5 项化学描述符的字典
    """
    if isinstance(mol_input, Chem.Mol):
        mol = mol_input
    elif isinstance(mol_input, str):
        # 兼容预置添加剂名称与原始 SMILES 字符串
        smiles = CANDIDATE_ADDITIVES.get(mol_input, mol_input)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法解析分子或 SMILES 分子式: '{mol_input}'")
    else:
        raise TypeError(f"输入类型错误，期望 str 或 Chem.Mol，实际为: {type(mol_input)}")

    features = {
        "mol_MolWt": float(Descriptors.MolWt(mol)),
        "mol_TPSA": float(Descriptors.TPSA(mol)),
        "mol_LogP": float(Crippen.MolLogP(mol)),
        "mol_NumHDonors": float(Lipinski.NumHDonors(mol)),
        "mol_NumHAcceptors": float(Lipinski.NumHAcceptors(mol))
    }
    return features


def extract_batch_molecular_features(
    additive_data: Union[Dict[str, str], pd.DataFrame],
    id_col: str = "sample_id",
    smiles_col: str = "smiles"
) -> pd.DataFrame:
    """
    批量提取添加剂样本集的 RDKit 分子特征并构建结构化 DataFrame
    """
    rows = []
    if isinstance(additive_data, dict):
        for sid, smi in additive_data.items():
            feats = extract_rdkit_molecular_features(smi)
            feats[id_col] = sid
            feats["smiles"] = smi
            rows.append(feats)
    elif isinstance(additive_data, pd.DataFrame):
        for _, row in additive_data.iterrows():
            sid = str(row[id_col])
            smi = str(row[smiles_col])
            feats = extract_rdkit_molecular_features(smi)
            feats[id_col] = sid
            if "additive_name" in row:
                feats["additive_name"] = row["additive_name"]
            rows.append(feats)
    else:
        raise TypeError("additive_data 必须为 Dict[sample_id, smiles] 或 DataFrame")

    df_mol = pd.DataFrame(rows)
    # 将 sample_id 作为主键首列
    cols = [id_col] + [c for c in df_mol.columns if c != id_col]
    return df_mol[cols]


# ==============================================================================
# 3. 多模态特征融合器：分子特征 + Origin 实验特征无缝拼接并生成 PyTorch 张量
# ==============================================================================
class MultimodalBatteryFeaturePipeline:
    """
    多模态特征工程管道：
    负责将 RDKit 分子特征与 Origin 电化学特征按样本行严格对齐并横向拼接融合，
    输出适合 4 层神经网络输入的最终联合 PyTorch FloatTensor。
    """
    def __init__(
        self,
        id_column: str = "sample_id",
        feature_range: Tuple[float, float] = (0.0, 1.0)
    ):
        self.id_column = id_column
        self.feature_range = feature_range
        self.mol_scaler = MinMaxScaler(feature_range=feature_range)
        self.exp_scaler = MinMaxScaler(feature_range=feature_range)

        self.mol_feature_names: List[str] = [
            "mol_MolWt", "mol_TPSA", "mol_LogP", "mol_NumHDonors", "mol_NumHAcceptors"
        ]
        self.exp_feature_names: List[str] = []
        self.all_feature_names: List[str] = []
        self.fused_sample_ids: List[str] = []

    def fuse_and_build_tensor(
        self,
        df_molecular: pd.DataFrame,
        df_experimental: pd.DataFrame
    ) -> Tuple[torch.Tensor, pd.DataFrame]:
        """
        核心融合逻辑：
        1. 基于 sample_id 进行样本行严格交集对齐 (Inner Join / Row-wise Alignment)
        2. 对分子特征与实验特征分别进行独立的 Min-Max 标准化，避免大数值特征压制小数值
        3. 在行方向 (每个样本行) 上横向无缝拼接: [X_mol (5维), X_exp (M维)]
        4. 构建并封装为标准 PyTorch FloatTensor: 形状为 (N_samples, 5 + M)
        """
        logger.info(">>> 开始执行行级多模态特征融合 (Molecular + Experimental Fusion) <<<")

        # 统一索引为主键列
        df_mol = df_molecular.set_index(self.id_column) if self.id_column in df_molecular.columns else df_molecular.copy()
        df_exp = df_experimental.set_index(self.id_column) if self.id_column in df_experimental.columns else df_experimental.copy()

        # 仅保留数值列
        df_mol_num = df_mol[self.mol_feature_names].apply(pd.to_numeric, errors='coerce')
        df_exp_num = df_exp.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1)

        # 1. 样本行级严格交集对齐
        common_samples = df_mol_num.index.intersection(df_exp_num.index)
        if len(common_samples) == 0:
            raise ValueError("分子特征与实验数据没有共同的 sample_id，无法按行对齐！")

        df_mol_aligned = df_mol_num.loc[common_samples]
        df_exp_aligned = df_exp_num.loc[common_samples]
        self.fused_sample_ids = common_samples.tolist()
        self.exp_feature_names = df_exp_aligned.columns.tolist()
        self.all_feature_names = self.mol_feature_names + self.exp_feature_names

        logger.info(f"成功对齐 {len(common_samples)} 个独立样本。")

        # 2. 独立归一化处理 (平衡分子结构参数与电化学电压/电流等各模态量纲)
        X_mol_scaled = self.mol_scaler.fit_transform(df_mol_aligned).astype(np.float32)
        X_exp_scaled = self.exp_scaler.fit_transform(df_exp_aligned).astype(np.float32)

        # 3. 特征维度横向无缝拼接: (N, 5) 拼接 (N, M) -> (N, 5 + M)
        X_fused = np.concatenate([X_mol_scaled, X_exp_scaled], axis=1)

        df_fused = pd.DataFrame(
            X_fused,
            index=self.fused_sample_ids,
            columns=self.all_feature_names
        )

        # 4. 生成标准 PyTorch Tensor
        joint_tensor = torch.tensor(X_fused, dtype=torch.float32)

        logger.info("联合特征张量构建完成:")
        logger.info(f" - 分子化学特征 ({len(self.mol_feature_names)} 维): {self.mol_feature_names}")
        logger.info(f" - 电化学实验特征 ({len(self.exp_feature_names)} 维): {self.exp_feature_names}")
        logger.info(f" - 张量尺寸 (Tensor Shape): {tuple(joint_tensor.shape)} -> (样本数 N, 特征总维度 D)")

        return joint_tensor, df_fused


# ==============================================================================
# 4. 后续 4 层前馈神经网络结构 (ZnBattery4LayerMLP)
# ==============================================================================
class ZnBattery4LayerMLP(nn.Module):
    """
    接收 [分子特征 + 实验特征] 联合输入张量的 4 层全连接神经网络 (FCNN)
    网络架构: Input(D) -> FC(64) -> BatchNorm1d -> ReLU -> Dropout -> FC(32) -> ReLU -> FC(16) -> ReLU -> FC(1)
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 输入变换与高维投射
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Layer 2: 特征非线性交互表征
            nn.Linear(64, 32),
            nn.ReLU(),
            # Layer 3: 隐空间降维抽象
            nn.Linear(32, 16),
            nn.ReLU(),
            # Layer 4: 性能回归预测层 (例如预测沉积过电位/循环寿命/库伦效率)
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# 5. 模拟实验数据生成器 (Mock Origin 2024b 电化学特征: CV, Tafel, XPS 等)
# ==============================================================================
def create_mock_data(n_samples: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成用于验证特征工程流水线的 Mock 数据：
    包含 2-氨基-4-溴蒽醌-2-磺酸钠、甘氨酸等关键候选添加剂及多模态实验指标
    """
    np.random.seed(2024)
    target_names = [
        "2-氨基-4-溴蒽醌-2-磺酸钠",
        "甘氨酸",
        "柠檬酸",
        "硫脲",
        "D-葡萄糖",
        "无水烟酰胺"
    ][:n_samples]

    samples_info = []
    exp_records = []

    for i, name in enumerate(target_names):
        sid = f"Zn_Sample_{i+1:02d}_{name[:4]}"
        smiles = CANDIDATE_ADDITIVES[name]
        samples_info.append({
            "sample_id": sid,
            "additive_name": name,
            "smiles": smiles
        })

        # 模拟 Origin 2024b 处理得到的实验特征 (CV, Tafel, XPS, Raman 等)
        exp_records.append({
            "sample_id": sid,
            # CV 特征: 峰电位差 (V), 库伦效率, 剥离电量 (mC)
            "cv_delta_Ep_V": float(np.random.normal(0.085, 0.015)),
            "cv_coulombic_eff": float(np.random.normal(0.975, 0.015)),
            "cv_stripping_charge_mC": float(np.random.normal(3200.0, 300.0)),
            # Tafel 特征: 腐蚀电位 (V), 腐蚀电流对数, 极化电阻 (Ω)
            "tafel_Ecorr_V": float(np.random.normal(-0.015, 0.005)),
            "tafel_log_Icorr": float(np.random.normal(-3.85, 0.25)),
            "tafel_Rp_ohm": float(np.random.normal(480.0, 50.0)),
            # XPS 特征: Zn 2p 结合能位移 (eV), 碱式硫酸锌 (ZHS) 面积比
            "xps_Zn2p_BE_shift_eV": float(np.random.normal(0.42, 0.08)),
            "xps_ZHS_ratio": float(np.random.normal(1.75, 0.30)),
            # Raman 特征: 硫酸根 v1 振动峰宽 (cm-1), 结合水/活性水比率
            "raman_SO4_FWHM_cm1": float(np.random.normal(11.5, 1.2)),
            "raman_bound_water_ratio": float(np.random.normal(0.65, 0.06))
        })

    df_additives = pd.DataFrame(samples_info)
    df_experimental = pd.DataFrame(exp_records)
    return df_additives, df_experimental


# ==============================================================================
# 6. 主执行入口：特征提取 -> 融合拼接 -> 生成张量 -> 送入 4 层 MLP 前向测试
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(">>> 硫酸锌 (ZnSO4) 电池分子化学与电化学多模态特征融合流水线 <<<")
    print("=" * 80)

    # 1. 独立验证核心添加剂 (2-氨基-4-溴蒽醌-2-磺酸钠、甘氨酸) 的 RDKit 特征提取
    print("\n[Step 1: RDKit 候选添加剂分子描述符提取测试]")
    test_molecules = ["2-氨基-4-溴蒽醌-2-磺酸钠", "甘氨酸"]
    for mol_name in test_molecules:
        smi = CANDIDATE_ADDITIVES[mol_name]
        feats = extract_rdkit_molecular_features(smi)
        print(f"\n  * 分子名称: {mol_name}")
        print(f"    SMILES  : {smi}")
        print(f"    - 分子量 (MolWt)              : {feats['mol_MolWt']:.4f} g/mol")
        print(f"    - 拓扑极性表面积 (TPSA)       : {feats['mol_TPSA']:.2f} Å²")
        print(f"    - 脂水分配系数 (LogP)         : {feats['mol_LogP']:.2f}")
        print(f"    - 氢键供体数 (NumHDonors)     : {int(feats['mol_NumHDonors'])}")
        print(f"    - 氢键受体数 (NumHAcceptors)  : {int(feats['mol_NumHAcceptors'])}")

    # 2. 生成 Mock 模拟数据集进行端到端跑通
    print("\n[Step 2: 构建样本集并提取多模态特征]")
    df_additives, df_experimental = create_mock_data(n_samples=6)
    print(f"已生成 {len(df_additives)} 个测试样本的信息与对应电化学实验指标。")

    # 批量提取分子特征
    df_molecular = extract_batch_molecular_features(df_additives)

    # 3. 特征融合与 PyTorch 联合张量构建
    print("\n[Step 3: 执行行级严格对齐与横向特征拼接]")
    pipeline = MultimodalBatteryFeaturePipeline(id_column="sample_id")
    joint_tensor, df_fused = pipeline.fuse_and_build_tensor(df_molecular, df_experimental)

    # 4. 打印最终联合特征张量
    print("\n" + "=" * 80)
    print(">>> [最终输出] 联合特征张量 (Joint Feature Tensor) 详细信息 <<<")
    print("=" * 80)
    print(f"Tensor 形状 (Shape) : {tuple(joint_tensor.shape)}  [样本数 N={joint_tensor.shape[0]}, 特征总数 D={joint_tensor.shape[1]}]")
    print(f"Tensor 数据类型     : {joint_tensor.dtype}")
    print(f"Tensor 设备位置     : {joint_tensor.device}")
    print(f"特征数值区间        : [{joint_tensor.min().item():.4f}, {joint_tensor.max().item():.4f}]")
    print("\n特征排列顺序 (前 5 项为分子特征，后续为电化学实验特征):")
    for idx, col in enumerate(pipeline.all_feature_names):
        prefix = "[分子特征]" if idx < 5 else "[实验特征]"
        print(f"  Col {idx:02d}: {prefix:<8} {col}")

    print("\n联合特征张量完整数值矩阵 (Printed Tensor):")
    print(joint_tensor)

    # 5. 验证联合张量直接送入 4 层神经网络 (ZnBattery4LayerMLP)
    print("\n" + "=" * 80)
    print(">>> [Step 4: 神经网络前向推理验证] 直通 4 层前馈神经网络 (FCNN) <<<")
    print("=" * 80)
    model = ZnBattery4LayerMLP(in_features=joint_tensor.shape[1])
    model.eval()

    with torch.no_grad():
        output = model(joint_tensor)

    print("网络架构:\n", model)
    print(f"\n输入张量尺寸 -> {tuple(joint_tensor.shape)}")
    print(f"输出预测尺寸 -> {tuple(output.shape)}")
    print("各样本网络前向预测输出 (如电化学性能回归值):")
    for i in range(len(output)):
        sid = pipeline.fused_sample_ids[i]
        print(f"  样本 [{sid}]: 预测值 = {output[i].item():.4f}")

    print("\n" + "=" * 80)
    print(">>> 第二阶段特征工程流水线全部成功跑通！<<<")
    print("=" * 80)
