"""
2 mol/L 硫酸锌 (ZnSO4) 电解液实验数据库本地数据加载器 (PyTorch Dataset & DataLoader)

功能模块：
1. 输入 1：使用 RDKit 提取 SMILES 的 5 维化学描述符：
   - MolWt: 分子量 (g/mol)
   - TPSA: 拓扑极性表面积 (Å²)
   - LogP: 脂水分配系数
   - NumHDonors: 氢键供体数量 (HBD)
   - NumHAcceptors: 氢键受体数量 (HBA)

2. 输入 2：读取经 Origin 2024b 预处理的 5 维电化学与物相特征：
   - CV 峰面积 (CV peak area)
   - Tafel 斜率 (Tafel slope)
   - XPS 结合能偏移 (XPS binding energy shift)
   - Raman 峰面积拟合值 (Raman peak area fitted)
   - XRD 晶面相对强度 (XRD relative intensity, 如 I(002)/I(101))

3. 数据处理：对输入 2 (实验特征) 使用 sklearn.preprocessing.MinMaxScaler 归一化至 [0, 1]

4. 输出格式：在行方向上将 [分子特征 + 实验特征] 横向拼接，封装为标准 PyTorch FloatTensor
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# RDKit 分子化学计算模块
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ZnBatteryLoader")


# ==============================================================================
# 1. 5 维 RDKit 化学描述符提取核心函数 (输入 1)
# ==============================================================================
MOL_FEATURE_KEYS = ["MolWt", "TPSA", "LogP", "NumHDonors", "NumHAcceptors"]

def extract_smiles_descriptors(smiles: str) -> Dict[str, float]:
    """
    针对输入的添加剂 SMILES 分子式，使用 RDKit 提取以下 5 个关键物理化学描述符：
    - MolWt: 分子量 (Molecular Weight, g/mol)
    - TPSA: 拓扑极性表面积 (Topological Polar Surface Area, Å²)
    - LogP: 脂水分配系数 (Wildman-Crippen LogP)
    - NumHDonors: 氢键供体数量 (HBD)
    - NumHAcceptors: 氢键受体数量 (HBA)
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"无效的 SMILES 字符串: '{smiles}'")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"RDKit 无法解析 SMILES 分子式: '{smiles}'")

    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol))
    }


# ==============================================================================
# 2. 5 维 Origin 2024b 实验特征映射规则 (输入 2)
# ==============================================================================
# 标准特征名称与常见 CSV 列名中英文同义词字典
EXP_FEATURE_ALIASES: Dict[str, List[str]] = {
    "CV峰面积": [
        "CV峰面积", "cv峰面积", "cv_peak_area", "cv_area", "CV_peak_area",
        "cv_stripping_area_mC", "stripping_area_mC", "cv_area_mC"
    ],
    "Tafel斜率": [
        "Tafel斜率", "tafel斜率", "tafel_slope", "Tafel_slope", "tafel_slope_mV_dec",
        "tafel_anodic_slope", "tafel_cathodic_slope", "tafel_log_Icorr", "log_I_corr_A_cm2"
    ],
    "XPS结合能偏移": [
        "XPS结合能偏移", "xps结合能偏移", "xps_be_shift", "XPS_BE_shift",
        "xps_Zn2p_BE_shift_eV", "Zn2p_BE_shift_eV", "xps_binding_energy_shift"
    ],
    "Raman峰面积拟合值": [
        "Raman峰面积拟合值", "raman峰面积拟合值", "raman_peak_area_fitted",
        "raman_peak_area", "raman_fitted_area", "raman_water_hbond_area_ratio", "water_hbond_area_ratio"
    ],
    "XRD晶面相对强度": [
        "XRD晶面相对强度", "xrd晶面相对强度", "xrd_relative_intensity", "xrd_intensity_ratio",
        "xrd_I_002_to_I_101_ratio", "I_002_to_I_101_ratio", "xrd_TC_002", "TC_002"
    ]
}


def resolve_experimental_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    智能匹配 CSV 中对应的 5 大 Origin 实验特征列名
    """
    resolved_cols: Dict[str, str] = {}
    df_cols = list(df.columns)

    for target_name, candidates in EXP_FEATURE_ALIASES.items():
        matched_col = None
        for cand in candidates:
            for col in df_cols:
                if col.strip().lower() == cand.strip().lower():
                    matched_col = col
                    break
            if matched_col is not None:
                break
        
        if matched_col is None:
            # 模糊匹配包含关键词
            for col in df_cols:
                col_clean = col.strip().lower()
                if any(cand.lower() in col_clean for cand in candidates):
                    matched_col = col
                    break

        if matched_col is None:
            raise KeyError(
                f"在 CSV 文件中未找到必需的 Origin 实验特征列: '{target_name}'。\n"
                f"CSV 当前列名包含: {df_cols}。\n"
                f"支持的匹配别名: {candidates}"
            )
        resolved_cols[target_name] = matched_col

    return resolved_cols


# ==============================================================================
# 3. 本地数据加载器数据集定义 (ZnBatteryDataset)
# ==============================================================================
class ZnBatteryDataset(Dataset):
    """
    2 mol/L 硫酸锌电解液实验数据库 PyTorch 数据集类。

    参数:
        csv_file_or_df: CSV 文件路径或已读取的 pandas.DataFrame
        smiles_col: SMILES 所在列名 (默认自动检测 'smiles' 或 'SMILES')
        id_col: 样本 ID 所在列名 (默认自动检测 'sample_id' 或 'id')
        scaler: 可选传入预训练好的 MinMaxScaler 实例；若为 None 则在当前数据上 fit_transform
    """
    def __init__(
        self,
        csv_file_or_df: Union[str, pd.DataFrame],
        smiles_col: Optional[str] = None,
        id_col: Optional[str] = None,
        scaler: Optional[MinMaxScaler] = None
    ):
        super().__init__()

        # 1. 加载 DataFrame
        if isinstance(csv_file_or_df, str):
            if not os.path.exists(csv_file_or_df):
                raise FileNotFoundError(f"未找到实验 CSV 数据库文件: '{csv_file_or_df}'")
            logger.info(f"正在从本地读取硫酸锌实验 CSV 数据库: {csv_file_or_df}")
            self.df = pd.read_csv(csv_file_or_df)
        elif isinstance(csv_file_or_df, pd.DataFrame):
            self.df = csv_file_or_df.copy()
        else:
            raise TypeError("csv_file_or_df 必须是 CSV 路径字符串或 pandas.DataFrame")

        if self.df.empty:
            raise ValueError("输入的实验数据集为空！")

        # 2. 识别主键 ID 列与 SMILES 列
        self.id_col = id_col or self._find_column(["sample_id", "sample", "id", "样本编号", "样本ID"], default="sample_id")
        self.smiles_col = smiles_col or self._find_column(["smiles", "SMILES", "Smiles", "结构式", "分子式"], default="smiles")

        if self.smiles_col not in self.df.columns:
            raise KeyError(f"CSV 数据表中缺少 SMILES 列 '{self.smiles_col}'，当前列: {list(self.df.columns)}")

        # 提取 sample_ids
        if self.id_col in self.df.columns:
            self.sample_ids = self.df[self.id_col].astype(str).tolist()
        else:
            self.sample_ids = [f"Sample_{i+1:03d}" for i in range(len(self.df))]

        self.smiles_list = self.df[self.smiles_col].astype(str).tolist()

        # ----------------------------------------------------------------------
        # 输入 1 处理：使用 RDKit 提取 SMILES 5 维分子特征
        # ----------------------------------------------------------------------
        logger.info(f"正在使用 RDKit 提取 {len(self.df)} 个样本的分子物理化学描述符 (MolWt, TPSA, LogP, NumHDonors, NumHAcceptors)...")
        mol_feature_rows = []
        for smi in self.smiles_list:
            feats = extract_smiles_descriptors(smi)
            mol_feature_rows.append([
                feats["MolWt"],
                feats["TPSA"],
                feats["LogP"],
                feats["NumHDonors"],
                feats["NumHAcceptors"]
            ])

        self.mol_feature_names = [f"mol_{k}" for k in MOL_FEATURE_KEYS]
        self.mol_features_np = np.array(mol_feature_rows, dtype=np.float32)
        self.mol_tensor = torch.tensor(self.mol_features_np, dtype=torch.float32)

        # ----------------------------------------------------------------------
        # 输入 2 处理：读取 5 维 Origin 2024b 电化学与物相特征
        # ----------------------------------------------------------------------
        self.resolved_exp_cols = resolve_experimental_columns(self.df)
        self.exp_standard_names = list(self.resolved_exp_cols.keys())
        self.exp_actual_cols = [self.resolved_exp_cols[k] for k in self.exp_standard_names]

        logger.info("已成功匹配并读取 5 大 Origin 2024b 预处理实验指标:")
        for std_name, act_col in self.resolved_exp_cols.items():
            logger.info(f"  - [{std_name}] <- CSV 列: '{act_col}'")

        df_exp_raw = self.df[self.exp_actual_cols].apply(pd.to_numeric, errors='coerce')
        if df_exp_raw.isnull().any().any():
            logger.warning("检测到实验特征中存在缺失值，采用列中位数进行安全填充...")
            df_exp_raw = df_exp_raw.fillna(df_exp_raw.median())

        self.exp_raw_np = df_exp_raw.to_numpy(dtype=np.float32)

        # ----------------------------------------------------------------------
        # 处理：对输入 2 (实验特征) 使用 MinMaxScaler 进行归一化至 [0, 1]
        # ----------------------------------------------------------------------
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            self.exp_scaled_np = self.scaler.fit_transform(self.exp_raw_np).astype(np.float32)
            logger.info("对输入 2 (Origin 实验特征) 执行 MinMaxScaler 拟合并归一化至 [0.0, 1.0]。")
        else:
            self.scaler = scaler
            self.exp_scaled_np = self.scaler.transform(self.exp_raw_np).astype(np.float32)
            logger.info("使用传入的预置 MinMaxScaler 对输入 2 执行归一化。")

        self.exp_feature_names = [f"exp_{k}" for k in self.exp_standard_names]
        self.exp_tensor_scaled = torch.tensor(self.exp_scaled_np, dtype=torch.float32)

        # ----------------------------------------------------------------------
        # 输出：拼接为 [分子特征 + 实验特征] 的联合 PyTorch 张量 (Joint Tensor)
        # ----------------------------------------------------------------------
        # 在行方向 (每行样本维度) 上横向拼接: (N, 5) + (N, 5) -> (N, 10)
        self.joint_tensor = torch.cat([self.mol_tensor, self.exp_tensor_scaled], dim=-1)
        self.all_feature_names = self.mol_feature_names + self.exp_feature_names

        logger.info(
            f"数据集构建完成！联合特征张量尺寸: {tuple(self.joint_tensor.shape)} "
            f"(样本数 N={self.joint_tensor.shape[0]}, 特征总维度 D={self.joint_tensor.shape[1]})"
        )

    def _find_column(self, candidates: List[str], default: str) -> str:
        for cand in candidates:
            for col in self.df.columns:
                if col.strip().lower() == cand.strip().lower():
                    return col
        return default

    def __len__(self) -> int:
        return len(self.joint_tensor)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回单个样本的数据字典，可直接被 PyTorch DataLoader 批处理整理为 Batch 张量
        """
        return {
            "joint_features": self.joint_tensor[idx],               # 联合张量 [10 维]
            "mol_features": self.mol_tensor[idx],                   # 输入 1: RDKit 分子特征 [5 维]
            "exp_features_scaled": self.exp_tensor_scaled[idx],     # 输入 2: MinMaxScaler 归一化实验特征 [5 维]
            "sample_id": self.sample_ids[idx],                      # 样本唯一标识
            "smiles": self.smiles_list[idx]                         # 原始 SMILES
        }

    def get_joint_tensor(self) -> torch.Tensor:
        """直接获取完整数据集的联合 PyTorch 张量"""
        return self.joint_tensor

    def get_feature_names(self) -> List[str]:
        """获取联合特征中所有维度的名称列表 (共 10 维)"""
        return self.all_feature_names

    def get_summary_dataframe(self) -> pd.DataFrame:
        """导出包含原始数据、提取分子特征与归一化实验特征的汇总 DataFrame"""
        summary_df = pd.DataFrame(
            self.joint_tensor.numpy(),
            columns=self.all_feature_names
        )
        summary_df.insert(0, "smiles", self.smiles_list)
        summary_df.insert(0, "sample_id", self.sample_ids)
        return summary_df


# ==============================================================================
# 4. DataLoader 工厂函数
# ==============================================================================
def create_zn_battery_dataloader(
    csv_file_or_df: Union[str, pd.DataFrame],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False
) -> Tuple[DataLoader, ZnBatteryDataset]:
    """
    便捷创建 PyTorch 本地数据加载器 (DataLoader)
    """
    dataset = ZnBatteryDataset(csv_file_or_df=csv_file_or_df)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return dataloader, dataset


# ==============================================================================
# 5. 测试与演示脚本入口
# ==============================================================================
if __name__ == "__main__":
    print("=" * 85)
    print(">>> 2 mol/L 硫酸锌电解液实验数据库 PyTorch 本地数据加载器 (ZnBatteryLoader) 测试 <<<")
    print("=" * 85)

    csv_path = "/home/a1810/zn_battery_experiment_database.csv"

    # 1. 实例化 Dataset 并创建 DataLoader
    batch_size = 4
    dataloader, dataset = create_zn_battery_dataloader(
        csv_file_or_df=csv_path,
        batch_size=batch_size,
        shuffle=False
    )

    # 2. 打印数据集全局特征概览
    joint_tensor = dataset.get_joint_tensor()
    feature_names = dataset.get_feature_names()

    print("\n[1. 联合特征张量全局信息]:")
    print(f"  - 全局张量形状 (Tensor Shape) : {tuple(joint_tensor.shape)} -> (N={joint_tensor.shape[0]} 样本, D={joint_tensor.shape[1]} 特征)")
    print(f"  - 张量数据类型 (Dtype)        : {joint_tensor.dtype}")
    print(f"  - 张量存储设备 (Device)       : {joint_tensor.device}")

    print("\n[2. 特征维度对应明细 (前 5 维为分子特征，后 5 维为归一化实验特征)]:")
    for i, name in enumerate(feature_names):
        feature_type = "[输入 1: RDKit 分子特征]" if i < 5 else "[输入 2: MinMaxScaler 实验特征]"
        print(f"  维度 {i:02d}: {feature_type:<24} -> {name}")

    # 3. 打印完整张量数值矩阵
    print("\n[3. 完整联合特征张量数值矩阵]:")
    print(joint_tensor)

    # 4. 模拟 PyTorch 训练时的 DataLoader 批次迭代
    print("\n" + "=" * 85)
    print(f">>> [4. 验证 PyTorch DataLoader 批次迭代 (Batch Size = {batch_size})] <<<")
    print("=" * 85)

    for batch_idx, batch in enumerate(dataloader):
        b_joint = batch["joint_features"]
        b_mol = batch["mol_features"]
        b_exp = batch["exp_features_scaled"]
        b_ids = batch["sample_id"]

        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"  样本 ID 列表              : {b_ids}")
        print(f"  联合特征批次张量 (Shape)  : {tuple(b_joint.shape)}")
        print(f"  输入 1 分子特征张量 (Shape): {tuple(b_mol.shape)}")
        print(f"  输入 2 实验特征张量 (Shape): {tuple(b_exp.shape)}")
        print(f"  Batch 内首个样本联合特征切片:")
        print(f"    - 样本 ID: {b_ids[0]}")
        print(f"    - 特征数值: {np.round(b_joint[0].numpy(), 4).tolist()}")

    # 5. 展示特征汇总表格
    print("\n" + "=" * 85)
    print(">>> [5. 数据集整理后的特征汇总表 (前 3 行预览)] <<<")
    print("=" * 85)
    summary_df = dataset.get_summary_dataframe()
    print(summary_df.head(3).to_string())

    print("\n" + "=" * 85)
    print(">>> 数据加载器全部功能验证通过！<<<")
    print("=" * 85)
