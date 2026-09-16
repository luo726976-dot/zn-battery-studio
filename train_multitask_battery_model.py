"""
2 mol/L 硫酸锌电池多任务前馈神经网络 (ZnBatteryMultiHeadNet) 训练脚本

功能模块：
1. 数据输入：读取 2 mol/L 硫酸锌实验数据库，构建 10 维 [分子特征 + 实验特征] 联合张量
2. 目标构建：3 个关键任务标签：
   - 任务 1: 对称电池循环寿命 (Cycle Life, h)
   - 任务 2: 库仑效率 (Coulombic Efficiency, CE)
   - 任务 3: 析氢反应 (HER) 过电位 (Overpotential, mV)
3. 损失函数：分别计算 3 个任务的 MSELoss，并加权求和：
   Total Loss = a * Loss_循环 + b * Loss_CE + c * Loss_HER
4. 优化器：使用 Adam，学习率设定为 1e-3 (1e-3)
5. 验证机制与 Early Stopping：按 8:2 划分训练集与验证集。若验证集总 Loss 连续 15 个 epoch 未下降，触发 Early Stopping
6. 模型保存：训练完毕后将最优模型权重保存为 best_multitask_model.pth
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler

# 导入前两步编写的数据加载器与多任务网络结构
from zn_battery_loader import ZnBatteryDataset
from zn_battery_multihead_model import ZnBatteryMultiHeadNet


# ==============================================================================
# 0. 固定全局随机种子 (保证 8:2 划分及网络权重的完全可复现性)
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. 实验数据库目标标签提取与预处理
# ==============================================================================
def prepare_multitask_data(
    csv_path: str
) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler, List[str], List[str], pd.DataFrame]:
    """
    加载数据库并返回联合特征张量 X (10 维) 与归一化多任务标签 Y (3 维) 及原始 DataFrame
    """
    # 1. 构建 Dataset，自动提取分子 5 维描述符与归一化 5 维实验特征
    dataset = ZnBatteryDataset(csv_file_or_df=csv_path)
    X_tensor = dataset.get_joint_tensor()  # (N, 10)
    df = dataset.df
    sample_ids = dataset.sample_ids
    feature_names = dataset.get_feature_names()

    # 2. 提取 3 个多任务预测目标
    target_cols = ["循环寿命_h", "库仑效率_CE", "HER过电位_mV"]
    for col in target_cols:
        if col not in df.columns:
            raise KeyError(f"CSV 数据库中缺少必需的目标列: '{col}'")

    y_raw = df[target_cols].to_numpy(dtype=np.float32)

    # 3. 对 3 个目标进行独立 MinMax 归一化至 [0, 1]，使 MSELoss 在相同量级稳定计算
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    y_scaled = target_scaler.fit_transform(y_raw).astype(np.float32)
    Y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    return X_tensor, Y_tensor, target_scaler, sample_ids, feature_names, df


# ==============================================================================
# 2. 多任务加权 MSE 损失函数
# ==============================================================================
class WeightedMultiTaskMSELoss(nn.Module):
    """
    多任务加权均方误差损失函数：
    总 Loss = a * Loss_循环 + b * Loss_CE + c * Loss_HER
    """
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.mse = nn.MSELoss()

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        :param preds: (batch_size, 3) 模型多任务预测输出
        :param targets: (batch_size, 3) 多任务目标标签
        :return: (加权总损失张量, 各分项损失字典)
        """
        loss_lifespan = self.mse(preds[:, 0], targets[:, 0])
        loss_ce = self.mse(preds[:, 1], targets[:, 1])
        loss_her = self.mse(preds[:, 2], targets[:, 2])

        total_loss = (
            self.a * loss_lifespan +
            self.b * loss_ce +
            self.c * loss_her
        )

        loss_breakdown = {
            "loss_lifespan": float(loss_lifespan.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_her": float(loss_her.item()),
            "total_loss": float(total_loss.item())
        }
        return total_loss, loss_breakdown


# ==============================================================================
# 3. 完整训练循环 (含 8:2 划分、Adam 1e-3、15 轮 Early Stopping 与权重持久化)
# ==============================================================================
def train_multitask_model(
    csv_path: str = "/home/a1810/zn_battery_experiment_database.csv",
    save_model_path: str = "/home/a1810/best_multitask_model.pth",
    weight_lifespan: float = 1.0,
    weight_ce: float = 1.0,
    weight_her: float = 1.0,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 4
) -> Tuple[ZnBatteryMultiHeadNet, Dict[str, List[float]]]:
    """
    多任务前馈神经网络完整训练流程
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备数据
    X_tensor, Y_tensor, target_scaler, sample_ids, feat_names, df = prepare_multitask_data(csv_path)
    full_dataset = TensorDataset(X_tensor, Y_tensor)
    n_samples = len(full_dataset)

    # 2. 按 8:2 划分训练与验证集
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("\n" + "=" * 80)
    print(">>> 硫酸锌电解液多任务神经网络训练启动 (8:2 划分 & Early Stopping) <<<")
    print("=" * 80)
    print(f"数据总样本数: {n_samples}")
    print(f" - 训练集规模 (80%): {train_size} 样本")
    print(f" - 验证集规模 (20%): {val_size} 样本")
    print(f"输入特征维度: {X_tensor.shape[1]} 维 -> 输出任务数: {Y_tensor.shape[1]} 维")
    print(f"加权损失权重: a(循环寿命)={weight_lifespan}, b(库仑效率)={weight_ce}, c(HER过电位)={weight_her}")
    print(f"优化器参数  : Adam (lr = {lr})")
    print(f"早停容忍度  : {patience} 个连续 epoch 无改善触发 Early Stopping")
    print("=" * 80)

    # 3. 实例化多任务网络
    model = ZnBatteryMultiHeadNet(in_features=X_tensor.shape[1]).to(device)

    # 4. 初始化损失函数与 Adam 优化器 (lr=1e-3)
    criterion = WeightedMultiTaskMSELoss(
        a=weight_lifespan,
        b=weight_ce,
        c=weight_her
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Early Stopping 状态追踪
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_loss_lifespan": [],
        "val_loss_ce": [],
        "val_loss_her": []
    }

    print("\n" + "-" * 88)
    print(f"{'Epoch':^7} | {'Train Total':^13} | {'Val Total':^13} | {'Val-循环':^11} | {'Val-CE':^11} | {'Val-HER':^11} | {'状态'}")
    print("-" * 88)

    # 6. Epoch 训练主循环
    for epoch in range(1, max_epochs + 1):
        # --------------------- 训练阶段 ---------------------
        model.train()
        running_train_loss = 0.0
        train_count = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            preds = model(bx)
            loss, _ = criterion(preds, by)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * bx.size(0)
            train_count += bx.size(0)

        epoch_train_loss = running_train_loss / train_count
        history["train_loss"].append(epoch_train_loss)

        # --------------------- 验证阶段 ---------------------
        model.eval()
        running_val_loss = 0.0
        running_val_lifespan = 0.0
        running_val_ce = 0.0
        running_val_her = 0.0
        val_count = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                val_loss, breakdown = criterion(preds, by)

                running_val_loss += val_loss.item() * bx.size(0)
                running_val_lifespan += breakdown["loss_lifespan"] * bx.size(0)
                running_val_ce += breakdown["loss_ce"] * bx.size(0)
                running_val_her += breakdown["loss_her"] * bx.size(0)
                val_count += bx.size(0)

        epoch_val_loss = running_val_loss / val_count
        epoch_val_life = running_val_lifespan / val_count
        epoch_val_ce = running_val_ce / val_count
        epoch_val_her = running_val_her / val_count

        history["val_loss"].append(epoch_val_loss)
        history["val_loss_lifespan"].append(epoch_val_life)
        history["val_loss_ce"].append(epoch_val_ce)
        history["val_loss_her"].append(epoch_val_her)

        # ---------------- Early Stopping 逻辑判定 ----------------
        status_msg = ""
        # 严格下降判定 (保留微小数值容限 1e-5)
        if epoch_val_loss < (best_val_loss - 1e-5):
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            # 及时将最优模型权重持久化保存到目标路径
            torch.save(model.state_dict(), save_model_path)
            status_msg = f"* 最优模型已保存 (Loss: {best_val_loss:.5f}) *"
        else:
            patience_counter += 1
            status_msg = f"未下降 [{patience_counter}/{patience}]"

        # 阶段性打印日志
        if epoch == 1 or epoch % 5 == 0 or patience_counter >= patience or "最优" in status_msg:
            print(
                f"{epoch:^7d} | {epoch_train_loss:^13.5f} | {epoch_val_loss:^13.5f} | "
                f"{epoch_val_life:^11.5f} | {epoch_val_ce:^11.5f} | {epoch_val_her:^11.5f} | {status_msg}"
            )

        # 触发早停机制
        if patience_counter >= patience:
            print("-" * 88)
            print(f">>> [Early Stopping 触发] 验证集总 Loss 已连续 {patience} 个 Epoch 未下降，训练提前终止！")
            break

    print("-" * 88)
    print(f"训练流程结束！")
    print(f" - 最优验证 Epoch: 第 {best_epoch} 轮")
    print(f" - 最低验证总 Loss: {best_val_loss:.6f}")
    print(f" - 权重已保存至: {save_model_path}")
    print("=" * 80)

    # 7. 加载并验证已保存的最优模型
    best_model = ZnBatteryMultiHeadNet(in_features=X_tensor.shape[1]).to(device)
    best_model.load_state_dict(torch.load(save_model_path))
    best_model.eval()

    # 8. 在验证集样本上进行预测比对展示
    print("\n>>> [验证集最优预测展示 (反归一化物理单位)] <<<")
    val_indices = val_dataset.indices
    X_val = X_tensor[val_indices].to(device)
    Y_val_raw = df.iloc[val_indices][["循环寿命_h", "库仑效率_CE", "HER过电位_mV"]].to_numpy()
    names_val = df.iloc[val_indices]["additive_name"].tolist()

    with torch.no_grad():
        preds_scaled = best_model(X_val).cpu().numpy()
        preds_raw = target_scaler.inverse_transform(preds_scaled)

    eval_rows = []
    for i in range(len(val_indices)):
        eval_rows.append({
            "添加剂": names_val[i],
            "真值-循环(h)": f"{Y_val_raw[i, 0]:.1f}",
            "预测-循环(h)": f"{preds_raw[i, 0]:.1f}",
            "真值-CE": f"{Y_val_raw[i, 1]:.4f}",
            "预测-CE": f"{preds_raw[i, 1]:.4f}",
            "真值-HER(mV)": f"{Y_val_raw[i, 2]:.1f}",
            "预测-HER(mV)": f"{preds_raw[i, 2]:.1f}"
        })
    print(pd.DataFrame(eval_rows).to_string(index=False))

    return best_model, history


if __name__ == "__main__":
    model, history = train_multitask_model(
        csv_path="/home/a1810/zn_battery_experiment_database.csv",
        save_model_path="/home/a1810/best_multitask_model.pth",
        weight_lifespan=1.0,
        weight_ce=1.0,
        weight_her=1.0,
        lr=1e-3,
        max_epochs=200,
        patience=15,
        batch_size=4
    )
