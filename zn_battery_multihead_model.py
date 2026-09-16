import torch
import torch.nn as nn
from typing import Dict, Union

class ZnBatteryMultiHeadNet(nn.Module):
    """
    基于 [5 维分子特征 + 5 维实验特征] 联合张量 (共 10 维输入) 的 4 层多任务前馈神经网络。

    架构特点：
    - Layer 1 (隐藏层 1): Linear(10, 64)  -> BatchNorm1d(64) -> ReLU()
    - Layer 2 (隐藏层 2): Linear(64, 32)  -> BatchNorm1d(32) -> ReLU()
    - Layer 3 (隐藏层 3): Linear(32, 16)  -> BatchNorm1d(16) -> ReLU()
    - Layer 4 (输出层分支): Multi-head 独立分支结构，同时输出 3 个神经元：
        * Head 1 (对称电池循环寿命): Linear(16, 1)
        * Head 2 (库仑效率 CE):        Linear(16, 1)
        * Head 3 (HER 析氢反应过电位): Linear(16, 1)
    """
    def __init__(self, in_features: int = 10):
        super().__init__()

        # 隐藏层 1
        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 隐藏层 2
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # 隐藏层 3 (共享隐层表征)
        self.layer3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # 多任务输出层分支 (Multi-head: 对应第 4 层连接)
        self.head_lifespan = nn.Linear(16, 1)  # 任务 1: 对称电池循环寿命 (Cycle Life, h)
        self.head_ce = nn.Linear(16, 1)        # 任务 2: 库仑效率 (Coulombic Efficiency, %)
        self.head_her = nn.Linear(16, 1)       # 任务 3: 析氢反应过电位 (HER Overpotential, mV)

    def forward(self, x: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播函数

        参数:
            x: 形状为 (batch_size, 10) 的联合特征张量
            return_dict: 若为 True，则返回包含任务名称的字典；默认 False，返回 (batch_size, 3) 拼接张量

        返回:
            torch.Tensor 或 Dict[str, torch.Tensor]: 包含 3 个预测指标输出
        """
        # 前向通过 3 个隐藏层
        feat = self.layer1(x)
        feat = self.layer2(feat)
        feat = self.layer3(feat)

        # 多任务独立分支预测 (各输出 1 个神经元)
        out_lifespan = self.head_lifespan(feat)
        out_ce = self.head_ce(feat)
        out_her = self.head_her(feat)

        if return_dict:
            return {
                "lifespan": out_lifespan,
                "ce": out_ce,
                "her_overpotential": out_her
            }

        # 在最后一维拼接为同时包含 3 个神经元的张量: (batch_size, 3)
        return torch.cat([out_lifespan, out_ce, out_her], dim=-1)
