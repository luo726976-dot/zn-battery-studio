# -*- coding: utf-8 -*-
"""
水系锌离子电池添加剂筛选交互式特征采集与分析工具 (命令行交互模式 - CLI)
适用于无桌面浏览器或纯 SSH 终端环境。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

from battery_feature_processor import (
    DatabaseManager,
    MolecularPropertyEngine,
    FeatureAlignmentPipeline,
    MOL_FEATURE_NAMES,
    EXP_FEATURE_NAMES,
    ALL_FEATURE_NAMES
)

def print_banner():
    print("=" * 72)
    print(" 🔋 水系锌离子电池 (2M ZnSO4) 添加剂特征采集与分析工具 (CLI 交互终端) ")
    print("=" * 72)

def main():
    print_banner()

    # 1. 初始化数据库与流水线
    db_manager = DatabaseManager()
    pipeline = FeatureAlignmentPipeline(db_manager)

    print(f"\n[模块 1: 数据库状态]")
    print(f"数据源: {db_manager.source_desc}")
    candidate_names = db_manager.get_additive_names()
    print(f"当前可用候选添加剂数量: {len(candidate_names)}")

    while True:
        print("\n" + "-" * 72)
        print("请选择工作模式:")
        print("  1. 从已有数据库中选择添加剂")
        print("  2. 输入全新候选添加剂 X (支持分子联想与 RDKit 特征提取)")
        print("  3. 切换或加载自定义 CSV/Excel 文件")
        print("  0. 退出系统")
        print("-" * 72)

        choice = input("请输入选项编号 [1/2/3/0]: ").strip()

        if choice == "0":
            print("\n👋 感谢使用，系统已安全退出。")
            break

        elif choice == "3":
            file_path = input("\n请输入本地 CSV 或 Excel 文件绝对路径: ").strip()
            if not os.path.exists(file_path):
                print(f"❌ 路径不存在: '{file_path}'")
                continue
            success, msg = db_manager.load_from_file_or_buffer(file_path, file_path)
            if success:
                pipeline = FeatureAlignmentPipeline(db_manager)
                candidate_names = db_manager.get_additive_names()
                print(f"✅ {msg}")
            else:
                print(f"❌ 加载失败: {msg}")
            continue

        elif choice == "1":
            print("\n已加载的候选添加剂列表:")
            for idx, name in enumerate(candidate_names, 1):
                print(f"  [{idx:2d}] {name}")
            
            sel = input(f"\n请选择添加剂编号 [1-{len(candidate_names)}]: ").strip()
            try:
                sel_idx = int(sel) - 1
                if 0 <= sel_idx < len(candidate_names):
                    chosen_name = candidate_names[sel_idx]
                else:
                    print("⚠️ 输入编号超出范围！")
                    continue
            except ValueError:
                print("⚠️ 请输入有效数字！")
                continue

            details = db_manager.get_additive_details(chosen_name)
            substance_name = chosen_name
            smiles = details["smiles"]
            exp_features = details["exp_features"].copy()
            print(f"\n已选择: 【{substance_name}】")
            print(f"SMILES: {smiles}")

        elif choice == "2":
            substance_name = input("\n请输入全新添加剂物质名称 (例如: 咖啡因, 醋酸锌, 新型季铵盐等): ").strip()
            if not substance_name:
                substance_name = "全新添加剂_X"

            print(f"\n正在尝试为 '{substance_name}' 智能检索 SMILES 分子式...")
            auto_smi, source, err = MolecularPropertyEngine.query_smiles(substance_name)
            default_smi = auto_smi if auto_smi else ""
            if auto_smi:
                print(f"💡 自动从 [{source}] 匹配到分子式: {auto_smi}")
            else:
                print(f"ℹ️ {err}")

            smi_input = input(f"请输入或确认 SMILES [默认: {default_smi}]: ").strip()
            smiles = smi_input if smi_input else default_smi
            exp_features = db_manager.feature_medians.copy()

        else:
            print("⚠️ 无效选项，请重新输入！")
            continue

        # 2. 校验 SMILES 并提取 5 维分子特征
        succ, mol_feats, mol_err = MolecularPropertyEngine.calculate_descriptors(smiles)
        if not succ:
            print(f"\n❌ SMILES 解析错误: {mol_err}")
            retry_smi = input("请手动重新输入有效的 SMILES (或回车放弃): ").strip()
            if retry_smi:
                succ, mol_feats, mol_err = MolecularPropertyEngine.calculate_descriptors(retry_smi)
                if succ:
                    smiles = retry_smi
                else:
                    print(f"❌ 依然无效，采用默认全 0 分子特征填充。")
                    mol_feats = {k: 0.0 for k in MOL_FEATURE_NAMES}
            else:
                mol_feats = {k: 0.0 for k in MOL_FEATURE_NAMES}

        print("\n" + "=" * 50)
        print("🔬 [RDKit 5 维分子化学特征计算结果]")
        for k, v in mol_feats.items():
            print(f"  - {k:15s}: {v}")
        print("=" * 50)

        # 3. 实验特征录入或确认
        print("\n📊 [电化学与物相 5 维实验特征确认/修改]")
        print("直接按回车可保留括号内的默认/历史数值:")

        for k in EXP_FEATURE_NAMES:
            default_val = exp_features.get(k, db_manager.feature_medians.get(k, 0.0))
            user_val = input(f"  - {k} [默认 {default_val}]: ").strip()
            if user_val:
                try:
                    exp_features[k] = float(user_val)
                except ValueError:
                    print(f"    ⚠️ 输入格式无效，保持默认值 {default_val}")
                    exp_features[k] = float(default_val)
            else:
                exp_features[k] = float(default_val)

        # 4. 执行特征对齐与张量打包
        print("\n🚀 正在执行特征对齐与 PyTorch Tensor 打包...")
        packed_res = pipeline.pack_sample_features(
            sample_name=substance_name,
            smiles=smiles,
            mol_features=mol_feats,
            exp_features_raw=exp_features
        )

        print("\n" + "=" * 72)
        print(f"🎉 样本 【{packed_res['sample_name']}】 特征画像已就绪！")
        print("=" * 72)
        print("\n【特征对齐画像对比表】:")
        print(packed_res["feature_table"].to_string(index=False))

        print("\n【PyTorch Tensor 打包信息】:")
        print(f"  - 张量形态 (Shape): {packed_res['tensor_shape']}")
        print(f"  - 数据类型 (Dtype): {packed_res['tensor_dtype']}")
        print(f"  - 数值内容 (Tensor):\n    {packed_res['torch_tensor']}")

        # 5. 可选神经网络性能预测
        preds = pipeline.run_optional_model_prediction(packed_res["torch_tensor"])
        if preds:
            print("\n【🤖 预训练多任务神经网络实时性能预测】:")
            print(f"  - 循环寿命 (Cycle Life) : {preds['循环寿命 (h)']} h")
            print(f"  - 库仑效率 (Coulombic Eff): {preds['库仑效率 CE (%)']}%")
            print(f"  - HER 析氢过电位 (HER OP) : {preds['HER析氢过电位 (mV)']} mV")

        # 6. 保存导出选项
        save_opt = input("\n是否导出该样本特征至 JSON 文件? [y/N]: ").strip().lower()
        if save_opt == "y":
            out_file = f"zn_feature_{substance_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_name": packed_res["sample_name"],
                    "smiles": packed_res["smiles"],
                    "raw_features": packed_res["raw_portrait"],
                    "scaled_features": packed_res["scaled_portrait"],
                    "tensor_vector": packed_res["numpy_vector"].tolist()
                }, f, indent=2, ensure_ascii=False)
            print(f"✅ 已成功保存至本地文件: {out_file}")

if __name__ == "__main__":
    main()
