# -*- coding: utf-8 -*-
"""
水系锌离子电池高容错工作台全流程测试用例套件
覆盖：
1. 表格缺失关键列精准报错
2. 行级脏数据（空值/非法字符）定位与安全填充
3. SMILES 语法错误（未闭合括号/非法原子）沙箱防御
4. 物理数值超出合理边界琥珀色告警
5. 模型缺失与内置 Mock 基准测试模型平滑回退
6. 高斯过程连续浓度响应曲线与置信带量化
"""

import io
import torch
import pandas as pd
from diagnostic_engine import (
    validate_uploaded_table,
    validate_smiles_sandbox,
    inspect_physical_bounds,
    run_model_inference_sandbox,
    calculate_gpr_concentration_curve
)

def test_missing_column_error():
    print("\n--- [测试 1: 上传缺失关键列的表格] ---")
    bad_csv = io.StringIO("""smiles,CV峰面积,XPS结合能偏移\nNCC(=O)O,3000.0,0.30\n""")
    report = validate_uploaded_table(bad_csv, "missing_cols.csv")
    print("缺失列报告:", report["missing_required_cols"])
    print("诊断信息:", report["diagnostic_messages"])
    assert "Tafel_Slope (Tafel 斜率)" in report["missing_required_cols"]
    assert "Raman_Area (Raman 拟合峰面积)" in report["missing_required_cols"]
    print("✅ 测试 1 通过: 缺失关键列已被精确识别并提示用户。")

def test_row_level_dirty_data():
    print("\n--- [测试 2: 表格行级脏数据精准定位] ---")
    dirty_csv = io.StringIO(
        "additive_name,smiles,CV峰面积,Tafel斜率,XPS结合能偏移,Raman峰面积拟合值,XRD晶面相对强度\n"
        "Sub_1,NCC(=O)O,3000.0,BAD_NUMBER,0.35,1600.0,2.1\n"
        "Sub_2,NC(=S)N,NaN,75.0,0.40,1500.0,2.2\n"
    )
    report = validate_uploaded_table(dirty_csv, "dirty.csv")
    print(f"检测到异常行数: {len(report['row_issues'])}")
    for issue in report["row_issues"]:
        print(f"  -> {issue['msg']}")
    assert len(report["row_issues"]) == 2
    # 验证均值填充未使 clean_df 崩溃
    clean_val1 = report["clean_df"].at[0, "Tafel斜率"]
    clean_val2 = report["clean_df"].at[1, "CV峰面积"]
    assert not pd.isna(clean_val1)
    assert not pd.isna(clean_val2)
    print("✅ 测试 2 通过: 行级非数值与空值被精确定位并自动实施安全填充。")

def test_smiles_syntax_sandbox():
    print("\n--- [测试 3: SMILES 语法异常沙箱防御] ---")
    # 测试未闭合括号
    res_unclosed = validate_smiles_sandbox("NCC(=O)O(")
    print("未闭合括号诊断:", res_unclosed["error_title"], "|", res_unclosed["error_detail"])
    assert not res_unclosed["valid"]
    assert "括号未配对" in res_unclosed["error_title"]

    # 测试非法元素或无法成键结构
    res_invalid_atom = validate_smiles_sandbox("C123456")
    print("非法化学结构诊断:", res_invalid_atom["error_title"], "|", res_invalid_atom["error_detail"])
    assert not res_invalid_atom["valid"]

    # 测试正常结构
    res_valid = validate_smiles_sandbox("NCC(=O)O")
    print("正常结构解析成功:", res_valid["valid"], res_valid["descriptors"])
    assert res_valid["valid"]
    print("✅ 测试 3 通过: SMILES 异常被沙箱完全拦截，未引起系统中断。")

def test_physical_bounds_warnings():
    print("\n--- [测试 4: 物理量边界合理性检查] ---")
    test_params = {
        "CV_Area": 50.0,          # 偏低 (< 100)
        "Tafel_Slope": 450.0,     # 偏高 (> 300)
        "XPS_Shift": 0.35,        # 正常
        "Raman_Area": 1650.0,     # 正常
        "XRD_Intensity": 2.10     # 正常
    }
    warnings = inspect_physical_bounds(test_params, concentration=15.0)  # 浓度偏高 (> 10)
    print(f"触发告警数: {len(warnings)}")
    for w in warnings:
        print(f"  -> {w['message']}")
    assert len(warnings) == 3
    print("✅ 测试 4 通过: 物理量超出合理区间触发精准琥珀色警告。")

def test_model_inference_sandbox():
    print("\n--- [测试 5: 模型推理沙箱与 Mock 回退] ---")
    dummy_tensor = torch.randn(1, 10)
    # 测试主模型
    res_main = run_model_inference_sandbox(dummy_tensor, model_path="/home/a1810/best_multitask_model.pth")
    print("主模型推理状态:", res_main["success"], res_main["predictions"])
    assert res_main["success"]

    # 测试权重缺失时平滑回退至 Mock 引擎
    res_mock = run_model_inference_sandbox(dummy_tensor, model_path="/not/exist/model.pth", use_mock=True)
    print("Mock 引擎推理状态:", res_mock["success"], "is_mock:", res_mock["is_mock"], res_mock["predictions"])
    assert res_mock["success"] and res_mock["is_mock"]

    # 测试非法维度张量防御
    bad_tensor = torch.randn(1, 5)
    res_bad = run_model_inference_sandbox(bad_tensor)
    print("非法维度防御:", res_bad["success"], res_bad["error_message"])
    assert not res_bad["success"]
    print("✅ 测试 5 通过: 推理沙箱健全，支持 Mock 自动回退与张量形状防呆。")

def test_gpr_optimization():
    print("\n--- [测试 6: 高斯过程连续浓度寻优] ---")
    gpr_res = calculate_gpr_concentration_curve("NCC(=O)O", 90.0, current_conc=1.5)
    print(f"推荐最佳浓度: {gpr_res['best_conc']} wt% | 最高得分: {gpr_res['max_score']} | 不确定度: ±{gpr_res['best_uncertainty']}")
    assert 0.5 <= gpr_res["best_conc"] <= 2.5
    assert len(gpr_res["concs"]) == len(gpr_res["upper_bound"]) == 80
    print("✅ 测试 6 通过: 高斯过程浓度优化与置信带量化计算正常。")

if __name__ == "__main__":
    test_missing_column_error()
    test_row_level_dirty_data()
    test_smiles_syntax_sandbox()
    test_physical_bounds_warnings()
    test_model_inference_sandbox()
    test_gpr_optimization()
    print("\n" + "=" * 60)
    print("🎉 所有 6 大高容错与精准排错诊断测试用例 100% 通过！")
    print("=" * 60)
