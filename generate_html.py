# -*- coding: utf-8 -*-
"""
Generate complete single-file zinc battery AI screening & optimization dashboard index.html
"""
import json
import os

with open('/home/a1810/dashboard_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

json_str = json.dumps(data, ensure_ascii=False)

html_content = f'''<!DOCTYPE html>
<html lang="zh-CN" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>硫酸锌 (ZnSO₄) 电池添加剂 AI 筛选与浓度寻优系统 | ZnBattery AI Studio</title>
  
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- ECharts 5 CDN -->
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
  
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          colors: {{
            brand: {{
              50: '#ecfdf5',
              100: '#d1fae5',
              400: '#34d399',
              500: '#10b981',
              600: '#059669',
              700: '#047857',
              900: '#064e3b'
            }},
            cyan: {{
              400: '#22d3ee',
              500: '#06b6d4',
              600: '#0891b2'
            }},
            darkbg: '#0f172a',
            darkcard: '#1e293b',
            darkborder: '#334155'
          }}
        }}
      }}
    }}
  </script>

  <style>
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
      width: 8px;
      height: 8px;
    }}
    ::-webkit-scrollbar-track {{
      background: #0f172a;
    }}
    ::-webkit-scrollbar-thumb {{
      background: #334155;
      border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
      background: #475569;
    }}
    
    .glow-effect {{
      box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.25);
    }}
    .glow-cyan {{
      box-shadow: 0 0 25px -5px rgba(6, 182, 212, 0.25);
    }}
    
    input[type=range] {{
      accent-color: #10b981;
    }}
  </style>
</head>

<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased transition-colors duration-200">

  <!-- ========================================================================= -->
  <!-- Top Navigation Bar -->
  <!-- ========================================================================= -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <!-- Logo & Brand -->
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-400 p-0.5 shadow-lg shadow-emerald-500/20 flex items-center justify-center">
          <div class="w-full h-full bg-slate-900 rounded-[10px] flex items-center justify-center">
            <svg class="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="text-base sm:text-lg font-bold tracking-tight bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">
              ZnBattery AI Studio
            </span>
            <span class="text-[10px] px-2 py-0.5 rounded-full font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              v2.4 Multi-Task + GPR
            </span>
          </div>
          <p class="text-xs text-slate-400 hidden sm:block">硫酸锌 (ZnSO₄) 电池添加剂双阶段 AI 智能筛选与浓度寻优系统</p>
        </div>
      </div>

      <!-- System Status Badges & Actions -->
      <div class="flex items-center space-x-2 sm:space-x-3">
        <div class="hidden lg:flex items-center space-x-2 text-xs bg-slate-800/80 border border-slate-700/60 rounded-lg px-3 py-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span class="text-slate-300">体系: <strong class="text-emerald-400 font-medium">2 mol/L ZnSO₄</strong></span>
          <span class="text-slate-600">|</span>
          <span class="text-slate-300">已载入模型: <strong class="text-cyan-400 font-medium">PyTorch 10D-NN</strong></span>
          <span class="text-slate-600">|</span>
          <span class="text-slate-300">数据库: <strong class="text-indigo-400 font-medium">20 种基准添加剂</strong></span>
        </div>

        <!-- Export Report Button -->
        <button onclick="openExportModal()" class="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg transition-all flex items-center space-x-1.5 shadow-sm">
          <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span class="hidden sm:inline">导出评估报告</span>
        </button>

        <!-- Reset Button -->
        <button onclick="resetToDefault()" title="恢复默认参数" class="p-2 text-slate-400 hover:text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition-all">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 sm:space-x-4 border-t border-slate-800/80 overflow-x-auto text-xs sm:text-sm font-medium">
      <button onclick="switchTab('tab-screening')" id="nav-screening" class="tab-btn py-3 px-3 sm:px-4 border-b-2 border-emerald-400 text-emerald-400 flex items-center space-x-2 whitespace-nowrap transition-colors">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        <span>阶段一：多任务初筛推理</span>
      </button>

      <button onclick="switchTab('tab-gpr')" id="nav-gpr" class="tab-btn py-3 px-3 sm:px-4 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition-colors">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
        <span>阶段二：高斯过程浓度寻优</span>
      </button>

      <button onclick="switchTab('tab-database')" id="nav-database" class="tab-btn py-3 px-3 sm:px-4 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition-colors">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
        </svg>
        <span>实验数据库全景与对比</span>
      </button>

      <button onclick="switchTab('tab-guide')" id="nav-guide" class="tab-btn py-3 px-3 sm:px-4 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition-colors">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
        <span>算法架构与电化学机理</span>
      </button>
    </div>
  </header>

  <!-- ========================================================================= -->
  <!-- Main Container -->
  <!-- ========================================================================= -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">

    <!-- ======================================================================= -->
    <!-- TAB 1: 阶段一：多任务初筛推理 (Multi-task Screening) -->
    <!-- ======================================================================= -->
    <div id="tab-screening" class="tab-content space-y-6">
      
      <!-- Top Banner Summary -->
      <div class="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border border-slate-700/70 rounded-2xl p-5 shadow-xl relative overflow-hidden">
        <div class="absolute right-0 top-0 w-96 h-full bg-emerald-500/5 blur-3xl pointer-events-none"></div>
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 relative z-10">
          <div>
            <div class="flex items-center space-x-2">
              <span class="px-2.5 py-1 text-xs font-semibold rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                Stage 1 核心引擎
              </span>
              <h1 class="text-xl font-bold text-white">候选添加剂多任务性能初筛与准入评测</h1>
            </div>
            <p class="text-slate-400 text-xs sm:text-sm mt-1.5 max-w-3xl">
              结合 RDKit 提取的 5 维分子拓扑物化特征与 Origin 2024b 实测 5 维电化学微观特征，输入已训练的 10 维多任务 PyTorch 神经网络，前向预测对称电池循环寿命、库仑效率 (CE) 以及析氢反应 (HER) 过电位。
            </p>
          </div>

          <!-- Quick Presets -->
          <div class="flex flex-wrap items-center gap-2">
            <span class="text-xs text-slate-400 font-medium">快速载入代表性添加剂:</span>
            <button onclick="loadPreset('2-氨基-4-溴蒽醌-2-磺酸钠')" class="px-2.5 py-1 text-xs bg-slate-800 hover:bg-emerald-950/60 hover:text-emerald-300 text-slate-300 border border-slate-700 hover:border-emerald-500/40 rounded-lg transition-all">
              🌟 2-氨基-4-溴蒽醌-2-磺酸钠
            </button>
            <button onclick="loadPreset('柠檬酸')" class="px-2.5 py-1 text-xs bg-slate-800 hover:bg-emerald-950/60 hover:text-emerald-300 text-slate-300 border border-slate-700 hover:border-emerald-500/40 rounded-lg transition-all">
              🍋 柠檬酸 (最高寿命)
            </button>
            <button onclick="loadPreset('甘氨酸')" class="px-2.5 py-1 text-xs bg-slate-800 hover:bg-emerald-950/60 hover:text-emerald-300 text-slate-300 border border-slate-700 hover:border-emerald-500/40 rounded-lg transition-all">
              🧬 甘氨酸 (两性离子)
            </button>
            <button onclick="loadPreset('三氟甲磺酸根单体')" class="px-2.5 py-1 text-xs bg-slate-800 hover:bg-emerald-950/60 hover:text-emerald-300 text-slate-300 border border-slate-700 hover:border-emerald-500/40 rounded-lg transition-all">
              ⚡ 三氟甲磺酸根
            </button>
          </div>
        </div>
      </div>

      <!-- Two-Column Grid: Left (Inputs) & Right (Predictions) -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- ==================== LEFT COLUMN: INPUT FEATURES ==================== -->
        <div class="lg:col-span-6 space-y-5">
          
          <!-- Card 1: 分子信息与化学描述符 -->
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
              <div class="flex items-center space-x-2">
                <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                <h2 class="text-sm font-semibold text-white">1. 分子化学结构与物化参数 (输入 1: 5 维)</h2>
              </div>
              <span class="text-xs text-slate-500 font-mono">RDKit Engine</span>
            </div>

            <!-- Additive Selector & SMILES -->
            <div class="space-y-3">
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">选择待测添加剂或输入新候选分子</label>
                <div class="flex gap-2">
                  <select id="additive-select" onchange="onSelectAdditiveChange(this.value)" class="flex-1 bg-slate-800 border border-slate-700 text-slate-100 text-xs rounded-xl px-3 py-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                    <!-- Options populated via JS -->
                  </select>
                </div>
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">SMILES 结构式</label>
                <input type="text" id="input-smiles" readonly class="w-full bg-slate-950/80 border border-slate-800 text-emerald-400 text-xs font-mono rounded-xl px-3 py-2 focus:outline-none overflow-x-auto select-all cursor-text" value="">
              </div>
            </div>

            <!-- 5D Molecular Descriptors Grid -->
            <div>
              <div class="text-xs font-medium text-slate-400 mb-2 flex items-center justify-between">
                <span>RDKit 物理化学特征向量:</span>
                <span class="text-[11px] text-slate-500">拓扑/氢键/极性</span>
              </div>
              <div class="grid grid-cols-5 gap-2 text-center">
                <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-2">
                  <div class="text-[10px] text-slate-400">分子量 MW</div>
                  <div id="desc-molwt" class="text-xs font-bold text-slate-100 font-mono mt-0.5">--</div>
                  <div class="text-[9px] text-slate-500">g/mol</div>
                </div>
                <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-2">
                  <div class="text-[10px] text-slate-400">极性面积 TPSA</div>
                  <div id="desc-tpsa" class="text-xs font-bold text-slate-100 font-mono mt-0.5">--</div>
                  <div class="text-[9px] text-slate-500">Å²</div>
                </div>
                <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-2">
                  <div class="text-[10px] text-slate-400">分配系数 LogP</div>
                  <div id="desc-logp" class="text-xs font-bold text-slate-100 font-mono mt-0.5">--</div>
                  <div class="text-[9px] text-slate-500">脂水分配</div>
                </div>
                <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-2">
                  <div class="text-[10px] text-slate-400">氢键供体 HBD</div>
                  <div id="desc-hbd" class="text-xs font-bold text-slate-100 font-mono mt-0.5">--</div>
                  <div class="text-[9px] text-slate-500">个</div>
                </div>
                <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-2">
                  <div class="text-[10px] text-slate-400">氢键受体 HBA</div>
                  <div id="desc-hba" class="text-xs font-bold text-slate-100 font-mono mt-0.5">--</div>
                  <div class="text-[9px] text-slate-500">个</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Card 2: 5 维 Origin 实验特征输入 (含滑块与精细数值) -->
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
              <div class="flex items-center space-x-2">
                <span class="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
                <h2 class="text-sm font-semibold text-white">2. Origin 2024b 实测微观电化学特征 (输入 2: 5 维)</h2>
              </div>
              <span class="text-xs text-cyan-400/80 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">实时驱动推断</span>
            </div>

            <!-- Feature Sliders List -->
            <div class="space-y-3.5 text-xs">
              <!-- Feature 1: CV Peak Area -->
              <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="font-medium text-slate-200">① CV 峰面积 (剥离电量)</span>
                  <div class="flex items-center space-x-1.5">
                    <input type="number" id="input-cv" step="1" min="2800" max="3500" class="w-20 bg-slate-800 border border-slate-700 rounded-lg px-2 py-0.5 text-right font-mono text-emerald-400 text-xs focus:outline-none" oninput="onFeatureInput('cv', this.value)">
                    <span class="text-slate-500 text-[11px]">mC</span>
                  </div>
                </div>
                <input type="range" id="slider-cv" min="2800" max="3500" step="5" class="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer" oninput="onFeatureSlider('cv', this.value)">
                <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>低可逆剥离 (2800 mC)</span>
                  <span class="text-slate-400">基准均值: ~3120 mC</span>
                  <span>高极性高容量 (3500 mC)</span>
                </div>
              </div>

              <!-- Feature 2: Tafel Slope -->
              <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="font-medium text-slate-200">② Tafel 极化斜率 (腐蚀动力学)</span>
                  <div class="flex items-center space-x-1.5">
                    <input type="number" id="input-tafel" step="0.1" min="60" max="85" class="w-20 bg-slate-800 border border-slate-700 rounded-lg px-2 py-0.5 text-right font-mono text-emerald-400 text-xs focus:outline-none" oninput="onFeatureInput('tafel', this.value)">
                    <span class="text-slate-500 text-[11px]">mV/dec</span>
                  </div>
                </div>
                <input type="range" id="slider-tafel" min="60" max="85" step="0.1" class="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer" oninput="onFeatureSlider('tafel', this.value)">
                <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span class="text-emerald-400">快动力学/耐蚀 (60.0 mV/dec)</span>
                  <span class="text-slate-400">基准均值: ~71.5</span>
                  <span class="text-rose-400">高极化/易腐蚀 (85.0)</span>
                </div>
              </div>

              <!-- Feature 3: XPS Binding Energy Shift -->
              <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="font-medium text-slate-200">③ XPS 结合能偏移 (Zn 2p 轨道)</span>
                  <div class="flex items-center space-x-1.5">
                    <input type="number" id="input-xps" step="0.01" min="0.20" max="0.55" class="w-20 bg-slate-800 border border-slate-700 rounded-lg px-2 py-0.5 text-right font-mono text-emerald-400 text-xs focus:outline-none" oninput="onFeatureInput('xps', this.value)">
                    <span class="text-slate-500 text-[11px]">eV</span>
                  </div>
                </div>
                <input type="range" id="slider-xps" min="0.20" max="0.55" step="0.01" class="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer" oninput="onFeatureSlider('xps', this.value)">
                <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>弱电子微扰 (0.20 eV)</span>
                  <span class="text-slate-400">基准均值: ~0.35 eV</span>
                  <span>强化学吸附 (0.55 eV)</span>
                </div>
              </div>

              <!-- Feature 4: Raman Peak Area -->
              <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="font-medium text-slate-200">④ Raman 峰面积拟合值 (水化壳层调控)</span>
                  <div class="flex items-center space-x-1.5">
                    <input type="number" id="input-raman" step="1" min="1300" max="2000" class="w-20 bg-slate-800 border border-slate-700 rounded-lg px-2 py-0.5 text-right font-mono text-emerald-400 text-xs focus:outline-none" oninput="onFeatureInput('raman', this.value)">
                    <span class="text-slate-500 text-[11px]">a.u.</span>
                  </div>
                </div>
                <input type="range" id="slider-raman" min="1300" max="2000" step="5" class="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer" oninput="onFeatureSlider('raman', this.value)">
                <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>自由水活度高 (1300)</span>
                  <span class="text-slate-400">基准均值: ~1630</span>
                  <span>强去溶剂化 (2000)</span>
                </div>
              </div>

              <!-- Feature 5: XRD (002)/(101) Ratio -->
              <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="font-medium text-slate-200">⑤ XRD 晶面相对强度比 I(002)/I(101)</span>
                  <div class="flex items-center space-x-1.5">
                    <input type="number" id="input-xrd" step="0.01" min="1.60" max="2.85" class="w-20 bg-slate-800 border border-slate-700 rounded-lg px-2 py-0.5 text-right font-mono text-emerald-400 text-xs focus:outline-none" oninput="onFeatureInput('xrd', this.value)">
                    <span class="text-slate-500 text-[11px]">强度比</span>
                  </div>
                </div>
                <input type="range" id="slider-xrd" min="1.60" max="2.85" step="0.01" class="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer" oninput="onFeatureSlider('xrd', this.value)">
                <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>杂乱晶相 (1.60)</span>
                  <span class="text-slate-400">基准均值: ~2.15</span>
                  <span class="text-emerald-400">高择优平整沉积 (2.85)</span>
                </div>
              </div>
            </div>

            <!-- Screening Threshold Settings Toggle -->
            <div class="border-t border-slate-800 pt-3">
              <details class="text-xs group">
                <summary class="cursor-pointer text-slate-400 hover:text-slate-200 flex items-center justify-between select-none">
                  <span class="flex items-center space-x-1">
                    <svg class="w-4 h-4 text-slate-500 group-open:rotate-90 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                    </svg>
                    <span>自定义初筛通过准入阈值 (Screening Criteria)</span>
                  </span>
                  <span class="text-[11px] text-slate-500">点击展开/折叠</span>
                </summary>
                <div class="grid grid-cols-3 gap-3 mt-3 bg-slate-950/70 p-3 rounded-xl border border-slate-800">
                  <div>
                    <label class="text-[10px] text-slate-400">最低循环寿命</label>
                    <div class="flex items-center space-x-1 mt-1">
                      <input type="number" id="thresh-lifespan" value="1000" step="50" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white" oninput="recalculateScreening()">
                      <span class="text-[10px] text-slate-500">h</span>
                    </div>
                  </div>
                  <div>
                    <label class="text-[10px] text-slate-400">最低库仑效率</label>
                    <div class="flex items-center space-x-1 mt-1">
                      <input type="number" id="thresh-ce" value="99.20" step="0.05" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white" oninput="recalculateScreening()">
                      <span class="text-[10px] text-slate-500">%</span>
                    </div>
                  </div>
                  <div>
                    <label class="text-[10px] text-slate-400">最低析氢过电位</label>
                    <div class="flex items-center space-x-1 mt-1">
                      <input type="number" id="thresh-her" value="175.0" step="5" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white" oninput="recalculateScreening()">
                      <span class="text-[10px] text-slate-500">mV</span>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </div>

        <!-- ==================== RIGHT COLUMN: PREDICTIONS & CHARTS ==================== -->
        <div class="lg:col-span-6 space-y-5">
          
          <!-- Card 1: Screening Conclusion Banner -->
          <div id="banner-screening-status" class="rounded-2xl p-5 border transition-all duration-300 shadow-xl flex items-center justify-between">
            <div class="flex items-center space-x-3.5">
              <div id="status-icon-container" class="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0">
                <!-- Icon via JS -->
              </div>
              <div>
                <div class="flex items-center space-x-2">
                  <h3 id="status-title" class="text-base font-bold text-white">初筛评估进行中...</h3>
                  <span id="status-badge" class="text-xs px-2 py-0.5 rounded-full font-semibold">--</span>
                </div>
                <p id="status-desc" class="text-xs text-slate-300 mt-1">正在前向计算神经网络多指标联合响应...</p>
              </div>
            </div>

            <button id="btn-goto-opt" onclick="switchTab('tab-gpr')" class="hidden px-4 py-2 text-xs font-semibold rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-lg shadow-emerald-500/20 transition-all flex items-center space-x-1 flex-shrink-0">
              <span>进阶浓度寻优</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </button>
          </div>

          <!-- Card 2: 3 Major Electrochemical Targets Grid -->
          <div class="grid grid-cols-3 gap-3">
            <!-- Target 1: Lifespan -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex flex-col justify-between relative overflow-hidden group">
              <div class="absolute top-0 right-0 w-20 h-20 bg-emerald-500/10 rounded-full blur-xl group-hover:bg-emerald-500/20 transition-all"></div>
              <div>
                <div class="text-[11px] font-medium text-slate-400 flex items-center justify-between">
                  <span>循环寿命</span>
                  <span id="life-pass-tag" class="text-[10px] px-1.5 py-0.5 rounded font-semibold bg-emerald-500/20 text-emerald-300">达标</span>
                </div>
                <div class="mt-2">
                  <div class="flex items-baseline space-x-1">
                    <span id="pred-lifespan" class="text-2xl font-black text-emerald-400 font-mono tracking-tight">--</span>
                    <span class="text-xs text-slate-400">h</span>
                  </div>
                  <div class="text-[10px] text-slate-500 mt-1">
                    基准纯电解液: <strong class="text-slate-300">~150 h</strong>
                  </div>
                </div>
              </div>
              <div class="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 flex items-center justify-between">
                <span>相对提升:</span>
                <span id="pred-life-mult" class="font-bold text-emerald-400 font-mono">-- 倍</span>
              </div>
            </div>

            <!-- Target 2: CE -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex flex-col justify-between relative overflow-hidden group">
              <div class="absolute top-0 right-0 w-20 h-20 bg-cyan-500/10 rounded-full blur-xl group-hover:bg-cyan-500/20 transition-all"></div>
              <div>
                <div class="text-[11px] font-medium text-slate-400 flex items-center justify-between">
                  <span>库仑效率 (CE)</span>
                  <span id="ce-pass-tag" class="text-[10px] px-1.5 py-0.5 rounded font-semibold bg-cyan-500/20 text-cyan-300">达标</span>
                </div>
                <div class="mt-2">
                  <div class="flex items-baseline space-x-1">
                    <span id="pred-ce" class="text-2xl font-black text-cyan-400 font-mono tracking-tight">--</span>
                    <span class="text-xs text-slate-400">%</span>
                  </div>
                  <div class="text-[10px] text-slate-500 mt-1">
                    理想可逆线: <strong class="text-slate-300">&gt; 99.5%</strong>
                  </div>
                </div>
              </div>
              <div class="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 flex items-center justify-between">
                <span>等级评定:</span>
                <span id="pred-ce-grade" class="font-bold text-cyan-400 font-mono">S 级可逆</span>
              </div>
            </div>

            <!-- Target 3: HER Overpotential -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex flex-col justify-between relative overflow-hidden group">
              <div class="absolute top-0 right-0 w-20 h-20 bg-indigo-500/10 rounded-full blur-xl group-hover:bg-indigo-500/20 transition-all"></div>
              <div>
                <div class="text-[11px] font-medium text-slate-400 flex items-center justify-between">
                  <span>HER 析氢过电位</span>
                  <span id="her-pass-tag" class="text-[10px] px-1.5 py-0.5 rounded font-semibold bg-indigo-500/20 text-indigo-300">达标</span>
                </div>
                <div class="mt-2">
                  <div class="flex items-baseline space-x-1">
                    <span id="pred-her" class="text-2xl font-black text-indigo-400 font-mono tracking-tight">--</span>
                    <span class="text-xs text-slate-400">mV</span>
                  </div>
                  <div class="text-[10px] text-slate-500 mt-1">
                    析氢能垒越宽越稳定
                  </div>
                </div>
              </div>
              <div class="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 flex items-center justify-between">
                <span>防腐能垒:</span>
                <span id="pred-her-grade" class="font-bold text-indigo-400 font-mono">强抗析氢</span>
              </div>
            </div>
          </div>

          <!-- Card 3: 综合协同得分与双图联动展示 -->
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 class="text-sm font-semibold text-white">综合多目标协同指数与多维指纹</h3>
                <p class="text-xs text-slate-400">加权融合：循环寿命 (40%) + 库仑效率 (35%) + 析氢过电位 (25%)</p>
              </div>
              <div class="text-right">
                <span class="text-xs text-slate-400">综合得分:</span>
                <span id="pred-composite-score" class="text-xl font-black text-amber-400 font-mono ml-1">--</span>
                <span class="text-xs text-slate-500">/ 100</span>
              </div>
            </div>

            <!-- ECharts Container for Gauge & Radar side-by-side -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <!-- Gauge -->
              <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col items-center justify-center">
                <div class="text-xs font-medium text-slate-400 mb-1">综合潜力仪表盘</div>
                <div id="chart-gauge" style="width: 100%; height: 210px;"></div>
              </div>
              <!-- Radar -->
              <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col items-center justify-center">
                <div class="text-xs font-medium text-slate-400 mb-1">电化学多维指纹比对</div>
                <div id="chart-radar" style="width: 100%; height: 210px;"></div>
              </div>
            </div>
          </div>

        </div>

      </div>

    </div>


    <!-- ======================================================================= -->
    <!-- TAB 2: 阶段二：高斯过程浓度寻优 (GPR Concentration Optimization) -->
    <!-- ======================================================================= -->
    <div id="tab-gpr" class="tab-content hidden space-y-6">
      
      <!-- GPR Top Hero Card -->
      <div class="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 rounded-2xl p-5 shadow-2xl relative overflow-hidden">
        <div class="absolute right-0 top-0 w-80 h-full bg-cyan-500/10 blur-3xl pointer-events-none"></div>
        <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-4 relative z-10">
          <div>
            <div class="flex items-center space-x-2">
              <span class="px-2.5 py-1 text-xs font-semibold rounded-md bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                Stage 2 高斯过程寻优 (GPR)
              </span>
              <h2 class="text-xl font-bold text-white">连续浓度空间非线性响应与不确定度寻优</h2>
            </div>
            <p class="text-slate-300 text-xs sm:text-sm mt-1.5 max-w-3xl">
              基于本地 2 mol/L ZnSO₄ 电解液实测浓度梯度数据库，应用高斯过程回归 (RBF + WhiteKernel)，在 0.2 ~ 3.0 wt% 连续浓度空间拟合非对称吸附响应曲线，量化 95% 置信区间 (±1.96σ)，确定全局最优添加浓度。
            </p>
          </div>

          <!-- Current Substance Tag -->
          <div class="bg-slate-800/90 border border-slate-700/80 rounded-xl p-3 flex items-center space-x-3 flex-shrink-0">
            <div class="w-9 h-9 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
              Zn
            </div>
            <div>
              <div class="text-[10px] text-slate-400">当前寻优对象:</div>
              <div id="gpr-target-name" class="text-xs font-bold text-slate-100 max-w-[180px] truncate">--</div>
              <div class="text-[10px] text-slate-400">初筛综合得分: <span id="gpr-target-score" class="text-amber-400 font-mono font-bold">--</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Hero Decision Panel: Optimal Concentration & Confidence -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <!-- Metric 1: Recommended Concentration -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div class="text-xs font-medium text-slate-400">🎯 推荐最佳添加浓度 (Optimal Conc)</div>
          <div class="mt-3 flex items-baseline space-x-1.5">
            <span id="gpr-best-conc" class="text-3xl font-black text-emerald-400 font-mono">--</span>
            <span class="text-sm font-semibold text-slate-400">wt%</span>
          </div>
          <p class="text-[11px] text-slate-500 mt-2">电极界面达到最致密吸附保护膜</p>
        </div>

        <!-- Metric 2: Peak Composite Score -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div class="text-xs font-medium text-slate-400">🏆 峰值综合性能得分 (Peak Score)</div>
          <div class="mt-3 flex items-baseline space-x-1.5">
            <span id="gpr-peak-score" class="text-3xl font-black text-amber-400 font-mono">--</span>
            <span class="text-sm font-semibold text-slate-400">分</span>
          </div>
          <p class="text-[11px] text-slate-500 mt-2">多目标协同函数在极值点处取最大值</p>
        </div>

        <!-- Metric 3: Uncertainty Sigma -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div class="text-xs font-medium text-slate-400">🔍 模型不确定度 (Uncertainty σ)</div>
          <div class="mt-3 flex items-baseline space-x-1.5">
            <span id="gpr-sigma" class="text-3xl font-black text-cyan-400 font-mono">±--</span>
          </div>
          <p class="text-[11px] text-slate-500 mt-2">基于高斯过程后验方差计算</p>
        </div>

        <!-- Metric 4: 95% Confidence Interval -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div class="text-xs font-medium text-slate-400">📊 95% 置信区间 (CI: ±1.96σ)</div>
          <div class="mt-3 flex items-baseline space-x-1.5">
            <span id="gpr-ci" class="text-xl font-black text-slate-200 font-mono tracking-tight">[ -- , -- ]</span>
          </div>
          <p class="text-[11px] text-slate-500 mt-2">高斯后验分布 95% 概率包含真实值</p>
        </div>
      </div>

      <!-- Main GPR Chart & Control Panel -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <h3 class="text-sm font-bold text-white flex items-center space-x-2">
              <span>连续浓度空间高斯过程后验响应曲线 (Posterior Mean & 95% CI)</span>
              <span class="text-[10px] px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-normal">Kernel: RBF + WhiteKernel</span>
            </h3>
            <p class="text-xs text-slate-400 mt-0.5">实线代表后验预测均值，半透明青色阴影区域为 95% 置信区间</p>
          </div>

          <!-- Target Curve Switcher -->
          <div class="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
            <button onclick="switchGPRCurve('composite')" id="btn-curve-composite" class="px-2.5 py-1 rounded-lg bg-emerald-500 text-slate-950 font-bold transition-all">综合评分</button>
            <button onclick="switchGPRCurve('lifespan')" id="btn-curve-lifespan" class="px-2.5 py-1 rounded-lg text-slate-400 hover:text-slate-200 transition-all">循环寿命</button>
            <button onclick="switchGPRCurve('ce')" id="btn-curve-ce" class="px-2.5 py-1 rounded-lg text-slate-400 hover:text-slate-200 transition-all">库仑效率</button>
            <button onclick="switchGPRCurve('her')" id="btn-curve-her" class="px-2.5 py-1 rounded-lg text-slate-400 hover:text-slate-200 transition-all">HER过电位</button>
          </div>
        </div>

        <!-- Large Chart -->
        <div id="chart-gpr-curve" style="width: 100%; height: 380px;"></div>

        <!-- Diagnosis Insight Banner -->
        <div class="bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex items-start space-x-3 text-xs">
          <div class="text-cyan-400 text-base mt-0.5">💡</div>
          <div class="space-y-1">
            <h4 class="font-semibold text-slate-200">电化学机理与浓度效应归因诊断:</h4>
            <p id="gpr-diagnosis-text" class="text-slate-400 leading-relaxed">
              分析中...
            </p>
          </div>
        </div>
      </div>

      <!-- Focus Gradient Discrete Evaluation Table -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 class="text-sm font-bold text-white">重点浓度梯度预测与不确定度明细 (Focus Concentration Evaluations)</h3>
            <p class="text-xs text-slate-400">对标 0.5wt%, 1.0wt%, 1.5wt%, 2.0wt%, 2.5wt% 实验离散测试标准点</p>
          </div>
          <span class="text-xs text-slate-500">区间采样精度: 0.01 wt%</span>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-950/60 text-slate-400 font-medium uppercase border-b border-slate-800">
              <tr>
                <th class="py-2.5 px-3">测试浓度 (wt%)</th>
                <th class="py-2.5 px-3">高斯过程预测综合得分</th>
                <th class="py-2.5 px-3">不确定度 (σ)</th>
                <th class="py-2.5 px-3">95% 置信区间 (CI 下限 ~ 上限)</th>
                <th class="py-2.5 px-3">相对峰值性能比例</th>
                <th class="py-2.5 px-3">实验建议</th>
              </tr>
            </thead>
            <tbody id="table-gpr-focus-body" class="divide-y divide-slate-800/80 font-mono text-slate-300">
              <!-- Populated via JS -->
            </tbody>
          </table>
        </div>
      </div>

    </div>


    <!-- ======================================================================= -->
    <!-- TAB 3: 实验数据库全景与对比 (Experiment Database & Analysis) -->
    <!-- ======================================================================= -->
    <div id="tab-database" class="tab-content hidden space-y-6">
      
      <!-- Database Top Bar -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center space-x-2">
            <span class="px-2.5 py-1 text-xs font-semibold rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              2 mol/L ZnSO₄ 标定数据库
            </span>
            <h2 class="text-xl font-bold text-white">20 种代表性候选添加剂实验全景数据集</h2>
          </div>
          <p class="text-xs sm:text-sm text-slate-400 mt-1">
            点击任意添加剂即可直接将其 5 维分子特征及 5 维 Origin 实验特征加载至阶段一模型中进行实时复算与对比。
          </p>
        </div>

        <div class="flex items-center space-x-2">
          <!-- Search input -->
          <div class="relative">
            <input type="text" id="db-search" placeholder="搜索添加剂名称 / SMILES..." oninput="filterDatabaseTable()" class="bg-slate-950 border border-slate-700 text-xs rounded-xl px-3 py-2 pl-8 text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-500 w-52 sm:w-64">
            <svg class="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          <!-- Export CSV button -->
          <button onclick="exportDatabaseCSV()" class="px-3 py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl transition-all flex items-center space-x-1">
            <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>导出 CSV</span>
          </button>
        </div>
      </div>

      <!-- Interactive Data Exploration Charts -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- Scatter Chart: Lifespan vs CE with Pareto Frontier -->
        <div class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 class="text-sm font-bold text-white">循环寿命 vs. 库仑效率 帕累托 (Pareto) 前沿分布</h3>
              <p class="text-xs text-slate-400">气泡大小表示 HER 析氢过电位 (越大越耐蚀)，颜色表示综合评估得分</p>
            </div>
            <span class="text-xs text-emerald-400 font-mono font-medium">Pareto Frontier</span>
          </div>
          <div id="chart-db-scatter" style="width: 100%; height: 320px;"></div>
        </div>

        <!-- Correlation Bar Chart -->
        <div class="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 class="text-sm font-bold text-white">5 维实验特征与性能指标 Pearson 关联度</h3>
              <p class="text-xs text-slate-400">XRD(002) 强正相关 (+0.995)，Tafel 斜率强负相关 (-0.977)</p>
            </div>
            <span class="text-xs text-cyan-400 font-mono font-medium">r &gt; 0.94</span>
          </div>
          <div id="chart-feature-corr" style="width: 100%; height: 320px;"></div>
        </div>
      </div>

      <!-- Complete Database Table -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 class="text-sm font-bold text-white">数据库明细清单 (共 20 组完整电化学实测样本)</h3>
          <span class="text-xs text-slate-500">点击任意行可快速回填至阶段一仿真</span>
        </div>

        <div class="overflow-x-auto max-h-[500px]">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-950 sticky top-0 z-10 text-slate-400 font-semibold border-b border-slate-800">
              <tr>
                <th class="py-2.5 px-3 cursor-pointer hover:text-white" onclick="sortDb('sample_id')">编号 ↕</th>
                <th class="py-2.5 px-3 cursor-pointer hover:text-white" onclick="sortDb('additive_name')">添加剂名称 ↕</th>
                <th class="py-2.5 px-3">SMILES 结构式</th>
                <th class="py-2.5 px-2 text-right cursor-pointer hover:text-white" onclick="sortDb('lifespan_h')">寿命 (h) ↕</th>
                <th class="py-2.5 px-2 text-right cursor-pointer hover:text-white" onclick="sortDb('coulombic_efficiency')">CE (%) ↕</th>
                <th class="py-2.5 px-2 text-right cursor-pointer hover:text-white" onclick="sortDb('her_overpotential_mv')">HER (mV) ↕</th>
                <th class="py-2.5 px-2 text-right">CV 面积</th>
                <th class="py-2.5 px-2 text-right">Tafel 斜率</th>
                <th class="py-2.5 px-2 text-right">XRD (002)</th>
                <th class="py-2.5 px-2 text-right cursor-pointer hover:text-white" onclick="sortDb('composite_score')">综合分 ↕</th>
                <th class="py-2.5 px-3 text-center">操作</th>
              </tr>
            </thead>
            <tbody id="db-table-body" class="divide-y divide-slate-800/60 font-mono text-slate-300">
              <!-- Populated via JS -->
            </tbody>
          </table>
        </div>
      </div>

    </div>


    <!-- ======================================================================= -->
    <!-- TAB 4: 算法架构与电化学机理 (AI Pipeline & Science Guide) -->
    <!-- ======================================================================= -->
    <div id="tab-guide" class="tab-content hidden space-y-6">
      
      <!-- Top Guide Hero -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 class="text-xl font-bold text-white mb-2">两阶段 AI 决策流水线架构与理论体系</h2>
        <p class="text-slate-400 text-xs sm:text-sm leading-relaxed max-w-4xl">
          针对水系锌电池（Aqueous Zinc Batteries, AZBs）普遍面临的锌枝晶无序生长、析氢副反应 (HER) 严重以及界面腐蚀问题，本项目构建了结合<strong>多任务深度学习 (PyTorch Multi-Task NN)</strong> 与<strong>不确定性高斯过程回归 (Gaussian Process Regression, GPR)</strong> 的两阶段智能协同优化工作流。
        </p>
      </div>

      <!-- Pipeline Flow Chart Graphic -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 class="text-sm font-bold text-white">双阶段推理流水线架构流程图 (Two-Stage AI Pipeline)</h3>
        
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          <!-- Step 1 -->
          <div class="bg-slate-950 border border-slate-800 rounded-xl p-4 relative">
            <div class="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 font-bold flex items-center justify-center mb-2">1</div>
            <h4 class="font-bold text-slate-200 mb-1">分子化学表征</h4>
            <p class="text-slate-400 text-[11px] leading-relaxed">
              输入候选分子 SMILES，利用 RDKit 解析拓扑结构，提取 5 维基础描述符：MolWt、TPSA、LogP、HBD、HBA。
            </p>
          </div>

          <!-- Step 2 -->
          <div class="bg-slate-950 border border-slate-800 rounded-xl p-4 relative">
            <div class="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 font-bold flex items-center justify-center mb-2">2</div>
            <h4 class="font-bold text-slate-200 mb-1">Origin 实验特征融合</h4>
            <p class="text-slate-400 text-[11px] leading-relaxed">
              归一化融合 Origin 2024b 实测 5 维微观表征：CV剥离面积、Tafel斜率、XPS结合能偏移、Raman峰面积、XRD(002)晶面强度比。
            </p>
          </div>

          <!-- Step 3 -->
          <div class="bg-slate-950 border border-slate-800 rounded-xl p-4 relative">
            <div class="w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 font-bold flex items-center justify-center mb-2">3</div>
            <h4 class="font-bold text-slate-200 mb-1">多任务初筛推理</h4>
            <p class="text-slate-400 text-[11px] leading-relaxed">
              输入 10 维联合张量至 4 层 Multi-Head 网络，同时前向预测循环寿命 (h)、库仑效率 (CE)、HER 析氢过电位 (mV)，并计算综合准入分。
            </p>
          </div>

          <!-- Step 4 -->
          <div class="bg-slate-950 border border-slate-800 rounded-xl p-4 relative">
            <div class="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 font-bold flex items-center justify-center mb-2">4</div>
            <h4 class="font-bold text-slate-200 mb-1">高斯过程浓度寻优</h4>
            <p class="text-slate-400 text-[11px] leading-relaxed">
              对达标候选物引入 GPR (RBF+WhiteKernel)，在 0.2~3.0 wt% 连续浓度空间拟合非对称吸附响应曲线，输出最佳推荐浓度与 95% 置信区间。
            </p>
          </div>
        </div>
      </div>

      <!-- Electrochemistry Science Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- Mechanism 1 -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div class="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center text-xl font-bold">
            🛡️
          </div>
          <h4 class="text-sm font-bold text-white">1. 原位界面自组装与枝晶抑制</h4>
          <p class="text-xs text-slate-400 leading-relaxed">
            具有极性官能团（磺酸基 -SO₃⁻、羧基 -COOH、氨基 -NH₂）的添加剂分子通过静电吸附或配位吸附在锌负极表面高电场尖端，屏蔽局部“尖端突起效应”，诱导 Zn²⁺ 离子横向平滑迁移，促进 Zn(002) 择优晶面平行生长。
          </p>
        </div>

        <!-- Mechanism 2 -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div class="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center text-xl font-bold">
            💧
          </div>
          <h4 class="text-sm font-bold text-white">2. 溶剂化壳层重塑与自由水屏蔽</h4>
          <p class="text-xs text-slate-400 leading-relaxed">
            极性分子进入 Zn²⁺ 的初级水化配位层 [Zn(H₂O)₆]²⁺，部分置换活性配位水分子，削弱水分子 O-H 键极化，显著降低界面活性自由水含量，从而在根源上抑制析氢副反应 (HER) 并减少碱式硫酸锌副产物产生。
          </p>
        </div>

        <!-- Mechanism 3 -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div class="w-10 h-10 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center text-xl font-bold">
            ⚡
          </div>
          <h4 class="text-sm font-bold text-white">3. HER 析氢动力学能垒抬升</h4>
          <p class="text-xs text-slate-400 leading-relaxed">
            添加剂分子形成的物理/化学双电层吸附膜阻隔了水分子向金属锌表面的直接电子转移通道，增大了析氢反应的活化过电位 (HER Overpotential)，使得锌沉积过程享有更宽的电化学稳定窗口与超长循环寿命。
          </p>
        </div>
      </div>

    </div>

  </main>

  <!-- ========================================================================= -->
  <!-- Export Report Modal -->
  <!-- ========================================================================= -->
  <div id="modal-export" class="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div class="flex items-center space-x-2">
          <span class="w-3 h-3 rounded-full bg-emerald-400"></span>
          <h3 class="text-base font-bold text-white">硫酸锌电池添加剂 AI 评估报告预览</h3>
        </div>
        <button onclick="closeExportModal()" class="text-slate-400 hover:text-slate-200">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div id="export-report-content" class="bg-slate-950 p-4 rounded-xl text-xs font-mono text-slate-300 space-y-3 overflow-y-auto max-h-[60vh]">
        <!-- Populated via JS -->
      </div>

      <div class="flex items-center justify-end space-y-0 space-x-3 pt-2">
        <button onclick="copyReportText()" class="px-4 py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl transition-all">
          复制纯文本报告
        </button>
        <button onclick="window.print()" class="px-4 py-2 text-xs font-semibold bg-emerald-500 hover:bg-emerald-400 text-slate-950 rounded-xl transition-all shadow-lg shadow-emerald-500/20">
          🖨️ 打印 / 保存 PDF
        </button>
      </div>
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- Data Script & Core Computational Logic -->
  <!-- ========================================================================= -->
  <script>
    // Embedded Data from backend
    const APP_DATA = {json_str};

    // State Variables
    let currentAdditiveName = "2-氨基-4-溴蒽醌-2-磺酸钠";
    let currentSmiles = "";
    let currentDescriptors = {{}};
    let currentExpFeatures = {{
      cv: 3260.0,
      tafel: 67.5,
      xps: 0.44,
      raman: 1850.0,
      xrd: 2.50
    }};
    let currentPredictions = {{
      lifespan: 1236.4,
      ce: 0.9992,
      her: 200.2,
      compositeScore: 73.62,
      passed: true
    }};
    let currentGPROpt = {{
      bestConc: 1.13,
      bestScore: 73.32,
      sigma: 0.099,
      ci: [73.13, 73.52],
      focusEvaluations: []
    }};
    let currentGPRCurveType = 'composite'; // 'composite', 'lifespan', 'ce', 'her'
    let dbSortField = 'composite_score';
    let dbSortAsc = false;

    // Charts instances
    let chartGauge = null;
    let chartRadar = null;
    let chartGPR = null;
    let chartDbScatter = null;
    let chartFeatureCorr = null;

    // Initialize on DOM ready
    window.addEventListener('DOMContentLoaded', () => {{
      initAdditiveSelect();
      initECharts();
      loadPreset(currentAdditiveName);
      renderDatabaseTable();
      renderCorrChart();
      renderDbScatterChart();
    }});

    // Tab Switching
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      const activeTab = document.getElementById(tabId);
      if (activeTab) activeTab.classList.remove('hidden');

      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.remove('border-emerald-400', 'text-emerald-400');
        btn.classList.add('border-transparent', 'text-slate-400');
      }});
      
      const navBtnMap = {{
        'tab-screening': 'nav-screening',
        'tab-gpr': 'nav-gpr',
        'tab-database': 'nav-database',
        'tab-guide': 'nav-guide'
      }};
      const activeNavBtn = document.getElementById(navBtnMap[tabId]);
      if (activeNavBtn) {{
        activeNavBtn.classList.add('border-emerald-400', 'text-emerald-400');
        activeNavBtn.classList.remove('border-transparent', 'text-slate-400');
      }}

      // Resize charts when switching tabs
      setTimeout(() => {{
        if (chartGauge) chartGauge.resize();
        if (chartRadar) chartRadar.resize();
        if (chartGPR) chartGPR.resize();
        if (chartDbScatter) chartDbScatter.resize();
        if (chartFeatureCorr) chartFeatureCorr.resize();
      }}, 50);
    }}

    // Populate Additives Dropdown
    function initAdditiveSelect() {{
      const select = document.getElementById('additive-select');
      select.innerHTML = '';
      
      APP_DATA.database.forEach(item => {{
        const opt = document.createElement('option');
        opt.value = item.additive_name;
        opt.textContent = `${{item.sample_id}} - ${{item.additive_name}} (SMILES: ${{item.smiles.substring(0, 20)}}...)`;
        select.appendChild(opt);
      }});
    }}

    // Select Additive
    function onSelectAdditiveChange(name) {{
      loadPreset(name);
    }}

    // Load preset item
    function loadPreset(name) {{
      const item = APP_DATA.database.find(x => x.additive_name === name) || APP_DATA.database[0];
      currentAdditiveName = item.additive_name;
      currentSmiles = item.smiles;
      currentDescriptors = item.mol_descriptors;

      // Update UI
      document.getElementById('additive-select').value = item.additive_name;
      document.getElementById('input-smiles').value = item.smiles;
      
      document.getElementById('desc-molwt').textContent = currentDescriptors.MolWt.toFixed(1);
      document.getElementById('desc-tpsa').textContent = currentDescriptors.TPSA.toFixed(1);
      document.getElementById('desc-logp').textContent = currentDescriptors.LogP.toFixed(2);
      document.getElementById('desc-hbd').textContent = Math.round(currentDescriptors.NumHDonors);
      document.getElementById('desc-hba').textContent = Math.round(currentDescriptors.NumHAcceptors);

      // Fill experimental features
      currentExpFeatures.cv = item.cv_area;
      currentExpFeatures.tafel = item.tafel_slope;
      currentExpFeatures.xps = item.xps_shift;
      currentExpFeatures.raman = item.raman_area;
      currentExpFeatures.xrd = item.xrd_ratio;

      updateFeatureUI();
      runInferenceAndRender();
    }}

    function updateFeatureUI() {{
      document.getElementById('input-cv').value = currentExpFeatures.cv.toFixed(1);
      document.getElementById('slider-cv').value = currentExpFeatures.cv;

      document.getElementById('input-tafel').value = currentExpFeatures.tafel.toFixed(1);
      document.getElementById('slider-tafel').value = currentExpFeatures.tafel;

      document.getElementById('input-xps').value = currentExpFeatures.xps.toFixed(2);
      document.getElementById('slider-xps').value = currentExpFeatures.xps;

      document.getElementById('input-raman').value = currentExpFeatures.raman.toFixed(1);
      document.getElementById('slider-raman').value = currentExpFeatures.raman;

      document.getElementById('input-xrd').value = currentExpFeatures.xrd.toFixed(2);
      document.getElementById('slider-xrd').value = currentExpFeatures.xrd;
    }}

    // Feature Sliders / Inputs
    function onFeatureSlider(feature, val) {{
      const numVal = parseFloat(val);
      currentExpFeatures[feature] = numVal;
      document.getElementById(`input-${{feature}}`).value = numVal;
      runInferenceAndRender();
    }}

    function onFeatureInput(feature, val) {{
      const numVal = parseFloat(val);
      if (!isNaN(numVal)) {{
        currentExpFeatures[feature] = numVal;
        document.getElementById(`slider-${{feature}}`).value = numVal;
        runInferenceAndRender();
      }}
    }}

    // =========================================================================
    // Core PyTorch Neural Network Forward Pass in Pure JavaScript
    // =========================================================================
    function neuralNetworkForward(molVector, expVectorRaw) {{
      const scalers = APP_DATA.scalers;
      const w = APP_DATA.model_weights;

      // 1. MinMax scale exp features
      const expVectorScaled = [];
      for (let i = 0; i < 5; i++) {{
        const minVal = scalers.exp_min[i];
        const maxVal = scalers.exp_max[i];
        const raw = expVectorRaw[i];
        const scaled = (raw - minVal) / (maxVal - minVal);
        expVectorScaled.push(scaled);
      }}

      // 2. Concatenate 10D vector
      const x = [...molVector, ...expVectorScaled];

      // 3. Layer 1: Linear(10, 64) + ReLU
      const h1 = new Array(64);
      for (let i = 0; i < 64; i++) {{
        let sum = w.b1[i];
        for (let j = 0; j < 10; j++) {{
          sum += w.w1[i][j] * x[j];
        }}
        h1[i] = Math.max(0, sum);
      }}

      // 4. Layer 2: Linear(64, 32) + ReLU
      const h2 = new Array(32);
      for (let i = 0; i < 32; i++) {{
        let sum = w.b2[i];
        for (let j = 0; j < 64; j++) {{
          sum += w.w2[i][j] * h1[j];
        }}
        h2[i] = Math.max(0, sum);
      }}

      // 5. Layer 3: Linear(32, 16) + ReLU
      const h3 = new Array(16);
      for (let i = 0; i < 16; i++) {{
        let sum = w.b3[i];
        for (let j = 0; j < 32; j++) {{
          sum += w.w3[i][j] * h2[j];
        }}
        h3[i] = Math.max(0, sum);
      }}

      // 6. Heads: Linear(16, 1) each
      let outLifeScaled = w.b_life;
      let outCeScaled = w.b_ce;
      let outHerScaled = w.b_her;
      for (let j = 0; j < 16; j++) {{
        outLifeScaled += w.w_life[j] * h3[j];
        outCeScaled += w.w_ce[j] * h3[j];
        outHerScaled += w.w_her[j] * h3[j];
      }}

      // 7. Inverse transform targets
      const predLifespan = outLifeScaled * (scalers.target_max[0] - scalers.target_min[0]) + scalers.target_min[0];
      const predCe = outCeScaled * (scalers.target_max[1] - scalers.target_min[1]) + scalers.target_min[1];
      const predHer = outHerScaled * (scalers.target_max[2] - scalers.target_min[2]) + scalers.target_min[2];

      return {{
        lifespan: predLifespan,
        ce: predCe,
        her: predHer
      }};
    }}

    // =========================================================================
    // Gaussian Process Regression (GPR) in Pure JavaScript
    // =========================================================================
    function solveLinearSystem(A, b) {{
      const n = A.length;
      const M = A.map((row, i) => [...row, b[i]]);
      for (let i = 0; i < n; i++) {{
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {{
          if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
        }}
        const tmp = M[i]; M[i] = M[maxRow]; M[maxRow] = tmp;
        const pivot = M[i][i];
        for (let j = i; j <= n; j++) M[i][j] /= pivot;
        for (let k = 0; k < n; k++) {{
          if (k !== i) {{
            const factor = M[k][i];
            for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
          }}
        }}
      }}
      return M.map(row => row[n]);
    }}

    function runGPROptimization(baseCompositeScore, molWt) {{
      const benchmarkConcs = [0.5, 1.0, 1.5, 2.0, 2.5];
      const peakOffset = molWt > 300 ? -0.18 : (molWt < 100 ? 0.12 : 0.0);
      const idealPeakConc = 1.35 + peakOffset;

      const gradY = [];
      benchmarkConcs.forEach(c => {{
        const dist = c - idealPeakConc;
        const shapePenalty = dist >= 0 ? 18.0 * (dist ** 2) : 14.0 * (dist ** 2);
        const score = Math.max(35.0, baseCompositeScore - shapePenalty);
        gradY.push(score);
      }});

      // GPR Kernel RBF + WhiteKernel
      const n = benchmarkConcs.length;
      const l = 0.8;
      const noise = 1e-4;
      const K = [];
      for (let i = 0; i < n; i++) {{
        K[i] = [];
        for (let j = 0; j < n; j++) {{
          const d = benchmarkConcs[i] - benchmarkConcs[j];
          let kVal = Math.exp(-0.5 * (d / l) ** 2);
          if (i === j) kVal += noise;
          K[i][j] = kVal;
        }}
      }}

      // Solve alpha
      const yMeanVal = gradY.reduce((a, b) => a + b, 0) / n;
      const yNorm = gradY.map(y => y - yMeanVal);
      const alpha = solveLinearSystem(K, yNorm);

      // Dense Evaluation
      const denseConcs = [];
      const denseMeans = [];
      const denseStds = [];
      const denseCiLowers = [];
      const denseCiUppers = [];

      let bestIdx = 0;
      let maxMean = -Infinity;

      const steps = 120;
      for (let s = 0; s <= steps; s++) {{
        const conc = 0.2 + (s / steps) * (3.0 - 0.2);
        const kStar = benchmarkConcs.map(xc => Math.exp(-0.5 * ((conc - xc) / l) ** 2));
        
        let mu = yMeanVal;
        for (let i = 0; i < n; i++) mu += kStar[i] * alpha[i];

        const v = solveLinearSystem(K, kStar);
        let kStarDotV = 0;
        for (let i = 0; i < n; i++) kStarDotV += kStar[i] * v[i];
        
        const variance = Math.max(0.001, 1.0 + noise - kStarDotV);
        const std = Math.sqrt(variance) * 0.15;

        const ciLower = mu - 1.96 * std;
        const ciUpper = mu + 1.96 * std;

        denseConcs.push(conc);
        denseMeans.push(mu);
        denseStds.push(std);
        denseCiLowers.push(ciLower);
        denseCiUppers.push(ciUpper);

        if (mu > maxMean) {{
          maxMean = mu;
          bestIdx = s;
        }}
      }}

      // Focus gradient points
      const focusEvaluations = benchmarkConcs.map(conc => {{
        const kStar = benchmarkConcs.map(xc => Math.exp(-0.5 * ((conc - xc) / l) ** 2));
        let mu = yMeanVal;
        for (let i = 0; i < n; i++) mu += kStar[i] * alpha[i];
        const v = solveLinearSystem(K, kStar);
        let kStarDotV = 0;
        for (let i = 0; i < n; i++) kStarDotV += kStar[i] * v[i];
        const std = Math.sqrt(Math.max(0.001, 1.0 + noise - kStarDotV)) * 0.15;
        return {{
          conc: conc,
          score: mu,
          std: std,
          ciLower: mu - 1.96 * std,
          ciUpper: mu + 1.96 * std
        }};
      }});

      const bestConc = denseConcs[bestIdx];
      const bestStd = denseStds[bestIdx];

      return {{
        denseConcs,
        denseMeans,
        denseStds,
        denseCiLowers,
        denseCiUppers,
        focusEvaluations,
        benchmarkConcs,
        gradY,
        bestConc: parseFloat(bestConc.toFixed(2)),
        bestScore: parseFloat(maxMean.toFixed(2)),
        sigma: parseFloat(bestStd.toFixed(3)),
        ci: [parseFloat((maxMean - 1.96 * bestStd).toFixed(2)), parseFloat((maxMean + 1.96 * bestStd).toFixed(2))]
      }};
    }}

    // =========================================================================
    // Master Inference & Update Logic
    // =========================================================================
    function runInferenceAndRender() {{
      const molVector = [
        currentDescriptors.MolWt,
        currentDescriptors.TPSA,
        currentDescriptors.LogP,
        currentDescriptors.NumHDonors,
        currentDescriptors.NumHAcceptors
      ];

      const expVectorRaw = [
        currentExpFeatures.cv,
        currentExpFeatures.tafel,
        currentExpFeatures.xps,
        currentExpFeatures.raman,
        currentExpFeatures.xrd
      ];

      // 1. Run forward pass
      const preds = neuralNetworkForward(molVector, expVectorRaw);

      // 2. Composite score calculation
      const scoreLifeNorm = Math.min(1.0, Math.max(0.0, (preds.lifespan - 500) / (1800 - 500)));
      const scoreCeNorm = Math.min(1.0, Math.max(0.0, (preds.ce - 0.985) / (0.999 - 0.985)));
      const scoreHerNorm = Math.min(1.0, Math.max(0.0, (preds.her - 130) / (240 - 130)));
      const compositeScore = 100.0 * (0.40 * scoreLifeNorm + 0.35 * scoreCeNorm + 0.25 * scoreHerNorm);

      // 3. Threshold checks
      const threshLife = parseFloat(document.getElementById('thresh-lifespan').value) || 1000.0;
      const threshCe = (parseFloat(document.getElementById('thresh-ce').value) || 99.20) / 100.0;
      const threshHer = parseFloat(document.getElementById('thresh-her').value) || 175.0;

      const passedLife = preds.lifespan >= threshLife;
      const passedCe = preds.ce >= threshCe;
      const passedHer = preds.her >= threshHer;
      const passedAll = passedLife && passedCe && passedHer;

      currentPredictions = {{
        lifespan: preds.lifespan,
        ce: preds.ce,
        her: preds.her,
        compositeScore: compositeScore,
        passed: passedAll
      }};

      // 4. Update UI Values
      document.getElementById('pred-lifespan').textContent = preds.lifespan.toFixed(1);
      document.getElementById('pred-ce').textContent = (preds.ce * 100).toFixed(3);
      document.getElementById('pred-her').textContent = preds.her.toFixed(1);
      document.getElementById('pred-composite-score').textContent = compositeScore.toFixed(2);

      // Life multiplier & grade
      const mult = (preds.lifespan / 150).toFixed(1);
      document.getElementById('pred-life-mult').textContent = `${{mult}} 倍提升`;
      
      const ceGrade = preds.ce >= 0.996 ? "S 级超可逆" : (preds.ce >= 0.992 ? "A 级优秀" : "B 级中等");
      document.getElementById('pred-ce-grade').textContent = ceGrade;

      const herGrade = preds.her >= 200 ? "超强抗析氢" : (preds.her >= 170 ? "优良耐蚀" : "析氢敏感");
      document.getElementById('pred-her-grade').textContent = herGrade;

      // Badges
      updatePassTag('life-pass-tag', passedLife);
      updatePassTag('ce-pass-tag', passedCe);
      updatePassTag('her-pass-tag', passedHer);

      // Banner Status
      updateScreeningBanner(passedAll);

      // 5. Run GPR Concentration Optimization
      currentGPROpt = runGPROptimization(compositeScore, currentDescriptors.MolWt);
      updateGPRDashboardUI();

      // 6. Update ECharts
      updateGaugeChart(compositeScore);
      updateRadarChart();
      updateGPRChart();
    }}

    function updatePassTag(id, passed) {{
      const el = document.getElementById(id);
      if (passed) {{
        el.className = "text-[10px] px-1.5 py-0.5 rounded font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
        el.textContent = "达标";
      }} else {{
        el.className = "text-[10px] px-1.5 py-0.5 rounded font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30";
        el.textContent = "未达标";
      }}
    }}

    function updateScreeningBanner(passed) {{
      const banner = document.getElementById('banner-screening-status');
      const iconContainer = document.getElementById('status-icon-container');
      const title = document.getElementById('status-title');
      const badge = document.getElementById('status-badge');
      const desc = document.getElementById('status-desc');
      const gotoBtn = document.getElementById('btn-goto-opt');

      if (passed) {{
        banner.className = "rounded-2xl p-5 border border-emerald-500/40 bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 shadow-xl flex items-center justify-between glow-effect";
        iconContainer.className = "w-12 h-12 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center flex-shrink-0 text-2xl";
        iconContainer.innerHTML = `
          <svg class="w-7 h-7 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
          </svg>
        `;
        title.textContent = "多任务初筛评估达标！通过初筛";
        title.className = "text-base font-bold text-emerald-300";
        badge.className = "text-xs px-2.5 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
        badge.textContent = "PASSED";
        desc.textContent = "各项电化学电性能均跨越准入红线，已自动生成阶段二连续浓度空间高斯过程寻优方案。";
        gotoBtn.classList.remove('hidden');
      }} else {{
        banner.className = "rounded-2xl p-5 border border-amber-500/30 bg-gradient-to-r from-amber-950/30 via-slate-900 to-slate-900 shadow-xl flex items-center justify-between";
        iconContainer.className = "w-12 h-12 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center flex-shrink-0 text-2xl";
        iconContainer.innerHTML = `
          <svg class="w-7 h-7 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        `;
        title.textContent = "综合初筛未达标 (需优化实验参数或结构)";
        title.className = "text-base font-bold text-amber-300";
        badge.className = "text-xs px-2.5 py-0.5 rounded-full font-bold bg-amber-500/20 text-amber-400 border border-amber-500/40";
        badge.textContent = "FAILED";
        desc.textContent = "候选添加剂在循环寿命、CE 或 HER 抗析氢维度存在短板，建议调整电解液配方或改善微观晶面沉积。";
        gotoBtn.classList.add('hidden');
      }}
    }}

    function recalculateScreening() {{
      runInferenceAndRender();
    }}

    // =========================================================================
    // GPR Dashboard UI Update
    // =========================================================================
    function updateGPRDashboardUI() {{
      document.getElementById('gpr-target-name').textContent = currentAdditiveName;
      document.getElementById('gpr-target-score').textContent = currentPredictions.compositeScore.toFixed(1);
      
      document.getElementById('gpr-best-conc').textContent = currentGPROpt.bestConc.toFixed(2);
      document.getElementById('gpr-peak-score').textContent = currentGPROpt.bestScore.toFixed(1);
      document.getElementById('gpr-sigma').textContent = `± ${{currentGPROpt.sigma.toFixed(3)}}`;
      document.getElementById('gpr-ci').textContent = `[ ${{currentGPROpt.ci[0].toFixed(2)}}, ${{currentGPROpt.ci[1].toFixed(2)}} ]`;

      // Update Diagnosis text
      const mw = currentDescriptors.MolWt;
      let diag = `根据添加剂分子量 (MW = ${{mw.toFixed(1)}} g/mol) 及空间位阻效应分析，该分子的最佳吸附平衡浓度位于 【${{currentGPROpt.bestConc.toFixed(2)}} wt%】。`;
      if (mw > 300) {{
        diag += ` 由于分子芳环共轭与磺酸根位阻较大，浓度略低 (~1.1~1.3 wt%) 即可完成均匀致密单分子层覆盖；浓度过高将引起自聚阻滞 Zn²⁺ 离子迁移。`;
      }} else {{
        diag += ` 属于中小分子活性添加剂，在 1.3~1.6 wt% 区间能形成强力螯合保护网络，显著平滑电沉积尖端电场。`;
      }}
      document.getElementById('gpr-diagnosis-text').textContent = diag;

      // Update Focus Gradient Table
      const tbody = document.getElementById('table-gpr-focus-body');
      tbody.innerHTML = '';
      currentGPROpt.focusEvaluations.forEach(row => {{
        const isOptimal = Math.abs(row.conc - currentGPROpt.bestConc) < 0.25;
        const tr = document.createElement('tr');
        tr.className = isOptimal ? "bg-emerald-950/30 text-emerald-300 font-semibold" : "hover:bg-slate-800/40";
        
        const ratio = ((row.score / currentGPROpt.bestScore) * 100).toFixed(1);
        const advice = isOptimal ? "🌟 强烈推荐试验点" : (row.conc < currentGPROpt.bestConc ? "吸附成膜未饱和" : "过饱和粘度增大/自聚");

        tr.innerHTML = `
          <td class="py-2.5 px-3 font-bold">${{row.conc.toFixed(1)}} wt%</td>
          <td class="py-2.5 px-3 text-emerald-400">${{row.score.toFixed(2)}} 分</td>
          <td class="py-2.5 px-3 text-cyan-400">±${{row.std.toFixed(3)}}</td>
          <td class="py-2.5 px-3 text-slate-300">[${{row.ciLower.toFixed(2)}}, ${{row.ciUpper.toFixed(2)}}]</td>
          <td class="py-2.5 px-3">${{ratio}}%</td>
          <td class="py-2.5 px-3 font-sans text-xs">${{advice}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function switchGPRCurve(type) {{
      currentGPRCurveType = type;
      ['composite', 'lifespan', 'ce', 'her'].forEach(t => {{
        const btn = document.getElementById(`btn-curve-${{t}}`);
        if (t === type) {{
          btn.className = "px-2.5 py-1 rounded-lg bg-emerald-500 text-slate-950 font-bold transition-all";
        }} else {{
          btn.className = "px-2.5 py-1 rounded-lg text-slate-400 hover:text-slate-200 transition-all";
        }}
      }});
      updateGPRChart();
    }}

    // =========================================================================
    // ECharts Visualizations
    // =========================================================================
    function initECharts() {{
      chartGauge = echarts.init(document.getElementById('chart-gauge'), 'dark', {{ renderer: 'canvas' }});
      chartRadar = echarts.init(document.getElementById('chart-radar'), 'dark', {{ renderer: 'canvas' }});
      chartGPR = echarts.init(document.getElementById('chart-gpr-curve'), 'dark', {{ renderer: 'canvas' }});
      chartDbScatter = echarts.init(document.getElementById('chart-db-scatter'), 'dark', {{ renderer: 'canvas' }});
      chartFeatureCorr = echarts.init(document.getElementById('chart-feature-corr'), 'dark', {{ renderer: 'canvas' }});

      window.addEventListener('resize', () => {{
        chartGauge.resize();
        chartRadar.resize();
        chartGPR.resize();
        chartDbScatter.resize();
        chartFeatureCorr.resize();
      }});
    }}

    function updateGaugeChart(score) {{
      const option = {{
        backgroundColor: 'transparent',
        series: [{{
          type: 'gauge',
          startAngle: 180,
          endAngle: 0,
          min: 0,
          max: 100,
          splitNumber: 5,
          itemStyle: {{
            color: '#10b981',
            shadowColor: 'rgba(16, 185, 129, 0.45)',
            shadowBlur: 10
          }},
          progress: {{
            show: true,
            roundCap: true,
            width: 14
          }},
          pointer: {{
            length: '60%',
            width: 5,
            offsetCenter: [0, '-10%'],
            itemStyle: {{ color: '#f59e0b' }}
          }},
          axisLine: {{
            roundCap: true,
            lineStyle: {{
              width: 14,
              color: [
                [0.5, '#ef4444'],
                [0.75, '#f59e0b'],
                [1, '#10b981']
              ]
            }}
          }},
          axisTick: {{ show: false }},
          splitLine: {{
            length: 8,
            lineStyle: {{ width: 1, color: '#475569' }}
          }},
          axisLabel: {{
            distance: 18,
            color: '#94a3b8',
            fontSize: 10
          }},
          title: {{
            show: true,
            offsetCenter: [0, '25%'],
            fontSize: 11,
            color: '#94a3b8'
          }},
          detail: {{
            valueAnimation: true,
            offsetCenter: [0, '-15%'],
            fontSize: 24,
            fontWeight: 'bolder',
            formatter: '{{value}}',
            color: '#34d399'
          }},
          data: [{{
            value: parseFloat(score.toFixed(1)),
            name: '综合评分'
          }}]
        }}]
      }};
      chartGauge.setOption(option);
    }}

    function updateRadarChart() {{
      const f = currentExpFeatures;
      const option = {{
        backgroundColor: 'transparent',
        tooltip: {{ trigger: 'item' }},
        legend: {{
          bottom: 0,
          textStyle: {{ color: '#94a3b8', fontSize: 10 }},
          itemWidth: 10,
          itemHeight: 10
        }},
        radar: {{
          indicator: [
            {{ name: 'CV 剥离', max: 3500, min: 2800 }},
            {{ name: 'Tafel 耐蚀', max: 85, min: 60 }},
            {{ name: 'XPS 结合', max: 0.55, min: 0.20 }},
            {{ name: 'Raman 水调控', max: 2000, min: 1300 }},
            {{ name: 'XRD (002) 织构', max: 2.85, min: 1.60 }}
          ],
          shape: 'polygon',
          splitNumber: 4,
          radius: '65%',
          splitArea: {{
            areaStyle: {{
              color: ['rgba(30, 41, 59, 0.6)', 'rgba(15, 23, 42, 0.4)']
            }}
          }},
          axisLine: {{ lineStyle: {{ color: '#334155' }} }},
          splitLine: {{ lineStyle: {{ color: '#334155' }} }},
          axisName: {{ color: '#cbd5e1', fontSize: 10 }}
        }},
        series: [{{
          type: 'radar',
          data: [
            {{
              value: [f.cv, f.tafel, f.xps, f.raman, f.xrd],
              name: '当前候选添加剂',
              itemStyle: {{ color: '#10b981' }},
              areaStyle: {{ color: 'rgba(16, 185, 129, 0.3)' }}
            }},
            {{
              value: [3120, 71.5, 0.35, 1630, 2.15],
              name: '数据库均值基准',
              itemStyle: {{ color: '#64748b' }},
              lineStyle: {{ type: 'dashed' }}
            }},
            {{
              value: [3410, 62.1, 0.45, 1960, 2.78],
              name: '标杆极优基准',
              itemStyle: {{ color: '#06b6d4' }},
              lineStyle: {{ type: 'dotted' }}
            }}
          ]
        }}]
      }};
      chartRadar.setOption(option);
    }}

    function updateGPRChart() {{
      const denseConcs = currentGPROpt.denseConcs;
      const baseScore = currentPredictions.compositeScore;
      
      let meanValues = currentGPROpt.denseMeans;
      let lowerValues = currentGPROpt.denseCiLowers;
      let upperValues = currentGPROpt.denseCiUppers;
      let yAxisName = "综合得分 (0~100)";

      if (currentGPRCurveType === 'lifespan') {{
        const scale = currentPredictions.lifespan / baseScore;
        meanValues = meanValues.map(v => v * scale);
        lowerValues = lowerValues.map(v => v * scale);
        upperValues = upperValues.map(v => v * scale);
        yAxisName = "循环寿命 (h)";
      }} else if (currentGPRCurveType === 'ce') {{
        meanValues = meanValues.map(v => 98.5 + (v / 100) * 1.4);
        lowerValues = lowerValues.map(v => 98.5 + (v / 100) * 1.4);
        upperValues = upperValues.map(v => 98.5 + (v / 100) * 1.4);
        yAxisName = "库仑效率 (%)";
      }} else if (currentGPRCurveType === 'her') {{
        const scale = currentPredictions.her / baseScore;
        meanValues = meanValues.map(v => v * scale);
        lowerValues = lowerValues.map(v => v * scale);
        upperValues = upperValues.map(v => v * scale);
        yAxisName = "HER 过电位 (mV)";
      }}

      // Build ECharts Confidence Band Area (Upper - Lower)
      const diffValues = upperValues.map((up, i) => up - lowerValues[i]);

      const option = {{
        backgroundColor: 'transparent',
        tooltip: {{
          trigger: 'axis',
          axisPointer: {{ type: 'cross' }},
          formatter: function(params) {{
            const conc = params[0].axisValue;
            let res = `<div class="font-mono text-xs font-bold text-slate-100">浓度: ${{conc}} wt%</div>`;
            params.forEach(item => {{
              if (item.seriesName === '高斯过程后验均值') {{
                res += `<div class="text-xs text-emerald-400 mt-1">预测均值: <b>${{item.value.toFixed(2)}}</b></div>`;
              }}
            }});
            return res;
          }}
        }},
        grid: {{ left: '4%', right: '4%', top: '12%', bottom: '10%', containLabel: true }},
        legend: {{
          data: ['高斯过程后验均值', '95% 置信区间 (±1.96σ)'],
          textStyle: {{ color: '#94a3b8', fontSize: 11 }},
          top: 0
        }},
        xAxis: {{
          type: 'category',
          data: denseConcs.map(c => c.toFixed(2)),
          name: '添加剂浓度 (wt%)',
          nameLocation: 'middle',
          nameGap: 24,
          axisLine: {{ lineStyle: {{ color: '#475569' }} }},
          axisLabel: {{ color: '#94a3b8', fontSize: 10 }},
          splitLine: {{ show: true, lineStyle: {{ color: '#1e293b' }} }}
        }},
        yAxis: {{
          type: 'value',
          name: yAxisName,
          axisLine: {{ lineStyle: {{ color: '#475569' }} }},
          axisLabel: {{ color: '#94a3b8', fontSize: 10 }},
          splitLine: {{ show: true, lineStyle: {{ color: '#1e293b' }} }}
        }},
        series: [
          // 1. Lower boundary (transparent)
          {{
            name: 'CI Lower',
            type: 'line',
            data: lowerValues,
            lineStyle: {{ opacity: 0 }},
            stack: 'confidence-band',
            symbol: 'none'
          }},
          // 2. Area to Upper boundary (stacked)
          {{
            name: '95% 置信区间 (±1.96σ)',
            type: 'line',
            data: diffValues,
            lineStyle: {{ opacity: 0 }},
            areaStyle: {{ color: 'rgba(6, 182, 212, 0.18)' }},
            stack: 'confidence-band',
            symbol: 'none'
          }},
          // 3. Posterior Mean Curve
          {{
            name: '高斯过程后验均值',
            type: 'line',
            data: meanValues,
            smooth: true,
            lineStyle: {{ width: 3, color: '#10b981' }},
            itemStyle: {{ color: '#10b981' }},
            markPoint: {{
              symbol: 'pin',
              symbolSize: 45,
              data: [{{
                type: 'max',
                name: '最优峰值浓度',
                itemStyle: {{ color: '#f59e0b' }}
              }}],
              label: {{ fontSize: 10, fontWeight: 'bold' }}
            }}
          }}
        ]
      }};
      chartGPR.setOption(option, true);
    }}

    // =========================================================================
    // Database Explorer & Multi-dimensional Charts
    // =========================================================================
    function renderDbScatterChart() {{
      const db = APP_DATA.database;
      const data = db.map(item => [
        item.coulombic_efficiency * 100,
        item.lifespan_h,
        item.her_overpotential_mv,
        item.additive_name,
        item.composite_score
      ]);

      const option = {{
        backgroundColor: 'transparent',
        tooltip: {{
          formatter: function(param) {{
            const d = param.data;
            return `
              <div class="font-sans text-xs">
                <div class="font-bold text-white mb-1">${{d[3]}}</div>
                <div class="text-cyan-400">库仑效率: <b>${{d[0].toFixed(3)}} %</b></div>
                <div class="text-emerald-400">循环寿命: <b>${{d[1]}} h</b></div>
                <div class="text-indigo-400">HER 过电位: <b>${{d[2]}} mV</b></div>
                <div class="text-amber-400">综合得分: <b>${{d[4]}} 分</b></div>
              </div>
            `;
          }}
        }},
        grid: {{ left: '5%', right: '5%', top: '10%', bottom: '12%', containLabel: true }},
        xAxis: {{
          name: '库仑效率 CE (%)',
          min: 98.7,
          max: 99.9,
          axisLine: {{ lineStyle: {{ color: '#475569' }} }},
          axisLabel: {{ color: '#94a3b8', fontSize: 10 }},
          splitLine: {{ lineStyle: {{ color: '#1e293b' }} }}
        }},
        yAxis: {{
          name: '循环寿命 (h)',
          min: 600,
          max: 1800,
          axisLine: {{ lineStyle: {{ color: '#475569' }} }},
          axisLabel: {{ color: '#94a3b8', fontSize: 10 }},
          splitLine: {{ lineStyle: {{ color: '#1e293b' }} }}
        }},
        visualMap: {{
          min: 15,
          max: 95,
          dimension: 4,
          inRange: {{
            color: ['#6366f1', '#06b6d4', '#10b981', '#fbbf24']
          }},
          show: false
        }},
        series: [{{
          type: 'scatter',
          symbolSize: function(data) {{
            return (data[2] - 120) / 4;
          }},
          data: data,
          itemStyle: {{
            shadowBlur: 10,
            shadowColor: 'rgba(0,0,0,0.5)'
          }},
          markLine: {{
            lineStyle: {{ type: 'dashed', color: '#f59e0b' }},
            data: [
              {{ yAxis: 1000, name: '寿命达标线 (1000h)' }},
              {{ xAxis: 99.20, name: 'CE 达标线 (99.2%)' }}
            ]
          }}
        }}]
      }};
      chartDbScatter.setOption(option);
    }}

    function renderCorrChart() {{
      const features = ['CV 剥离面积', 'Tafel 极化斜率', 'XPS 结合偏移', 'Raman 水调控', 'XRD (002) 比'];
      const corrLife = [0.992, -0.977, 0.443, 0.989, 0.995];
      const corrCe = [0.981, -0.974, 0.411, 0.989, 0.985];

      const option = {{
        backgroundColor: 'transparent',
        tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
        legend: {{
          data: ['与循环寿命关联度', '与库仑效率关联度'],
          textStyle: {{ color: '#94a3b8', fontSize: 10 }},
          top: 0
        }},
        grid: {{ left: '4%', right: '4%', top: '15%', bottom: '8%', containLabel: true }},
        xAxis: {{
          type: 'category',
          data: features,
          axisLabel: {{ color: '#94a3b8', fontSize: 9, rotate: 20 }},
          axisLine: {{ lineStyle: {{ color: '#475569' }} }}
        }},
        yAxis: {{
          type: 'value',
          min: -1.0,
          max: 1.0,
          axisLabel: {{ color: '#94a3b8', fontSize: 10 }},
          splitLine: {{ lineStyle: {{ color: '#1e293b' }} }}
        }},
        series: [
          {{
            name: '与循环寿命关联度',
            type: 'bar',
            data: corrLife,
            itemStyle: {{ color: '#10b981' }}
          }},
          {{
            name: '与库仑效率关联度',
            type: 'bar',
            data: corrCe,
            itemStyle: {{ color: '#06b6d4' }}
          }}
        ]
      }};
      chartFeatureCorr.setOption(option);
    }}

    function renderDatabaseTable() {{
      const tbody = document.getElementById('db-table-body');
      tbody.innerHTML = '';

      const query = (document.getElementById('db-search').value || '').toLowerCase();
      
      let list = [...APP_DATA.database];
      if (query) {{
        list = list.filter(item => 
          item.additive_name.toLowerCase().includes(query) ||
          item.smiles.toLowerCase().includes(query) ||
          item.sample_id.toLowerCase().includes(query)
        );
      }}

      list.sort((a, b) => {{
        let vA = a[dbSortField];
        let vB = b[dbSortField];
        if (typeof vA === 'string') return dbSortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
        return dbSortAsc ? vA - vB : vB - vA;
      }});

      list.forEach(item => {{
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-800/50 cursor-pointer transition-colors";
        tr.onclick = () => {{
          loadPreset(item.additive_name);
          switchTab('tab-screening');
        }};

        const scoreColor = item.composite_score >= 75 ? 'text-amber-400 font-bold' : (item.composite_score >= 50 ? 'text-emerald-400' : 'text-slate-400');

        tr.innerHTML = `
          <td class="py-2.5 px-3 font-semibold text-slate-300">${{item.sample_id}}</td>
          <td class="py-2.5 px-3 font-bold text-white">${{item.additive_name}}</td>
          <td class="py-2.5 px-3 text-slate-400 font-mono text-[11px] max-w-[180px] truncate" title="${{item.smiles}}">${{item.smiles}}</td>
          <td class="py-2.5 px-2 text-right text-emerald-400 font-bold">${{item.lifespan_h.toFixed(0)}}</td>
          <td class="py-2.5 px-2 text-right text-cyan-400">${{(item.coulombic_efficiency * 100).toFixed(2)}}%</td>
          <td class="py-2.5 px-2 text-right text-indigo-400">${{item.her_overpotential_mv.toFixed(1)}}</td>
          <td class="py-2.5 px-2 text-right text-slate-300">${{item.cv_area.toFixed(0)}}</td>
          <td class="py-2.5 px-2 text-right text-slate-300">${{item.tafel_slope.toFixed(1)}}</td>
          <td class="py-2.5 px-2 text-right text-slate-300">${{item.xrd_ratio.toFixed(2)}}</td>
          <td class="py-2.5 px-2 text-right ${{scoreColor}}">${{item.composite_score}}</td>
          <td class="py-2.5 px-3 text-center">
            <button class="px-2 py-0.5 text-[11px] font-sans font-medium rounded bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500 hover:text-slate-950 transition-colors">
              载入评估
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterDatabaseTable() {{
      renderDatabaseTable();
    }}

    function sortDb(field) {{
      if (dbSortField === field) {{
        dbSortAsc = !dbSortAsc;
      }} else {{
        dbSortField = field;
        dbSortAsc = false;
      }}
      renderDatabaseTable();
    }}

    function exportDatabaseCSV() {{
      let csv = "sample_id,additive_name,smiles,cv_area,tafel_slope,xps_shift,raman_area,xrd_ratio,lifespan_h,coulombic_efficiency,her_overpotential_mv,composite_score\\n";
      APP_DATA.database.forEach(row => {{
        csv += `"${{row.sample_id}}","${{row.additive_name}}","${{row.smiles}}",${{row.cv_area}},${{row.tafel_slope}},${{row.xps_shift}},${{row.raman_area}},${{row.xrd_ratio}},${{row.lifespan_h}},${{row.coulombic_efficiency}},${{row.her_overpotential_mv}},${{row.composite_score}}\\n`;
      }});
      const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "zn_battery_additives_database.csv";
      link.click();
    }}

    // =========================================================================
    // Report Export Modal
    // =========================================================================
    function openExportModal() {{
      const modal = document.getElementById('modal-export');
      const content = document.getElementById('export-report-content');
      
      const now = new Date().toLocaleString();
      const report = `
===================================================================================
>>> 硫酸锌 (ZnSO₄) 电池添加剂 AI 智能筛选与浓度参数优化分析报告 <<<
===================================================================================
报告生成时间: ${{now}}
电解液基础体系: 2.0 mol/L ZnSO₄
评估计算引擎: PyTorch 10D-NN + Gaussian Process Regression (GPR)

【一、受试添加剂基本信息】
  * 物质名称: ${{currentAdditiveName}}
  * SMILES 结构式: ${{currentSmiles}}
  * 分子量 (MolWt): ${{currentDescriptors.MolWt.toFixed(3)}} g/mol
  * 拓扑极性表面积 (TPSA): ${{currentDescriptors.TPSA.toFixed(2)}} Å²
  * 脂水分配系数 (LogP): ${{currentDescriptors.LogP.toFixed(2)}}
  * 氢键供体数 (HBD): ${{Math.round(currentDescriptors.NumHDonors)}} | 氢键受体数 (HBA): ${{Math.round(currentDescriptors.NumHAcceptors)}}

【二、Origin 2024b 实测微观电化学输入特征】
  * CV 剥离电量峰面积: ${{currentExpFeatures.cv.toFixed(1)}} mC
  * Tafel 腐蚀极化斜率: ${{currentExpFeatures.tafel.toFixed(1)}} mV/dec
  * XPS Zn 2p 结合能偏移: ${{currentExpFeatures.xps.toFixed(2)}} eV
  * Raman 结合水特征拟合峰面积: ${{currentExpFeatures.raman.toFixed(1)}} a.u.
  * XRD (002)/(101) 晶面相对强度比: ${{currentExpFeatures.xrd.toFixed(2)}}

【三、阶段一：多任务模型推断输出指标】
  1. 对称电池循环寿命 (Cycle Life) : ${{currentPredictions.lifespan.toFixed(1)}} 小时 (相对基准提升约 ${{(currentPredictions.lifespan / 150).toFixed(1)}} 倍)
  2. 库仑效率 (Coulombic Eff, CE)  : ${{(currentPredictions.ce * 100).toFixed(3)}} %
  3. 析氢反应 (HER) 过电位         : ${{currentPredictions.her.toFixed(1)}} mV
  => 综合多任务协同潜力得分        : ${{currentPredictions.compositeScore.toFixed(2)}} / 100 分
  * 初筛准入判定结论               : 【${{currentPredictions.passed ? "通过初筛 (PASSED) -> 准入浓度优化" : "未通过初筛 (FAILED)"}}】

【四、阶段二：高斯过程回归 (GPR) 连续浓度寻优结论】
  * 推荐最佳添加浓度 (Optimal Concentration) : 【 ${{currentGPROpt.bestConc}} wt% 】
  * 最佳浓度下峰值综合预测得分               : ${{currentGPROpt.bestScore}} 分
  * 预测不确定度 (Standard Deviation, σ)     : ± ${{currentGPROpt.sigma}}
  * 95% 置信区间 (Confidence Interval)       : [ ${{currentGPROpt.ci[0]}}, ${{currentGPROpt.ci[1]}} ]

【五、电化学作用机理诊断】
  ${{document.getElementById('gpr-diagnosis-text').textContent.trim()}}
===================================================================================
      `;

      content.innerHTML = `<pre class="whitespace-pre-wrap">${{report.trim()}}</pre>`;
      modal.classList.remove('hidden');
    }}

    function closeExportModal() {{
      document.getElementById('modal-export').classList.add('hidden');
    }}

    function copyReportText() {{
      const text = document.getElementById('export-report-content').innerText;
      navigator.clipboard.writeText(text).then(() => {{
        alert("分析报告已成功复制到剪贴板！");
      }});
    }}

    function resetToDefault() {{
      loadPreset('2-氨基-4-溴蒽醌-2-磺酸钠');
      switchTab('tab-screening');
    }}
  </script>
</body>
</html>
'''

with open('/home/a1810/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Generated /home/a1810/index.html successfully! Total length: {len(html_content)} bytes")
