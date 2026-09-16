---
title: AI-Driven Aqueous Zinc-Ion Battery Discovery Platform
emoji: 🔋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: High-Dim ECFP4 Multimodal Stacking & Bayesian Studio for Zn Batteries
---

# 🔬 AI-Driven Aqueous Zinc-Ion Battery Discovery Platform

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-science, reproducible computational platform designed in alignment with rigorous **Nature Communications / Advanced Materials** research standards for the discovery and optimization of electrolyte additives in aqueous zinc-ion batteries (AZIBs).

---

## 🏛️ Architecture Overview

The platform integrates deep chemoinformatics, heterogeneous ensemble meta-learning, cooperative game-theoretic attribution, and active-learning experimental design into an end-to-end web workbench:

```
+-----------------------------------------------------------------------------------------+
|                                1. Multimodal Feature Engineering                         |
|  - Chemical Domain: RDKit Morgan Circular Topological Fingerprints (ECFP4, 2048-bit)    |
|  - Electrochemical Domain: 5-Dim Origin 2024b Metrics (CV, Tafel, XPS, Raman, XRD)     |
|  - Independent StandardScaler (Z-Score) Alignment -> 2053-Dim Fused Input Tensor        |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                        2. Heterogeneous Stacking Ensemble Engine                         |
|  - Branch A (PyTorch MLP): 2053 -> 256 -> 64 -> 32 -> 1, with BatchNorm1d & Dropout(0.3)|
|  - Branch B (XGBoost Regressor): TruncatedSVD (2048 -> 32 dims) + 5-dim physical metrics|
|  - Meta-Learner: 5-Fold OOF CV via scipy.optimize.nnls with L1 Simplex Projection       |
|    Constraint: sum(w_i) = 1.000, w_i >= 0 (Bounded Variance, Zero Out-of-Domain Drift)  |
+--------------------------------------------+--------------------------------------------+
                                             |
                      +----------------------+----------------------+
                      |                                             |
                      v                                             v
+-------------------------------------------+ +-------------------------------------------+
| 3. High-Dim Condensed SHAP Attribution    | | 4. Bayesian Active Learning Optimization  |
| - GradientExplainer over PyTorch Network  | | - Composite Kernel: Matern(nu=2.5) +      |
| - Feature Aggregation via Additivity:     | |   WhiteKernel (Fixed Experimental Noise)  |
|   phi(mol_Topology) = sum(phi_ECFP4_bits) | | - Acquisition: Upper Confidence Bound     |
| - Publication-Quality 6-Dim Waterfall Plot| |   UCB(x) = mu(x) + kappa * sigma(x)       |
| - Chemical vs. Electrochemical Subspaces  | | - Origin Publication Data Export (CSV)    |
+-------------------------------------------+ +-------------------------------------------+
```

### 1. High-Dimensional Topological Feature Extraction (ECFP4)
- **Topological Invariance**: Replaces conventional 5-dimensional scalar representations with 2048-bit Extended Connectivity Fingerprints (ECFP4, radius=2) calculated via RDKit, encoding circular atomic invariants, hybridization states, and conjugated ring systems.
- **Multimodal Tensor Alignment**: Fuses 2048-dimensional chemical graph vectors with 5-dimensional interfacial electrochemical parameters (CV stripping area, Tafel polarization slope, XPS Zn 2p binding energy shift, Raman solvation peak area ratio, and XRD (002)/(101) texture ratio) via independent Z-score transformations into a unified **(2048 + 5 = 2053)** dimensional multimodal tensor.

### 2. Heterogeneous Stacking Ensemble & Simplex-Projected Meta-Learner
- **PyTorch High-Dim MLP**: Features a 4-tier structural bottleneck (`2053 -> 256 -> 64 -> 32 -> 1`) equipped with inter-layer `BatchNorm1d` to counteract internal covariate shift and `Dropout(p=0.3)` with AdamW weight decay to prevent co-adaptation over ultra-sparse inputs.
- **Orthogonalized XGBoost Branch**: Mitigates tree-partitioning degradation across sparse vectors by applying `TruncatedSVD` (compressing 2048 bits into 32 orthogonal components) concatenated with the 5 physical scalars.
- **Probability Simplex Meta-Learner**: Solves optimal ensemble weights via 5-Fold Cross-Validation Out-of-Fold (OOF) predictions with `scipy.optimize.nnls` subject to strict $L_1$ Simplex Projection ($\sum w_i = 1$, $w_i \ge 0$), eliminating unbounded extrapolation variance.

### 3. Condensed Dual-Perspective SHAP Interpretability
- **Shapley Additivity Condensation**: Leverages the Efficiency Axiom of cooperative game theory to condense all 2048 Morgan bit attributions into a unified global scalar:
  $$\phi(\text{mol\_Substructure\_Topology}) = \sum_{i=0}^{2047} \phi(\text{ECFP4\_bit}_i)$$
- **Orthogonal Subspace Partitioning**: Delivers publication-ready, overlap-free waterfall plots partitioning impact into **Chemical Intrinsic Subspace** vs. **Interfacial Electrochemical Kinetics Subspace**.

### 4. Bayesian Active Learning with Physical Noise Prior
- **Gaussian Process Regression (GPR)**: Evaluates the continuous concentration manifold ($0.5 \sim 2.5\text{ wt}\%$) using a composite $\text{Mat\'ern}(\nu=2.5) + \text{WhiteKernel}$ prior that embeds intrinsic cycle test variance ($\pm 20\sim 25\text{ h}$).
- **Autonomous UCB Policy**: Directs closed-loop iteration via Upper Confidence Bound ($\text{UCB} = \mu + \kappa \sigma$), and generates publication-grade Origin CSV datasets with $95\%$ confidence envelopes.

---

## 🚢 Cloud & Container Deployment

This repository is strictly configured for **instant one-click containerized deployment** on **Hugging Face Spaces (Docker SDK)**.

- **Base Image**: `python:3.10-slim`
- **Security & Port Mapping**: Runs under a dedicated non-root user (`user`, UID `1000`) and binds to standard port `7860`.
- **System Dependencies**: Pre-installs shared C/C++ libraries (`libxrender1`, `libxext6`, `libgl1`, `libglib2.0-0`, `libgomp1`) required by RDKit, PyTorch, and headless Matplotlib.

---

## 📄 License & Open-Science Reproducibility

This project is released under the **MIT License**. All experimental benchmark records are permanently version-controlled in `zn_battery_experiment_database.csv` to ensure reproducibility across laboratories worldwide.
