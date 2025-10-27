<!-- BADGES_START -->

[🇬🇧 English](README.md) ｜ [⭐ 收藏](https://github.com/jia-ning-tech/lymphoma-ewarn/stargazers) ｜ [🐛 问题](https://github.com/jia-ning-tech/lymphoma-ewarn/issues)

![GitHub Repo stars](https://img.shields.io/github/stars/jia-ning-tech/lymphoma-ewarn?style=flat)
![GitHub issues](https://img.shields.io/github/issues/jia-ning-tech/lymphoma-ewarn?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Release](https://img.shields.io/github/v/release/jia-ning-tech/lymphoma-ewarn?display_name=tag&sort=semver)

<!-- BADGES_END -->

<!-- TOC_START -->

## 目录

- [目录](#目录)
- [摘要 / Highlights](#摘要-highlights)
- [1. 引言（Introduction）](#1-引言introduction)
- [2. 方法（Methods）](#2-方法methods)
  - [2.1 队列与窗口（Cohort & Windows）](#21-队列与窗口cohort-windows)
  - [2.2 模型与训练（Models & Training）](#22-模型与训练models-training)
  - [2.3 评价指标（Metrics）](#23-评价指标metrics)
- [3. 结果（Results）](#3-结果results)
  - [3.1 判别性能（窗口级，Test）](#31-判别性能窗口级test)
  - [3.2 校准（Calibration）](#32-校准calibration)
  - [Decision Thresholds (auto-generated)](#decision-thresholds-auto-generated)
  - [3.3 领先时间（Lead Time, Test）](#33-领先时间lead-time-test)
  - [3.4 可解释性（SHAP）](#34-可解释性shap)
  - [3.5 错误分析（Error Analysis，test示例）](#35-错误分析error-analysistest示例)
  - [3.6 消融实验（Ablation，24h, 5折, 600棵树）](#36-消融实验ablation24h-5折-600棵树)
  - [Keep-only（仅保留某一组）](#keep-only仅保留某一组)
  - [Drop-one（从全集移除某一组）](#drop-one从全集移除某一组)
- [4. 仓库结构（Repository Layout & Project Structure）](#4-仓库结构repository-layout-project-structure)
- [5. 复现实验（Reproducibility）](#5-复现实验reproducibility)
  - [5.1 环境准备](#51-环境准备)
  - [5.2 从曲线到解释的完整流程](#52-从曲线到解释的完整流程)
- [6. 目录与功能导引（Repository Guide）](#6-目录与功能导引repository-guide)
- [7. 路线图（Roadmap）](#7-路线图roadmap)
- [8. 致谢与声明（Acknowledgements & Disclaimer）](#8-致谢与声明acknowledgements-disclaimer)
- [消融实验](#消融实验)
  - [消融研究（h=24）](#消融研究h24)
- [决策曲线分析（DCA）](#决策曲线分析dca)
  - [决策曲线分析（DCA）— 预测窗 24 小时，数据集：val](#决策曲线分析dca-预测窗-24-小时数据集val)
  - [决策曲线分析（DCA）— 预测窗 24 小时，数据集：test](#决策曲线分析dca-预测窗-24-小时数据集test)
  - [决策曲线分析（DCA）— 预测窗 48 小时，数据集：val](#决策曲线分析dca-预测窗-48-小时数据集val)
  - [决策曲线分析（DCA）— 预测窗 48 小时，数据集：test](#决策曲线分析dca-预测窗-48-小时数据集test)

<!-- TOC_END -->
<!-- Badges / Shields -->

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-ff9900?logo=scikitlearn&logoColor=white" />
  <img alt="SHAP" src="https://img.shields.io/badge/%E5%8F%AF%E8%A7%A3%E9%87%8A-SHAP-8A2BE2" />
  <img alt="Calibration" src="https://img.shields.io/badge/%E6%A0%A1%E5%87%86-Isotonic%20%7C%20Sigmoid-2aa198" />
  <img alt="CI" src="https://img.shields.io/badge/%E7%8A%B6%E6%80%81-%E7%A0%94%E7%A9%B6%E5%AE%9E%E9%AA%8C%E5%BC%8F-lightgrey" />
  <img alt="PRs" src="https://img.shields.io/badge/PRs-%E6%AC%A2%E8%BF%8E-brightgreen" />
  <img alt="Love" src="https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red" />
</p>

<h1 align="center">Lymphoma-EWARN：ICU淋巴瘤患者的早期预警系统</h1>

<p align="center">
  <em>面向 24h/48h 预警窗的风险预测：支持住院级聚合、领先时间评估、后处理校准、SHAP 可解释性、错误分析与消融实验。</em><br/>
  <a href="README.md">English Version</a>
</p>

---

## 摘要 / Highlights

* **两种预测视野（horizon）**：24h 与 48h
* **判别性能（test）**：AUROC ≈ **0.79**（24h）/ **0.81**（48h）；AP ≈ **0.14** / **0.29**
* **领先时间（test）**：中位数 ≈ **39.83h（24h模型）** / **34.35h（48h模型）**
* **后处理校准**：支持 Isotonic / Sigmoid，跟踪窗口级与住院级 ECE
* **可解释性**：SHAP 全局与个体解释
* **错误分析**：FP/FN 住院清单与 Top-K 窗口
* **消融实验**：vitals / labs / vent / others 分组贡献

---

## 1. 引言（Introduction）

在 ICU 淋巴瘤患者中，临床状态恶化的识别往往面临多源异构、噪声与稀疏性的挑战。**Lymphoma-EWARN** 通过窗口化特征工程与基于树的集成模型，为未来 24/48 小时提供风险评分；并在**窗口级**与**住院级**进行系统评估，量化**领先时间**，进行**后处理校准**以提升概率可信度，借助 **SHAP** 解释关键特征贡献，结合**错误分析**和**消融实验**辅助临床理解与部署前评估。实验显示，本系统在判别性能与领先时间方面具有潜在临床价值，可用于后续的影子部署与持续监测。

---

## 2. 方法（Methods）

### 2.1 队列与窗口（Cohort & Windows）

* 窗口总数：**60,593**（位于 `data_interim/trainset_hXX.parquet`）
* ICU 住院数：**721**
* 窗口阳性率（test）：**2.72%（24h）** / **3.87%（48h）**
* 住院阳性率：**15.53%**
* 每住院窗口数中位数：**52**（p10–p90：19–163）
* 每住院时间跨度（中位数）：**51h**

<details>
<summary>复现实验统计的命令</summary>

```bash
python -m src.cli.cohort_stats --horizon 24
python -m src.cli.cohort_stats --horizon 48
# 产出：
# outputs/reports/cohort_stats_h24_all.json
# outputs/reports/cohort_stats_h48_all.json
# outputs/reports/cohort_missingness_hXX_all.csv
# outputs/reports/cohort_numeric_summary_hXX_all.csv
```

</details>

### 2.2 模型与训练（Models & Training）

* **分类器**：`RandomForestClassifier`（如 `n_estimators=600`, `class_weight="balanced_subsample"`）
* **预处理**：`SimpleImputer`（mean）封装在 sklearn `Pipeline`
* **后处理校准**：Isotonic 或 Sigmoid（Platt scaling）
* **评估拆分**：`val`、`test`（窗口级）；住院级通过**同住院窗口概率最大值**进行聚合

### 2.3 评价指标（Metrics）

* **窗口级**：AUROC、AP、Brier、ECE
* **住院级**：在指定阈值（固定阈值或目标报警率）下统计 Precision / Recall / F1
* **领先时间**：每个住院从**第一条报警**到**首次事件**之间的小时数

---

## 3. 结果（Results）

### 3.1 判别性能（窗口级，Test）

| Horizon |      AUROC |         AP | 曲线                                                                                |
| ------- | ---------: | ---------: | --------------------------------------------------------------------------------- |
| 24h     | **0.7903** | **0.1372** | ROC → `outputs/figures/roc_h24_test.png` · PR → `outputs/figures/pr_h24_test.png` |
| 48h     | **0.8121** | **0.2881** | ROC → `outputs/figures/roc_h48_test.png` · PR → `outputs/figures/pr_h48_test.png` |

```bash
python -m src.cli.plot_curves --horizon 24 --split test
python -m src.cli.plot_curves --horizon 48 --split test
```

<div align="center">
  <img src="outputs/figures/roc_h24_test.png" alt="ROC 24h" width="45%"/>
  <img src="outputs/figures/pr_h24_test.png" alt="PR 24h" width="45%"/><br/>
  <img src="outputs/figures/roc_h48_test.png" alt="ROC 48h" width="45%"/>
  <img src="outputs/figures/pr_h48_test.png" alt="PR 48h" width="45%"/>
</div>

### 3.2 校准（Calibration）

**窗口级（test）：**

| Horizon |      Brier |        ECE | 图像                                                                                                        |
| ------- | ---------: | ---------: | --------------------------------------------------------------------------------------------------------- |
| 24h     | **0.0497** | **0.0284** | `outputs/figures/calibration_h24_test_window.png`, `outputs/figures/calibration_hist_h24_test_window.png` |
| 48h     | **0.0747** | **0.0478** | `outputs/figures/calibration_h48_test_window.png`, `outputs/figures/calibration_hist_h48_test_window.png` |

**住院级（test）：**

| Horizon |      Brier |        ECE | 说明                   |
| ------- | ---------: | ---------: | -------------------- |
| 24h     | **0.1768** | **0.1506** | 住院概率 = 该住院所有窗口概率的最大值 |
| 48h     | **0.1595** | **0.1350** | 同上                   |

```bash
# 窗口级
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform
# 住院级
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform --stay_level
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform --stay_level
```

<!-- THRESHOLDS_START -->

### Decision Thresholds (auto-generated)

| Horizon | Method   | Chosen Threshold | Source JSON |
|:-------:|:--------:|:----------------:|:------------|
| 24h | isotonic | **0.0577** | `outputs/reports/posthoc_calibration_h24_isotonic.json` |
| 24h | sigmoid | **0.0369** | `outputs/reports/posthoc_calibration_h24_sigmoid.json` |
| 48h | isotonic | **0.0544** | `outputs/reports/posthoc_calibration_h48_isotonic.json` |
| 48h | sigmoid | **0.0355** | `outputs/reports/posthoc_calibration_h48_sigmoid.json` |

<!-- THRESHOLDS_END -->




### 3.3 领先时间（Lead Time, Test）

| Horizon |  n | 均值(h) | 中位数(h) |  P10 |  P25 |   P75 |   P90 |   Max |
| ------- | -: | ----: | -----: | ---: | ---: | ----: | ----: | ----: |
| 48h     | 13 | 30.91 |  34.35 | 0.84 | 6.98 | 52.25 | 57.50 | 72.25 |
| 24h     |  9 | 34.81 |  39.83 | 0.51 | 2.00 | 54.50 | 61.05 | 72.25 |

```bash
python -m src.cli.leadtime_plot --horizon 48 --split test --threshold 0.24142504229084366
python -m src.cli.leadtime_plot --horizon 24 --split test --threshold 0.1205866239132141
# 图像：
# outputs/figures/leadtime_hist_hXX_test_thr*.png
# outputs/figures/leadtime_box_hXX_test_thr*.png
```

### 3.4 可解释性（SHAP）

* 全局：`outputs/figures/shap_global_beeswarm_h48_test.png`、`outputs/figures/shap_global_bar_h48_test.png`
* 明细：`outputs/reports/shap_values_top_h48_test.csv`（含特征重要度），并可导出 Top-K 个体解释

```bash
python -m src.cli.shap_explain --horizon 48 --split test --top_n 200 --top_k_individual 5
python -m src.cli.shap_explain --horizon 24 --split test --top_n 200 --top_k_individual 5
```

<!-- FI_START -->
## 特征重要性

> 结果为置换重要度（多次重复取均值的 ΔAUC）。当前 SHAP 因模型封装未运行，后续会修复补充。

### 预测提前窗 h=24 — 验证集

![特征重要性 — h24 val](outputs/figures/fi_forest_h24_val.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.07928 | 0.00493 |
| `resp_rate__24h__count` | 0.00000 | 0.02508 | 0.00587 |
| `dbp__24h__count` | 0.00000 | 0.02164 | 0.00495 |
| `sbp__24h__count` | 0.00000 | 0.02109 | 0.00668 |
| `heart_rate__24h__count` | 0.00000 | 0.01949 | 0.00670 |
| `peep__6h__count` | 0.00000 | 0.01846 | 0.00923 |
| `hco3__24h__max` | 0.00000 | 0.01639 | 0.00088 |
| `hco3__24h__min` | 0.00000 | 0.01475 | 0.00270 |
| `spo2__24h__count` | 0.00000 | 0.01462 | 0.00264 |
| `hco3__24h__mean` | 0.00000 | 0.01456 | 0.00127 |
| `mbp__24h__count` | 0.00000 | 0.01282 | 0.00410 |
| `na__24h__mean` | 0.00000 | 0.01192 | 0.00538 |
| `na__24h__min` | 0.00000 | 0.01178 | 0.00284 |
| `temperature__24h__count` | 0.00000 | 0.01168 | 0.00628 |
| `cl__24h__mean` | 0.00000 | 0.01126 | 0.00388 |
| `cl__24h__max` | 0.00000 | 0.01015 | 0.00245 |
| `hco3__24h__std` | 0.00000 | 0.00990 | 0.00099 |
| `hco3__24h__last` | 0.00000 | 0.00983 | 0.00148 |
| `dbp__24h__max` | 0.00000 | 0.00959 | 0.00215 |
| `spo2__24h__mean` | 0.00000 | 0.00891 | 0.00449 |

### 预测提前窗 h=24 — 测试集

![特征重要性 — h24 test](outputs/figures/fi_forest_h24_test.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.04246 | 0.00154 |
| `cl__24h__mean` | 0.00000 | 0.00261 | 0.00034 |
| `peep__24h__std` | 0.00000 | 0.00246 | 0.00048 |
| `dbp__24h__mean` | 0.00000 | 0.00241 | 0.00116 |
| `cl__24h__min` | 0.00000 | 0.00207 | 0.00097 |
| `temperature__24h__mean` | 0.00000 | 0.00191 | 0.00035 |
| `heart_rate__6h__mean` | 0.00000 | 0.00173 | 0.00036 |
| `peep__24h__mean` | 0.00000 | 0.00168 | 0.00059 |
| `pt__24h__min` | 0.00000 | 0.00165 | 0.00040 |
| `lactate__24h__last` | 0.00000 | 0.00160 | 0.00070 |
| `spo2__24h__std` | 0.00000 | 0.00153 | 0.00046 |
| `na__6h__max` | 0.00000 | 0.00131 | 0.00012 |
| `peep__24h__max` | 0.00000 | 0.00130 | 0.00077 |
| `na__6h__mean` | 0.00000 | 0.00129 | 0.00017 |
| `heart_rate__24h__mean` | 0.00000 | 0.00120 | 0.00052 |
| `temperature__24h__max` | 0.00000 | 0.00113 | 0.00073 |
| `resp_rate__24h__std` | 0.00000 | 0.00112 | 0.00019 |
| `sbp__6h__min` | 0.00000 | 0.00111 | 0.00021 |
| `platelet__24h__mean` | 0.00000 | 0.00106 | 0.00090 |
| `pao2__24h__min` | 0.00000 | 0.00106 | 0.00018 |

### 预测提前窗 h=48 — 验证集

![特征重要性 — h48 val](outputs/figures/fi_forest_h48_val.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.08343 | 0.00585 |
| `heart_rate__24h__count` | 0.00000 | 0.03088 | 0.00677 |
| `resp_rate__24h__count` | 0.00000 | 0.02665 | 0.00518 |
| `peep__6h__count` | 0.00000 | 0.02411 | 0.00649 |
| `sbp__24h__count` | 0.00000 | 0.02174 | 0.00532 |
| `ph__24h__min` | 0.00000 | 0.01863 | 0.00264 |
| `na__24h__mean` | 0.00000 | 0.01810 | 0.00643 |
| `spo2__24h__count` | 0.00000 | 0.01789 | 0.00408 |
| `dbp__24h__count` | 0.00000 | 0.01697 | 0.00680 |
| `mbp__24h__count` | 0.00000 | 0.01483 | 0.00472 |
| `ca__24h__max` | 0.00000 | 0.01478 | 0.00277 |
| `na__24h__min` | 0.00000 | 0.01420 | 0.00390 |
| `temperature__24h__count` | 0.00000 | 0.01020 | 0.00447 |
| `peep__24h__std` | 0.00000 | 0.00858 | 0.00394 |
| `hco3__24h__min` | 0.00000 | 0.00826 | 0.00059 |
| `spo2__24h__mean` | 0.00000 | 0.00803 | 0.00145 |
| `cl__24h__mean` | 0.00000 | 0.00791 | 0.00300 |
| `peep__24h__mean` | 0.00000 | 0.00710 | 0.00493 |
| `peep__24h__max` | 0.00000 | 0.00670 | 0.00362 |
| `cl__24h__min` | 0.00000 | 0.00649 | 0.00320 |

### 预测提前窗 h=48 — 测试集

![特征重要性 — h48 test](outputs/figures/fi_forest_h48_test.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.10095 | 0.00459 |
| `resp_rate__24h__max` | 0.00000 | 0.00641 | 0.00119 |
| `hgb__24h__min` | 0.00000 | 0.00563 | 0.00111 |
| `temperature__24h__count` | 0.00000 | 0.00520 | 0.00181 |
| `spo2__24h__mean` | 0.00000 | 0.00505 | 0.00183 |
| `platelet__24h__min` | 0.00000 | 0.00505 | 0.00126 |
| `bun__24h__max` | 0.00000 | 0.00483 | 0.00093 |
| `platelet__24h__max` | 0.00000 | 0.00421 | 0.00072 |
| `temperature__24h__mean` | 0.00000 | 0.00368 | 0.00065 |
| `peep__24h__std` | 0.00000 | 0.00358 | 0.00224 |
| `resp_rate__24h__mean` | 0.00000 | 0.00342 | 0.00223 |
| `ca__24h__mean` | 0.00000 | 0.00340 | 0.00242 |
| `lactate__24h__mean` | 0.00000 | 0.00302 | 0.00043 |
| `lactate__24h__last` | 0.00000 | 0.00299 | 0.00076 |
| `platelet__24h__mean` | 0.00000 | 0.00298 | 0.00103 |
| `creatinine__24h__min` | 0.00000 | 0.00294 | 0.00028 |
| `temperature__24h__min` | 0.00000 | 0.00283 | 0.00063 |
| `spo2__24h__count` | 0.00000 | 0.00272 | 0.00141 |
| `heart_rate__24h__count` | 0.00000 | 0.00269 | 0.00281 |
| `platelet__24h__last` | 0.00000 | 0.00243 | 0.00136 |
<!-- FI_END -->


### 3.5 错误分析（Error Analysis，test示例）

* 以 10% 报警率选阈值（48h，raw）：阈值 ≈ **0.0675**
* 住院级 @ thr=0.0675：**Precision=0.379**、**Recall=0.223**、**F1=0.281**
* 导出表格：

  * `outputs/reports/errors_fp_stay_h48_test_thr0.0675.parquet`
  * `outputs/reports/errors_fn_stay_h48_test_thr0.0675.parquet`
  * `outputs/reports/errors_fp_windows_top3_h48_test_thr0.0675.parquet`
  * `outputs/reports/errors_fn_windows_top3_h48_test_thr0.0675.parquet`

```bash
python -m src.cli.error_analysis --horizon 48 --split test --alert_rate 0.10
# 或固定阈值
python -m src.cli.error_analysis --horizon 48 --split test --threshold 0.3346
# 或使用校准概率（先拟合后处理校准，再分析）
python -m src.cli.posthoc_calibrate --horizon 48 --method isotonic --refit_threshold_rate 0.10
python -m src.cli.error_analysis --horizon 48 --split test --alert_rate 0.10 --calibrated isotonic
```

### 3.6 消融实验（Ablation，24h, 5折, 600棵树）

* Baseline_all：AUROC ≈ **0.797 ± 0.023**；AP ≈ **0.130 ± 0.040**
* 去除 vitals：AUROC ≈ **0.771 ± 0.028**；AP ≈ **0.070 ± 0.020**
* 其余分组（labs / vent / others）正在/可继续评估

```bash
python -m src.cli.ablation_study --horizon 24 --folds 5 --n_estimators 600 --mode both
```


我们在 **keep-only** 与 **drop-one** 两种设定下评估各特征组的贡献（5 折、600 棵树、24h 任务）。完整结果表见：`outputs/reports/ablation_h24.csv`。

### Keep-only（仅保留某一组）
| 组别      | AUROC（均值±标准差） | AP（均值±标准差） |
|-----------|---------------------:|------------------:|
| **Vitals（n=98）** | **0.6604 ± 0.0494** | **0.0602 ± 0.0197** |
| **Labs（n=112）**  | **0.6662 ± 0.0444** | **0.0488 ± 0.0091** |
| **Vent（n=14）**   | **0.7719 ± 0.0236** | **0.0686 ± 0.0156** |
| **Others（n=148）**| **0.4985 ± 0.0503** | **0.0296 ± 0.0076** |

**观察。** 仅使用通气相关（Vent）单组就能取得相对更强的判别力（AUROC ~0.77）；Vitals/Labs 单独使用时表现中等；Others 单独使用较弱。

### Drop-one（从全集移除某一组）
基线（全部特征）：**AUROC 0.7974 ± 0.0228**，**AP 0.1303 ± 0.0398**（n_features=372）

| 移除组别 | 保留特征数 | AUROC（均值±标准差） | AP（均值±标准差） |
|----------|-----------:|---------------------:|------------------:|
| **Vitals** | 274 | **0.7711 ± 0.0276** | **0.0696 ± 0.0197** |
| **Labs**   | 260 | **0.7646 ± 0.0274** | **0.1260 ± 0.0447** |
| **Others** | 224 | **0.8021 ± 0.0262** | **0.1249 ± 0.0360** |

**解读。** **Vitals** 与 **Labs** 在与其他组联合时**贡献明显**，移除它们会使 AUROC 下滑更明显；移除 **Others** 基本不伤大局，AUROC 甚至可能在波动范围内略有上浮，提示该组在本队列/设定下的**边际效用较有限**。

> 复现命令：
> ```bash
> python -m src.cli.ablation_study --horizon 24 --folds 5 --n_estimators 600 --mode both
> # 汇总表 → outputs/reports/ablation_h24.csv
> ```



---

## 4. 仓库结构（Repository Layout & Project Structure）

> 下方部分由 **Makefile 自动生成**，可通过 `make structure && make inject-structure` 刷新。
> 该块位于锚点之间，请勿手动编辑。

<!-- PROJECT_STRUCTURE:START -->

*(此区块由 `make inject-structure` 自动注入，勿手改。)*

<!-- PROJECT_STRUCTURE:END -->

---

## 5. 复现实验（Reproducibility）

### 5.1 环境准备

```bash
# 示例环境
conda create -n ewarn python=3.10 -y
conda activate ewarn
pip install -r requirements.txt
```

### 5.2 从曲线到解释的完整流程

```bash
# ROC/PR 曲线
python -m src.cli.plot_curves --horizon 24 --split test
python -m src.cli.plot_curves --horizon 48 --split test

# 校准（窗口级 & 住院级）
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform --stay_level
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform --stay_level

# 领先时间
python -m src.cli.leadtime_plot --horizon 48 --split test --threshold 0.24142504229084366
python -m src.cli.leadtime_plot --horizon 24 --split test --threshold 0.1205866239132141

# SHAP 解释
python -m src.cli.shap_explain --horizon 48 --split test --top_n 200 --top_k_individual 5
python -m src.cli.shap_explain --horizon 24 --split test --top_n 200 --top_k_individual 5

# 后处理校准 & 错误分析
python -m src.cli.posthoc_calibrate --horizon 24 --method isotonic --refit_threshold_rate 0.10
python -m src.cli.error_analysis --horizon 24 --split test --alert_rate 0.10 --calibrated isotonic

# 队列统计
python -m src.cli.cohort_stats --horizon 24
python -m src.cli.cohort_stats --horizon 48
```

---

## 6. 目录与功能导引（Repository Guide）

* `src/` —— 源代码
* `src/cli/` —— 训练、评估、作图、发布打包等命令行工具
* `data_raw/` —— 原始数据（不纳入版本控制）
* `data_interim/` —— 中间特征与工程化产物
* `outputs/` —— 自动产出结果

  * `outputs/models/` —— 训练好的模型（`.joblib`）
  * `outputs/preds/` —— 预测/验证结果（`.parquet`）
  * `outputs/reports/` —— 供论文与 README 使用的指标/表格
  * `outputs/figures/` —— ROC/PR、校准、SHAP、领先时间等图
  * `outputs/release/` —— 部署/监测用的打包产出
* `notebooks/` —— 探索性分析（可选）
* `scripts/` —— 辅助脚本（可选）

> 建议使用 `make structure && make inject-structure` 定期刷新 README 中的**结构**区块，确保与实际目录一致。

---

## 7. 路线图（Roadmap）

* ✅ 已完成：曲线、校准、领先时间、SHAP、错误表、带进度条的消融
* ⏳ 进行中：影子部署与标准化监测
* ⏳ 论文图表自动同步、更多外部验证与公平性分析

---

## 8. 致谢与声明（Acknowledgements & Disclaimer）

感谢临床合作者与开源社区（scikit-learn、SHAP 等）的支持。本项目目前仅用于**科研探索**，任何临床部署都需严格的外部验证、治理与伦理审查。

> 我们深知仍有诸多不足与改进空间。如有问题或建议，欢迎提交 Issue/PR 或与我们交流。非常感谢你的耐心与指正。🙏

---

**语言切换**： [English](README.md)｜[中文](README.zh-CN.md)

---

<!-- ABLATION-ZH:START -->
## 消融实验

本节统一汇报 **keep-only** 与 **drop-one** 两类消融，所有点均为 5 折交叉验证的**均值±标准差**。

### 消融研究（h=24）

| 设置 | 分组 | 特征数 | AUROC(均值±std) | AP(均值±std) |
|---|---:|---:|---:|---:|
| baseline_all | - | 372 | 0.7974 ± 0.0228 | 0.1303 ± 0.0398 |
| drop-one | labs | 260 | 0.7646 ± 0.0274 | 0.1260 ± 0.0447 |
| drop-one | others | 224 | 0.8021 ± 0.0262 | 0.1249 ± 0.0360 |
| drop-one | vent | 358 | 0.7225 ± 0.0465 | 0.0773 ± 0.0148 |
| drop-one | vitals | 274 | 0.7711 ± 0.0276 | 0.0696 ± 0.0197 |
| keep-only | labs | 112 | 0.6662 ± 0.0444 | 0.0488 ± 0.0091 |
| keep-only | others | 148 | 0.4985 ± 0.0503 | 0.0296 ± 0.0076 |
| keep-only | vent | 14 | 0.7719 ± 0.0236 | 0.0686 ± 0.0156 |
| keep-only | vitals | 98 | 0.6604 ± 0.0494 | 0.0602 ± 0.0197 |

<!-- ABLATION-ZH:END -->


<!-- DCA_ZH_START -->
## 决策曲线分析（DCA）

### 决策曲线分析（DCA）— 预测窗 24 小时，数据集：val

- 下表展示约 0.05 / 0.10 / 0.20 三个典型阈值（自动就近取 CSV 网格）。
- `净获益` 为每例患者（若在绘图时用 `--per-100`，则为每百例患者）。

| 变体 | 阈值 | 净获益（模型） | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 0.1803 | -3.4450 | 0.0000 |
| raw | 0.100 | 0.1909 | -9.1920 | 0.0000 |
| raw | 0.200 | 0.0283 | -22.8410 | 0.0000 |
| isotonic | 0.050 | 0.3388 | -3.4450 | 0.0000 |
| isotonic | 0.100 | 0.2244 | -9.1920 | 0.0000 |
| isotonic | 0.200 | 0.1628 | -22.8410 | 0.0000 |
| sigmoid | 0.050 | 0.1724 | -3.4450 | 0.0000 |
| sigmoid | 0.100 | 0.1279 | -9.1920 | 0.0000 |
| sigmoid | 0.200 | 0.1345 | -22.8410 | 0.0000 |

**曲线**
![h=24, val, raw](outputs/figures/dca_h24_val.png)
![h=24, val, isotonic](outputs/figures/dca_h24_val_cal_isotonic.png)
![h=24, val, sigmoid](outputs/figures/dca_h24_val_cal_sigmoid.png)

### 决策曲线分析（DCA）— 预测窗 24 小时，数据集：test

- 下表展示约 0.05 / 0.10 / 0.20 三个典型阈值（自动就近取 CSV 网格）。
- `净获益` 为每例患者（若在绘图时用 `--per-100`，则为每百例患者）。

| 变体 | 阈值 | 净获益（模型） | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 1.2371 | 0.3376 | 0.0000 |
| raw | 0.100 | 0.2109 | -5.1992 | 0.0000 |
| raw | 0.200 | -0.0540 | -18.3491 | 0.0000 |
| isotonic | 0.050 | 2.1992 | 0.3376 | 0.0000 |
| isotonic | 0.100 | -0.1134 | -5.1992 | 0.0000 |
| isotonic | 0.200 | -0.3753 | -18.3491 | 0.0000 |
| sigmoid | 0.050 | 1.3452 | 0.3376 | 0.0000 |
| sigmoid | 0.100 | 0.2936 | -5.1992 | 0.0000 |
| sigmoid | 0.200 | -0.3543 | -18.3491 | 0.0000 |

**曲线**
![h=24, test, raw](outputs/figures/dca_h24_test.png)
![h=24, test, isotonic](outputs/figures/dca_h24_test_cal_isotonic.png)
![h=24, test, sigmoid](outputs/figures/dca_h24_test_cal_sigmoid.png)

### 决策曲线分析（DCA）— 预测窗 48 小时，数据集：val

- 下表展示约 0.05 / 0.10 / 0.20 三个典型阈值（自动就近取 CSV 网格）。
- `净获益` 为每例患者（若在绘图时用 `--per-100`，则为每百例患者）。

| 变体 | 阈值 | 净获益（模型） | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 0.2330 | -2.9880 | 0.0000 |
| raw | 0.100 | 0.1290 | -8.7096 | 0.0000 |
| raw | 0.200 | 0.1062 | -22.2983 | 0.0000 |
| isotonic | 0.050 | 0.3487 | -2.9880 | 0.0000 |
| isotonic | 0.100 | 0.2192 | -8.7096 | 0.0000 |
| isotonic | 0.200 | 0.1840 | -22.2983 | 0.0000 |
| sigmoid | 0.050 | 0.2295 | -2.9880 | 0.0000 |
| sigmoid | 0.100 | 0.1657 | -8.7096 | 0.0000 |
| sigmoid | 0.200 | 0.1109 | -22.2983 | 0.0000 |

**曲线**
![h=48, val, raw](outputs/figures/dca_h48_val.png)
![h=48, val, isotonic](outputs/figures/dca_h48_val_cal_isotonic.png)
![h=48, val, sigmoid](outputs/figures/dca_h48_val_cal_sigmoid.png)

### 决策曲线分析（DCA）— 预测窗 48 小时，数据集：test

- 下表展示约 0.05 / 0.10 / 0.20 三个典型阈值（自动就近取 CSV 网格）。
- `净获益` 为每例患者（若在绘图时用 `--per-100`，则为每百例患者）。

| 变体 | 阈值 | 净获益（模型） | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 3.2239 | 3.8269 | 0.0000 |
| raw | 0.100 | 1.6348 | -1.5160 | 0.0000 |
| raw | 0.200 | 0.7657 | -14.2055 | 0.0000 |
| isotonic | 0.050 | 4.3478 | 3.8269 | 0.0000 |
| isotonic | 0.100 | 1.1650 | -1.5160 | 0.0000 |
| isotonic | 0.200 | 0.5495 | -14.2055 | 0.0000 |
| sigmoid | 0.050 | 2.4969 | 3.8269 | 0.0000 |
| sigmoid | 0.100 | 1.5801 | -1.5160 | 0.0000 |
| sigmoid | 0.200 | 0.7326 | -14.2055 | 0.0000 |

**曲线**
![h=48, test, raw](outputs/figures/dca_h48_test.png)
![h=48, test, isotonic](outputs/figures/dca_h48_test_cal_isotonic.png)
![h=48, test, sigmoid](outputs/figures/dca_h48_test_cal_sigmoid.png)


<!-- DCA_ZH_END -->

<!-- DCA_START -->

### 决策曲线（窗口=24小时，数据集=test）

![DCA（原始概率）](outputs/figures/dca_h24_test.png)

![DCA（后处理校准：isotonic）](outputs/figures/dca_h24_test_cal_isotonic.png)

![DCA（后处理校准：sigmoid）](outputs/figures/dca_h24_test_cal_sigmoid.png)


### 决策曲线（窗口=48小时，数据集=test）

![DCA（原始概率）](outputs/figures/dca_h48_test.png)

![DCA（后处理校准：isotonic）](outputs/figures/dca_h48_test_cal_isotonic.png)

![DCA（后处理校准：sigmoid）](outputs/figures/dca_h48_test_cal_sigmoid.png)

<!-- DCA_END -->
