# =========================
# Lymphoma-EWARN Makefile
# 生成/注入项目结构，并辅助生成 README
# =========================

SHELL := /bin/bash

# -------- 可配置变量 --------
PY ?= python
HORIZONS ?= 24,48
SPLIT ?= test
README ?= README.md

# -------- 产物路径 --------
DOCS_DIR := outputs/docs
STRUCT_MD := $(DOCS_DIR)/project_structure.md

# 伪目标
.PHONY: help structure show-structure inject-structure readme clean-structure

help:
	@echo "Usage:"
	@echo "  make structure           # 生成 outputs/docs/project_structure.md"
	@echo "  make show-structure      # 预览自动生成的项目结构的前 120 行"
	@echo "  make inject-structure    # 把项目结构注入 README 的锚点块"
	@echo "  make readme              # 生成 README（保持不变），随后注入结构"
	@echo "  make clean-structure     # 清理生成的 project_structure.md"
	@echo ""
	@echo "Env vars:"
	@echo "  PY=$(PY)"
	@echo "  HORIZONS=$(HORIZONS)"
	@echo "  SPLIT=$(SPLIT)"
	@echo "  README=$(README)"

# 目录
$(DOCS_DIR):
	mkdir -p $(DOCS_DIR)

# 生成项目结构 Markdown 文件（自动扫描目录，含摘要表与树形结构）
$(STRUCT_MD): scripts/gen_structure.py | $(DOCS_DIR)
	$(PY) scripts/gen_structure.py --out $(STRUCT_MD)

structure: $(STRUCT_MD)

show-structure: $(STRUCT_MD)
	nl -ba $(STRUCT_MD) | sed -n '1,120p'

# 将 $(STRUCT_MD) 注入 README 的锚点之间
inject-structure: $(STRUCT_MD) scripts/inject_structure.py
	$(PY) scripts/inject_structure.py \
	  --readme $(README) \
	  --block-start "<!-- PROJECT_STRUCTURE:START -->" \
	  --block-end "<!-- PROJECT_STRUCTURE:END -->" \
	  --source $(STRUCT_MD)

# 示例：如你后续有 README 自动生成脚本，可在这里调用；目前仅做注入
readme: structure inject-structure
	@echo "[Makefile] README updated with project structure."

clean-structure:
	rm -f $(STRUCT_MD)
	@echo "[Makefile] Removed $(STRUCT_MD)"


# ---------------- Ablation plotting ----------------
# 可配置参数
AB_HORIZONS ?= 24,48
REPORTS_DIR ?= outputs/reports
FIG_DIR     ?= outputs/figures

.PHONY: ablation-plots show-ablation clean-ablation

## ablation-plots: 读取 $(REPORTS_DIR)/ablation_h{h}.csv 并生成森林图/柱状图到 $(FIG_DIR)
ablation-plots:
	@echo "[Makefile] Plotting ablation for horizons=$(AB_HORIZONS)"
	@$(PY) scripts/plot_ablation.py --horizons $(AB_HORIZONS) --reports_dir $(REPORTS_DIR) --fig_dir $(FIG_DIR)

## show-ablation: 快速查看已生成的消融图片与用于 README 的整洁表
show-ablation:
	@echo "---- Figures ----"
	@ls -1 $(FIG_DIR)/ablation_* 2>/dev/null || echo "(no ablation figures yet)"
	@echo ""
	@echo "---- Tidy CSV for README ----"
	@ls -1 $(REPORTS_DIR)/ablation_*tidy_for_readme.csv 2>/dev/null || echo "(no tidy csv yet)"

## clean-ablation: 清理消融图片（不动 ablation_h{h}.csv 原始结果）
clean-ablation:
	@rm -f $(FIG_DIR)/ablation_*_bar.png $(FIG_DIR)/ablation_*_forest.png || true
	@echo "[Makefile] cleaned ablation figures under $(FIG_DIR)"


# ---------------- Inject ablation section into README ----------------
AB_REPORTS_DIR ?= outputs/reports
AB_FIG_DIR     ?= outputs/figures
AB_OUT_DIR     ?= outputs/docs

AB_EN_MD := $(AB_OUT_DIR)/ablation_section.md
AB_ZH_MD := $(AB_OUT_DIR)/ablation_section_zh.md

.PHONY: ablation-md inject-ablation readme-ablation

## ablation-md: 仅生成 Markdown 片段（不改 README）
ablation-md:
	@mkdir -p $(AB_OUT_DIR)
	@$(PY) scripts/inject_ablation.py \
		--reports_dir $(AB_REPORTS_DIR) \
		--fig_dir $(AB_FIG_DIR) \
		--out_dir $(AB_OUT_DIR) \
		--emit_only
	@echo "[Makefile] Generated: $(AB_EN_MD) and $(AB_ZH_MD)"

## inject-ablation: 生成片段 + 注入到 README 和 README.zh-CN.md
inject-ablation:
	@mkdir -p $(AB_OUT_DIR)
	@$(PY) scripts/inject_ablation.py \
		--reports_dir $(AB_REPORTS_DIR) \
		--fig_dir $(AB_FIG_DIR) \
		--out_dir $(AB_OUT_DIR) \
		--readme README.md \
		--readme_zh README.zh-CN.md
	@echo "[Makefile] Ablation section injected into README files."

## readme-ablation: 先画图（若需要）再注入
readme-ablation: ablation-plots inject-ablation
	@echo "[Makefile] README ablation updated."


# === Sync selected figures to docs/figures for GitHub rendering ===
DOCS_FIG := docs/figures
.PHONY: sync-figs
sync-figs:
	@mkdir -p $(DOCS_FIG)
	# 你想公开展示的少量关键图（请按需增减）
	@cp -f outputs/figures/roc_h24_test.png        $(DOCS_FIG)/roc_h24_test.png || true
	@cp -f outputs/figures/pr_h24_test.png         $(DOCS_FIG)/pr_h24_test.png || true
	@cp -f outputs/figures/roc_h48_test.png        $(DOCS_FIG)/roc_h48_test.png || true
	@cp -f outputs/figures/pr_h48_test.png         $(DOCS_FIG)/pr_h48_test.png || true
	@cp -f outputs/figures/calibration_h24_test_window.png $(DOCS_FIG)/calib_h24_window.png || true
	@cp -f outputs/figures/calibration_h48_test_window.png $(DOCS_FIG)/calib_h48_window.png || true
	@cp -f outputs/figures/shap_global_beeswarm_h48_test.png $(DOCS_FIG)/shap_beeswarm_h48.png || true
	@cp -f outputs/figures/shap_global_bar_h48_test.png      $(DOCS_FIG)/shap_bar_h48.png || true
	@cp -f outputs/figures/leadtime_box_h48_test_thr0.2414.png  $(DOCS_FIG)/leadtime_box_h48.png || true
	@cp -f outputs/figures/leadtime_hist_h48_test_thr0.2414.png $(DOCS_FIG)/leadtime_hist_h48.png || true
	@echo "[Makefile] Copied selected figures to docs/figures"

# 一键：同步图 + 重新注入结构 + 注入消融块
.PHONY: publish-readme
publish-readme: sync-figs structure inject-structure inject-ablation
	@echo "[Makefile] README materials refreshed."



## ---- Auto thresholds snippet for README (no heredoc) ----
.PHONY: thresholds-md inject-thresholds readme-thresholds

# 生成 Markdown 片段：outputs/docs/thresholds_for_readme.md
thresholds-md:
	@mkdir -p outputs/docs
	@$(PY) scripts/mk_threshold_snippets.py

# 注入到 README 与 README.zh-CN（锚点不存在则自动追加到文末）
inject-thresholds: thresholds-md
	@$(PY) scripts/inject_thresholds.py

# 一键：生成 + 注入
readme-thresholds: inject-thresholds
	@echo "[Makefile] Thresholds snippet refreshed and injected."





# ===== DCA (Decision Curve Analysis) =====
.PHONY: dca
dca:
	@echo "[Makefile] DCA for h=24/48 on val/test (raw prob)"
	@$(PY) -m src.cli.dca_plot --horizon 24 --split val
	@$(PY) -m src.cli.dca_plot --horizon 24 --split test
	@$(PY) -m src.cli.dca_plot --horizon 48 --split val
	@$(PY) -m src.cli.dca_plot --horizon 48 --split test
	@echo "[Makefile] Done -> outputs/reports/dca_*.csv and outputs/figures/dca_*.png"

.PHONY: dca-cal
dca-cal:
	@echo "[Makefile] DCA for calibrated prob (isotonic & sigmoid)"
	@$(PY) -m src.cli.dca_plot --horizon 24 --split test --calibrated isotonic || true
	@$(PY) -m src.cli.dca_plot --horizon 48 --split test --calibrated isotonic || true
	@$(PY) -m src.cli.dca_plot --horizon 24 --split test --calibrated sigmoid  || true
	@$(PY) -m src.cli.dca_plot --horizon 48 --split test --calibrated sigmoid  || true
	@echo "[Makefile] Done -> outputs/reports/dca_*_cal_*.csv and outputs/figures/dca_*_cal_*.png"

# -------- DCA (Decision Curve Analysis) --------
.PHONY: dca-batch dca-md inject-dca dca

dca-batch:
	@echo "[Makefile] Running DCA batch (24,48 × val,test × raw/isotonic/sigmoid)"
	@python scripts/dca_batch.py --per-100

dca-md:
	@echo "[Makefile] Building DCA markdown sections (no inject)"
	@python scripts/inject_dca.py --no-inject

inject-dca:
	@echo "[Makefile] Building & injecting DCA sections into README.md / README.zh-CN.md"
	@python scripts/inject_dca.py

dca: dca-batch inject-dca
	@echo "[Makefile] DCA done."

# -------- README polishing (badges + TOC) --------
.PHONY: readme-polish readme-all

readme-polish:
	@echo "[Makefile] Injecting badges + TOC to README..."
	@python scripts/polish_readme.py

# 串联：项目结构 + 消融 + DCA + 阈值片段 + 徽章/TOC
readme-all: structure inject-structure ablation-md inject-ablation dca-batch inject-dca readme-thresholds readme-polish
	@echo "[Makefile] README (ALL) done."
