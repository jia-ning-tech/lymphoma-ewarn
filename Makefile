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
