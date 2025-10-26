from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rel_to_root(p: Path) -> str:
    """将任意路径转换为相对仓库根目录的 POSIX 路径字符串。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        # 若不在仓库树内，尽量只保留从 outputs/ 开始的相对路径
        try:
            idx = p.as_posix().split("/").index("outputs")
            return "/".join(p.as_posix().split("/")[idx:])
        except Exception:
            # 兜底：只返回文件名
            return p.name
