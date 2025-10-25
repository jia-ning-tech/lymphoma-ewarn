from __future__ import annotations

import os
import sys
import json
import time
import pathlib
from typing import Any, Mapping, MutableMapping, Optional

# 软依赖：PyYAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

__all__ = ["get_cfg", "reload_cfg", "as_dict"]

# ------------------------ internal helpers ------------------------ #

class _DotDict(dict):
    """dict with attribute access. Nested dicts are converted recursively."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_dotdict(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return _DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    else:
        return obj


def _expand_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return p
    p = os.path.expandvars(os.path.expanduser(str(p)))
    return str(pathlib.Path(p).resolve())


def _ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def _read_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError(
            "Missing dependency 'PyYAML'. Please install it first:\n"
            "  pip install pyyaml\n"
            "or add it to env/requirements.txt and reinstall the environment."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dicts(a: MutableMapping[str, Any], b: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep-ish merge: nested dicts merged recursively, others overridden."""
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, Mapping):
            _merge_dicts(a[k], v)
        else:
            a[k] = v
    return a


def _resolve_paths(cfg: dict) -> dict:
    # expand and ensure key paths
    p = cfg.get("paths", {})
    for key in ("root", "raw_hosp", "raw_icu", "interim", "features", "outputs", "reports"):
        if key in p and isinstance(p[key], str):
            p[key] = _expand_path(p[key])
    _ensure_dir(p.get("interim", _expand_path("./data_interim")))
    _ensure_dir(p.get("features", _expand_path("./data_features")))
    _ensure_dir(p.get("outputs", _expand_path("./outputs")))
    _ensure_dir(p.get("reports", _expand_path("./reports")))

    # logging directory
    lg = cfg.get("logging", {})
    if "log_dir" in lg and isinstance(lg["log_dir"], str):
        lg["log_dir"] = _expand_path(lg["log_dir"])
        _ensure_dir(lg["log_dir"])
    else:
        lg["log_dir"] = os.path.join(p.get("outputs", _expand_path("./outputs")), "logs")
        _ensure_dir(lg["log_dir"])

    # artifacts files -> absolute paths
    arts = cfg.get("artifacts", {})
    for k, v in list(arts.items()):
        if isinstance(v, str):
            if not os.path.isabs(v):
                v = os.path.join(p.get("root", os.getcwd()), v)
            arts[k] = _expand_path(v)

    cfg["paths"] = p
    cfg["logging"] = lg
    cfg["artifacts"] = arts
    return cfg


def _basic_validate(cfg: dict) -> None:
    errs = []
    raw_hosp = cfg.get("paths", {}).get("raw_hosp")
    raw_icu = cfg.get("paths", {}).get("raw_icu")
    if not (raw_hosp and os.path.exists(raw_hosp)):
        errs.append(f"paths.raw_hosp not found: {raw_hosp}")
    if not (raw_icu and os.path.exists(raw_icu)):
        errs.append(f"paths.raw_icu not found: {raw_icu}")

    horizons = cfg.get("prediction", {}).get("horizons_hours", [])
    if not horizons or not all(isinstance(h, int) and h > 0 for h in horizons):
        errs.append("prediction.horizons_hours must be positive integers, e.g., [24, 48]")

    wins = cfg.get("windows", {}).get("hours", [])
    if not wins or not all(isinstance(h, int) and h > 0 for h in wins):
        errs.append("windows.hours must be positive integers, e.g., [6, 12, 24]")

    if errs:
        msg = "Configuration validation failed:\n  - " + "\n  - ".join(errs)
        raise ValueError(msg)


def _find_project_root(start: pathlib.Path, max_up: int = 5) -> pathlib.Path:
    """
    Walk upwards from `start` looking for a directory that contains conf/config.yaml.
    Returns the found directory; raises FileNotFoundError if not found.
    """
    cur = start
    for _ in range(max_up + 1):
        if (cur / "conf" / "config.yaml").exists() and (cur / "conf" / "dictionaries.yaml").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(
        f"Could not locate project root from {start}. "
        f"Tried walking up {max_up} levels looking for conf/config.yaml."
    )

# ------------------------ public API ------------------------ #

_CFG_CACHE: _DotDict | None = None

def get_cfg(reload: bool = False) -> _DotDict:
    """
    Load and return project configuration as a dot-accessible dict.
    This reads conf/config.yaml and conf/dictionaries.yaml at project root.
    """
    global _CFG_CACHE
    if _CFG_CACHE is not None and not reload:
        return _CFG_CACHE

    here = pathlib.Path(__file__).resolve()
    # robust root detection
    root = _find_project_root(here.parent)  # start from src/
    conf_dir = root / "conf"
    config_path = conf_dir / "config.yaml"
    dicts_path = conf_dir / "dictionaries.yaml"

    cfg = _read_yaml(str(config_path))
    dicts = _read_yaml(str(dicts_path))

    # attach dictionaries
    cfg["dictionaries_file"] = str(dicts_path)
    cfg["dictionaries_content"] = dicts

    # resolve paths & ensure dirs
    cfg = _resolve_paths(cfg)

    # runtime meta
    meta = {
        "loaded_at": time.strftime(cfg.get("logging", {}).get("timefmt", "%Y-%m-%d %H:%M:%S"), time.localtime()),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "project_root": str(root),
    }
    cfg.setdefault("runtime", {}).update(meta)

    _basic_validate(cfg)
    _CFG_CACHE = _to_dotdict(cfg)
    return _CFG_CACHE


def reload_cfg() -> _DotDict:
    """Force reload configuration from disk."""
    return get_cfg(reload=True)


def as_dict() -> dict:
    """Return a plain dict copy of the cached configuration."""
    c = get_cfg()
    return json.loads(json.dumps(c))


if __name__ == "__main__":
    try:
        c = get_cfg()
        print("Config loaded OK.")
        print("Project root:", c.runtime.project_root)
        print("Root:", c.paths.root)
        print("Raw hosp:", c.paths.raw_hosp)
        print("Raw icu:", c.paths.raw_icu)
        print("Horizons:", c.prediction.horizons_hours)
        print("Windows:", c.windows.hours)
        print("Logs dir:", c.logging.log_dir)
    except Exception as e:
        print("Failed to load config:", repr(e))
        sys.exit(1)
