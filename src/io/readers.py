from __future__ import annotations

import os
import glob
from typing import Generator, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat, progress_bar


Area = Union[str, None]  # "hosp" | "icu" | None


# ------------------------------ helpers ------------------------------ #

def _table_globs(table: str) -> List[str]:
    """
    Given a logical table name, return common filename patterns.
    MIMIC-IV CSV 通常是 *.csv 或 *.csv.gz
    """
    return [f"{table}.csv", f"{table}.csv.gz"]


def _area_root(area: Area) -> str:
    cfg = get_cfg()
    if area is None:
        # 自动探测（不推荐），返回 raw 根目录
        return cfg.paths.root + "/data_raw"
    if str(area).lower() == "hosp":
        return cfg.paths.raw_hosp
    if str(area).lower() == "icu":
        return cfg.paths.raw_icu
    raise ValueError(f"Unknown area: {area}. Use 'hosp' or 'icu'.")


def _resolve_table_files(table: str, area: Area) -> List[str]:
    """
    Resolve table -> existing file paths (supports .csv and .csv.gz)
    """
    root = _area_root(area)
    files = []
    for patt in _table_globs(table):
        files.extend(glob.glob(os.path.join(root, patt)))
    # 某些表在子目录中（较少见），可选：再扫描一层
    if not files:
        for patt in _table_globs(table):
            files.extend(glob.glob(os.path.join(root, "**", patt), recursive=True))
    return sorted(files)


def list_available_tables(area: Area = "hosp") -> List[str]:
    """
    List unique table base names under data_raw/<area>.
    """
    root = _area_root(area)
    out = set()
    for p in glob.glob(os.path.join(root, "**", "*.csv*"), recursive=True):
        base = os.path.basename(p)
        if base.endswith(".gz"):
            base = base[:-3]
        if base.endswith(".csv"):
            base = base[:-4]
        out.add(base)
    return sorted(out)


def dataset_for(table: str, area: Area = "hosp") -> ds.Dataset:
    """
    Build a pyarrow Dataset for given table.
    """
    files = _resolve_table_files(table, area)
    if not files:
        raise FileNotFoundError(f"No files found for table '{table}' in area '{area}'. "
                                f"Looked under: {_area_root(area)} with patterns: {_table_globs(table)}")
    fmt = ds.CsvFileFormat()  # 支持 .csv/.csv.gz
    return ds.dataset(files, format=fmt, ignore_invalid_files=True)


def make_filter_equals(equals: Optional[dict] = None) -> Optional[ds.Expression]:
    """
    Convenience: build ds.Expression of equality conditions combined by AND.
    Example: make_filter_equals({'subject_id': 123, 'hadm_id': 456})
    """
    if not equals:
        return None
    expr = None
    for k, v in equals.items():
        term = ds.field(k) == v
        expr = term if expr is None else (expr & term)
    return expr


# ------------------------------ scanners ----------------------------- #

def scan_batches(
    table: str,
    area: Area = "hosp",
    columns: Optional[Sequence[str]] = None,
    filter_expr: Optional[ds.Expression] = None,
    batch_size: int = 50_000,
    to_pandas_kwargs: Optional[dict] = None,
    show_progress_desc: Optional[str] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream-scanning a large CSV as pandas DataFrame batches with column projection and predicate pushdown.
    - columns: select only needed columns (much faster & memory friendly)
    - filter_expr: pyarrow.dataset.Expression for pushdown
    - batch_size: rows per batch to convert to pandas
    """
    logger = get_logger("readers")
    cfg = get_cfg()

    dset = dataset_for(table, area)
    scanner = dset.scan(
        columns=list(columns) if columns else None,
        filter=filter_expr,
        batch_size=batch_size,
        use_threads=True,
    )

    to_pandas_kwargs = to_pandas_kwargs or {"use_threads": True}

    # 估算总行数（可选，可能会触发一次统计扫描；对极大表可跳过）
    total = None
    try:
        total = dset.count_rows(filter=filter_expr)
    except Exception:
        total = None

    desc = show_progress_desc or f"scan {area}.{table}"
    bar_iter = progress_bar(scanner.to_batches(), total=total, desc=desc)

    with heartbeat(logger, secs=int(cfg.logging.get("heartbeat_secs", 30)), note=f"scan {area}.{table}"):
        for rb in bar_iter:
            if isinstance(rb, pa.RecordBatch):
                tbl = pa.Table.from_batches([rb])
            elif isinstance(rb, pa.Table):
                tbl = rb
            else:
                # 某些版本 to_batches() 直接返回 RecordBatch
                tbl = pa.Table.from_batches([rb])
            # 转 pandas
            df = tbl.to_pandas(**to_pandas_kwargs)
            yield df


def read_pandas(
    table: str,
    area: Area = "hosp",
    columns: Optional[Sequence[str]] = None,
    filter_expr: Optional[ds.Expression] = None,
    limit: Optional[int] = None,
    to_pandas_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Read (relatively small/filtered) table into a single pandas DataFrame.
    For huge tables use scan_batches().
    """
    dset = dataset_for(table, area)
    # 若设置 limit，用 take/head 进行截断（pyarrow 9+ 可用 head）
    if limit is not None and hasattr(dset, "head"):
        tbl = dset.head(limit, columns=list(columns) if columns else None)
        if filter_expr is not None:
            # head 不支持 filter，退回到 scan
            # 这里改为用 scanner + 限制条数
            cnt = 0
            dfs = []
            for df in scan_batches(table, area, columns, filter_expr, batch_size=50_000):
                dfs.append(df)
                cnt += len(df)
                if cnt >= limit:
                    break
            if not dfs:
                return pd.DataFrame(columns=list(columns) if columns else None)
            out = pd.concat(dfs, ignore_index=True)
            return out.iloc[:limit].reset_index(drop=True)
        else:
            return tbl.to_pandas(**(to_pandas_kwargs or {"use_threads": True}))

    # 无 limit 或需要 filter：使用 scanner
    dfs = []
    cnt = 0
    for df in scan_batches(table, area, columns, filter_expr, batch_size=100_000,
                           show_progress_desc=f"read {area}.{table}"):
        dfs.append(df)
        cnt += len(df)
        if limit is not None and cnt >= limit:
            break
    if not dfs:
        return pd.DataFrame(columns=list(columns) if columns else None)
    out = pd.concat(dfs, ignore_index=True)
    if limit is not None:
        out = out.iloc[:limit].reset_index(drop=True)
    return out


def count_rows(table: str, area: Area = "hosp", filter_expr: Optional[ds.Expression] = None) -> Optional[int]:
    """
    Count rows quickly using pyarrow dataset (may be approximate on CSV).
    Returns None if not supported or fails.
    """
    try:
        dset = dataset_for(table, area)
        return int(dset.count_rows(filter=filter_expr))
    except Exception:
        return None


# ------------------------------ self-test ---------------------------- #

if __name__ == "__main__":
    logger = get_logger("readers.selftest")
    cfg = get_cfg()
    logger.info("readers self-test: listing tables under data_raw/ ...")
    try:
        hosp_tables = list_available_tables("hosp")
        icu_tables = list_available_tables("icu")
        logger.info(f"hosp tables (sample): {hosp_tables[:10]}")
        logger.info(f"icu  tables (sample): {icu_tables[:10]}")
    except Exception as e:
        logger.error(f"list tables failed: {e}")

    # 轻量 smoke：若存在 patients/admissions，就尝试读取前几行与列裁剪
    for area, tbl, cols in [
        ("hosp", "patients", ["subject_id", "anchor_age", "gender"]),
        ("hosp", "admissions", ["subject_id", "hadm_id", "admittime"]),
        ("icu",  "icustays", ["subject_id", "stay_id", "intime", "outtime"]),
    ]:
        try:
            files = _resolve_table_files(tbl, area)
            if files:
                logger.info(f"Try head read: {area}.{tbl} -> files[0]={os.path.basename(files[0])}")
                df = read_pandas(tbl, area, columns=cols, limit=5)
                logger.info(f"{area}.{tbl} head:\n{df}")
            else:
                logger.info(f"Skip {area}.{tbl}: not found.")
        except Exception as e:
            logger.warning(f"Self-test failed on {area}.{tbl}: {e}")
