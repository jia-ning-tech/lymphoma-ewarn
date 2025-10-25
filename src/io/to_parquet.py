from __future__ import annotations

import os
from typing import Optional, Sequence, Iterable, List

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat, progress_bar


def _resolve_out_path(table: str, area: str, out_path: Optional[str]) -> str:
    cfg = get_cfg()
    if out_path:
        return out_path
    fname = f"{area}_{table}.parquet"
    return os.path.join(cfg.paths.interim, fname)


def _resolve_csv_files(table: str, area: str) -> List[str]:
    cfg = get_cfg()
    root = cfg.paths.raw_hosp if area == "hosp" else cfg.paths.raw_icu
    patterns = [f"{table}.csv", f"{table}.csv.gz"]
    files: List[str] = []
    for patt in patterns:
        p = os.path.join(root, patt)
        if os.path.exists(p):
            files.append(p)
    if not files:
        import glob
        for patt in patterns:
            files.extend(glob.glob(os.path.join(root, "**", patt), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found for {area}.{table} under {root}")
    return sorted(files)


def _make_scanner_compat(dset: ds.Dataset,
                         columns: Optional[Sequence[str]],
                         filter_expr: Optional[ds.Expression],
                         batch_size: int = 100_000):
    # Try modern API
    try:
        return ds.Scanner.from_dataset(
            dset,
            columns=list(columns) if columns else None,
            filter=filter_expr,
            batch_size=batch_size,
            use_threads=True,
        )
    except Exception:
        pass

    # Fallback wrapper (older versions)
    class _Wrap:
        def __init__(self, dataset, columns, filter_expr, batch_size):
            self.dataset = dataset
            self.columns = list(columns) if columns else None
            self.filter = filter_expr
            self.batch_size = batch_size

        def to_batches(self) -> Iterable[pa.RecordBatch]:
            # Prefer dataset.to_batches if available
            try:
                for rb in self.dataset.to_batches(columns=self.columns, filter=self.filter, batch_size=self.batch_size):
                    yield rb
                return
            except Exception:
                pass
            # Fallback: to_table then manual chunk
            tbl = self.dataset.to_table(columns=self.columns, filter=self.filter)
            n = tbl.num_rows
            start = 0
            while start < n:
                end = min(start + self.batch_size, n)
                yield pa.Table.from_batches([pa.RecordBatch.from_pandas(tbl.slice(start, end - start).to_pandas())])
                start = end
    return _Wrap(dset, columns, filter_expr, batch_size)


def _iter_csv_chunks_pandas(file_path: str,
                            columns: Optional[Sequence[str]],
                            chunksize: int,
                            engine: str) -> Iterable[pd.DataFrame]:
    """
    Iterate CSV file as pandas chunks with specific engine settings.
    """
    kw = dict(
        chunksize=chunksize,
        usecols=list(columns) if columns else None,
        sep=",",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    if engine == "c":
        kw.update(dict(engine="c", low_memory=False))
    else:  # python engine does not support low_memory
        kw.update(dict(engine="python"))
    return pd.read_csv(file_path, **kw)


def _pandas_stream_write(files: List[str],
                         out_path: str,
                         columns: Optional[Sequence[str]],
                         row_group_rows: int,
                         compression: str,
                         limit_rows: Optional[int]) -> int:
    """
    Fallback writer using pandas.read_csv(chunksize=...) -> ParquetWriter.
    Try engine='c' first; if zero chunks read, retry engine='python'.
    """
    writer: pq.ParquetWriter | None = None
    written = 0
    chunksize = 200_000

    def _run_with_engine(engine: str) -> int:
        nonlocal writer, written
        wrote_before = written
        file_iter = progress_bar(files, total=len(files), desc=f"pandas[{engine}] csv→parquet")
        for fp in file_iter:
            # iterate chunks
            try:
                chunk_iter = _iter_csv_chunks_pandas(fp, columns, chunksize, engine=engine)
            except Exception:
                continue
            for chunk in chunk_iter:
                if chunk is None or len(chunk) == 0:
                    continue
                tbl = pa.Table.from_pandas(chunk, preserve_index=False)
                if limit_rows is not None:
                    remain = limit_rows - written
                    if remain <= 0:
                        break
                    if tbl.num_rows > remain:
                        tbl = tbl.slice(0, remain)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, tbl.schema, compression=compression, use_dictionary=True)
                # row groups
                start = 0
                n = tbl.num_rows
                while start < n:
                    end = min(start + row_group_rows, n)
                    sub = tbl.slice(start, end - start)
                    writer.write_table(sub)
                    written += sub.num_rows
                    start = end
                if limit_rows is not None and written >= limit_rows:
                    break
            if limit_rows is not None and written >= limit_rows:
                break
        return written - wrote_before

    # try C engine first
    wrote_c = _run_with_engine("c")
    if wrote_c == 0:
        # retry with python engine
        wrote_py = _run_with_engine("python")
        return wrote_c + wrote_py
    return wrote_c


def _pandas_direct_read_write(files: List[str],
                              out_path: str,
                              columns: Optional[Sequence[str]],
                              compression: str,
                              limit_rows: Optional[int]) -> int:
    """
    Last-resort fallback: direct read (no chunks) then write once.
    Only used when streaming produces zero rows (e.g., rare chunksize issues).
    """
    writer: pq.ParquetWriter | None = None
    written = 0
    # 设一个安全上限，避免误读超大表；若 limit_rows 未给，则最多读 1,000,000 行以保护内存
    max_rows = limit_rows if limit_rows is not None else 1_000_000
    file_iter = progress_bar(files, total=len(files), desc="pandas[direct] csv→parquet")
    for fp in file_iter:
        try:
            df = pd.read_csv(
                fp, nrows=max_rows, usecols=list(columns) if columns else None,
                sep=",", encoding="utf-8", engine="c", low_memory=False, on_bad_lines="skip"
            )
        except Exception:
            # 再试 python 引擎
            df = pd.read_csv(
                fp, nrows=max_rows, usecols=list(columns) if columns else None,
                sep=",", encoding="utf-8", engine="python", on_bad_lines="skip"
            )
        if df is None or len(df) == 0:
            continue
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema, compression=compression, use_dictionary=True)
        writer.write_table(tbl)
        written += tbl.num_rows
        # 若设了 limit_rows，读完一个文件就够了
        if limit_rows is not None and written >= limit_rows:
            break
    if writer is not None:
        writer.close()
    return written


def csv_to_parquet(
    table: str,
    area: str = "hosp",
    out_path: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    filter_expr: Optional[ds.Expression] = None,
    row_group_rows: int = 500_000,
    compression: str = "zstd",
    limit_rows: Optional[int] = None,
) -> str:
    """
    Stream CSV(.gz) -> Parquet with Arrow scanner; if no batches produced, fallback to pandas streaming (C->python);
    if still zero, last-resort direct read.
    """
    logger = get_logger("to_parquet")
    cfg = get_cfg()
    files = _resolve_csv_files(table, area)
    out_path = _resolve_out_path(table, area, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    # Arrow scanner path (best effort)
    try:
        fmt = ds.CsvFileFormat()
        dset = ds.dataset(files, format=fmt)
        scanner = _make_scanner_compat(dset, columns, filter_expr, batch_size=100_000)
        total = None
        try:
            total = dset.count_rows(filter=filter_expr)
            if limit_rows is not None:
                total = min(total, limit_rows)
        except Exception:
            total = None

        logger.info(f"Writing {area}.{table} -> {out_path} (compression={compression})")
        saw_any_batch = False
        with heartbeat(logger, secs=int(cfg.logging.get("heartbeat_secs", 30)), note=f"{area}.{table} -> parquet"):
            batches_iter = progress_bar(scanner.to_batches(), total=total, desc=f"csv→parquet {area}.{table}")
            writer: pq.ParquetWriter | None = None
            for rb in batches_iter:
                saw_any_batch = True
                tbl = rb if isinstance(rb, pa.Table) else pa.Table.from_batches([rb])
                if limit_rows is not None:
                    remain = limit_rows - written
                    if remain <= 0:
                        break
                    if tbl.num_rows > remain:
                        tbl = tbl.slice(0, remain)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, tbl.schema, compression=compression, use_dictionary=True)
                start = 0
                n = tbl.num_rows
                while start < n:
                    end = min(start + row_group_rows, n)
                    chunk = tbl.slice(start, end - start)
                    writer.write_table(chunk)
                    written += chunk.num_rows
                    start = end
                if limit_rows is not None and written >= limit_rows:
                    break
            if writer is not None:
                writer.close()
        if not saw_any_batch:
            logger.warning("Arrow scanner returned no batches; falling back to pandas streaming...")
    except Exception as e:
        logger.warning(f"Arrow scanner path failed ({e}); falling back to pandas streaming...")

    # Pandas fallback if Arrow path wrote nothing
    if written == 0:
        wrote = _pandas_stream_write(files, out_path, columns, row_group_rows, compression, limit_rows)
        written += wrote

    # Last-resort: direct read
    if written == 0:
        logger.warning("Pandas streaming produced zero rows; falling back to direct read...")
        wrote2 = _pandas_direct_read_write(files, out_path, columns, compression, limit_rows)
        written += wrote2

    if written == 0:
        raise RuntimeError(
            f"No rows were written to {out_path}. "
            f"Please check the source CSV files, column names, and filters."
        )

    logger.info(f"Done: {area}.{table} -> {out_path} | rows_written={written}")
    return out_path


# ------------------------------ self-test ----------------------------- #

if __name__ == "__main__":
    lg = get_logger("to_parquet.selftest")
    try:
        path = csv_to_parquet("icustays", area="icu", limit_rows=20000, row_group_rows=10000)
        lg.info(f"Self-test wrote: {path}")
        if os.path.exists(path):
            t = pq.read_table(path, columns=["subject_id", "stay_id", "intime", "outtime"])
            df = t.to_pandas().head()
            lg.info(f"Parquet head:\n{df}")
    except Exception as e:
        lg.error(f"to_parquet self-test failed: {e}")
