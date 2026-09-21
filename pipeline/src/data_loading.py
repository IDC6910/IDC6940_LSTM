"""
data_loading.py
----------------
Reads the raw OpenRCA directory structure:

    dataset/{SYSTEM}/query.csv
    dataset/{SYSTEM}/record.csv
    dataset/{SYSTEM}/telemetry/{YYYY_MM_DD}/{metric,log,trace}/*.csv

and returns concatenated DataFrames in the canonical column names
defined in config.COLUMN_MAP. Kept deliberately simple so that 
loading and cleaning stay independently testable.
"""
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

import config


def discover_date_dirs(system: str = config.SYSTEM) -> List[Path]:
    telemetry_dir = config.DATASET_ROOT / system / "telemetry"
    if config.DATE_DIRS is not None:
        return [telemetry_dir / d for d in config.DATE_DIRS]
    if not telemetry_dir.exists():
        return []
    return sorted(d for d in telemetry_dir.iterdir() if d.is_dir())


def _rename_to_canonical(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rename raw columns -> canonical names per config.COLUMN_MAP[kind].
    Silently ignores canonical fields that aren't present in this file
    (e.g. trace 'duration' may need deriving from start/end elsewhere)."""
    mapping = config.COLUMN_MAP[kind]
    inverse = {raw: canon for canon, raw in mapping.items() if raw in df.columns}
    return df.rename(columns=inverse)


def _load_one_kind(kind: str, system: str, date_dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for date_dir in date_dirs:
        kind_dir = date_dir / kind
        if not kind_dir.exists():
            continue
        for f in sorted(kind_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"  [warn] skipping unreadable file {f}: {e}")
                continue
            if df.empty:
                continue
            df = _rename_to_canonical(df, kind)
            df["__source_file"] = f.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def load_metrics(system: str = config.SYSTEM,
                  date_dirs: Optional[Iterable[Path]] = None) -> pd.DataFrame:
    date_dirs = list(date_dirs) if date_dirs is not None else discover_date_dirs(system)
    return _load_one_kind("metric", system, date_dirs)


def load_logs(system: str = config.SYSTEM,
              date_dirs: Optional[Iterable[Path]] = None) -> pd.DataFrame:
    date_dirs = list(date_dirs) if date_dirs is not None else discover_date_dirs(system)
    return _load_one_kind("log", system, date_dirs)


def load_traces(system: str = config.SYSTEM,
                 date_dirs: Optional[Iterable[Path]] = None) -> pd.DataFrame:
    date_dirs = list(date_dirs) if date_dirs is not None else discover_date_dirs(system)
    return _load_one_kind("trace", system, date_dirs)


def _parse_ts(series: pd.Series) -> pd.Series:
    """OpenRCA timestamps show up as either epoch seconds/ms or ISO strings
    depending on export. Try numeric first, fall back to string parsing."""
    if pd.api.types.is_numeric_dtype(series):
        # heuristic: ms epoch is ~13 digits, s epoch is ~10 digits
        sample = series.dropna().iloc[0] if series.notna().any() else 0
        unit = "ms" if sample > 10**12 else "s"
        ts = pd.to_datetime(series, unit=unit, utc=True)
    else:
        ts = pd.to_datetime(series, utc=True, errors="coerce")
    # raw clock is Asia/Shanghai; if to_datetime gave tz-naive-turned-UTC
    # (no offset info in string), localize properly instead of assuming UTC.
    return ts


def load_failure_records(system: str = config.SYSTEM) -> pd.DataFrame:
    """query.csv holds the ground-truth root-cause datetime per failure case
    (the label source). record.csv holds broader historical issue records.
    We primarily need query.csv's occurrence datetime for labeling."""
    path = config.DATASET_ROOT / system / "query.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download OpenRCA and place it under "
            f"{config.DATASET_ROOT}/{system}/ before running the pipeline."
        )
    df = pd.read_csv(path)
    return df


def standardize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return df
    df = df.copy()
    df[ts_col] = _parse_ts(df[ts_col])
    df = df.dropna(subset=[ts_col])
    return df
