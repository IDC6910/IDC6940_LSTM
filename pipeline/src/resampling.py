"""
resampling.py
-------------
Turns cleaned long-format telemetry into the common time representation
described in the project brief:

    timestamp        cpu    memory    network    latency    ...
    12:00:00         ...
    12:00:01         ...

Steps:
  1. Pivot (timestamp, entity, metric_name) -> one column per entity.metric
  2. Resample onto a fixed regular grid (config.RESAMPLE_FREQ), which is how
     we handle irregular / inconsistent original sampling rates.
  3. Fill small gaps via interpolation; leave (and flag) larger gaps rather
     than fabricating data far from any real observation.
"""
import numpy as np
import pandas as pd

import config


def pivot_long_to_wide(df: pd.DataFrame, entity_col: str = "entity",
                        name_col: str = "name", value_col: str = "value") -> pd.DataFrame:
    """One column per (entity, metric) pair, e.g. 'podA.cpu_util'."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["__col"] = df[entity_col].astype(str) + "." + df[name_col].astype(str)
    wide = df.pivot_table(index="timestamp", columns="__col", values=value_col, aggfunc="mean")
    wide = wide.sort_index()
    return wide


def resample_to_grid(wide: pd.DataFrame, freq: str = config.RESAMPLE_FREQ) -> pd.DataFrame:
    """Resample onto a fixed grid. Mean-aggregate multiple raw points that
    fall in the same bin; leave bins with zero raw points as NaN (handled
    next by fill_small_gaps) rather than silently zero-filling, which would
    be indistinguishable from a genuine zero reading."""
    if wide.empty:
        return wide
    resampled = wide.resample(freq).mean()
    return resampled


def fill_small_gaps(wide: pd.DataFrame,
                     max_gap: int = config.MAX_GAP_TO_INTERPOLATE) -> pd.DataFrame:
    """Linearly interpolate runs of up to `max_gap` consecutive missing
    samples per column. Longer gaps are left as NaN -- do not forward-fill
    indefinitely, since that would fabricate flat telemetry across outages
    or missing-data windows, which is exactly the kind of window we can't
    trust for training."""
    if wide.empty:
        return wide
    out = wide.copy()
    for col in out.columns:
        out[col] = out[col].interpolate(method="linear", limit=max_gap, limit_area="inside")
    return out


def add_missingness_flags(wide: pd.DataFrame) -> pd.DataFrame:
    """Adds one *_isnan indicator column per feature before any further
    fill, so models can distinguish 'observed' from 'imputed/missing' --
    this is cheap and often meaningfully improves failure prediction, since
    missing telemetry is itself sometimes a symptom."""
    if wide.empty:
        return wide
    flags = wide.isna().astype(int).add_suffix("__isnan")
    return pd.concat([wide, flags], axis=1)


def build_common_time_representation(long_df: pd.DataFrame, freq: str = config.RESAMPLE_FREQ,
                                      entity_col: str = "entity", name_col: str = "name",
                                      value_col: str = "value") -> pd.DataFrame:
    """End-to-end: long cleaned telemetry -> wide, regular-grid DataFrame
    with missingness flags. Rows that are entirely NaN after resampling
    (no raw data anywhere near that timestamp) are dropped rather than
    kept as phantom rows with nothing to learn from."""
    wide = pivot_long_to_wide(long_df, entity_col, name_col, value_col)
    wide = resample_to_grid(wide, freq)
    wide = fill_small_gaps(wide)
    wide = add_missingness_flags(wide)
    value_cols = [c for c in wide.columns if not c.endswith("__isnan")]
    wide = wide.dropna(subset=value_cols, how="all")
    return wide
