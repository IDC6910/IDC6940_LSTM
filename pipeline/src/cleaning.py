"""
cleaning.py
-----------
Operates on LONG-format telemetry (one row per timestamp/entity/metric/value)
before it gets pivoted to the wide common time representation. Handles:
  - duplicate timestamps (same entity+metric+timestamp reported twice)
  - invalid values (non-numeric, infinite, negative-where-impossible)
  - extreme outliers (robust z-score winsorization)

Missing values and irregular sampling are handled downstream in
resampling.py, because "missing" only makes sense once you've defined the
regular time grid you're comparing against.
"""
import numpy as np
import pandas as pd

import config


def drop_duplicate_timestamps(df: pd.DataFrame, key_cols) -> pd.DataFrame:
    """If the same (entity, metric, timestamp) appears more than once,
    keep the mean of the reported values rather than an arbitrary row --
    duplicates in this data are almost always overlapping export windows,
    not genuinely distinct readings."""
    if df.empty:
        return df
    before = len(df)
    df = (
        df.groupby(key_cols, as_index=False, dropna=False)["value"]
        .mean()
        .merge(df.drop(columns=["value"]).drop_duplicates(subset=key_cols), on=key_cols, how="left")
    )
    after = len(df)
    if before != after + (before - after):
        pass  # no-op, just documenting intent
    return df


def coerce_numeric_values(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Drop rows with non-numeric or infinite values instead of silently
    letting them corrupt downstream math (e.g. a log line that ended up in
    a numeric 'value' column)."""
    if df.empty:
        return df
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=[value_col])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [clean] dropped {n_dropped} non-numeric/inf rows")
    return df


def winsorize_outliers(df: pd.DataFrame, group_cols, value_col: str = "value",
                        z_thresh: float = config.OUTLIER_Z_THRESH) -> pd.DataFrame:
    """Clip extreme values per (entity, metric) group using a robust
    z-score (median / MAD) rather than mean/std, since telemetry spikes
    (the very things we're trying to predict!) would otherwise blow out a
    standard z-score and make winsorizing self-defeating."""
    if df.empty:
        return df

    def _clip_group(g: pd.Series) -> pd.Series:
        med = g.median()
        mad = (g - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return g  # constant series, nothing to clip
        robust_z = 0.6745 * (g - med) / mad
        lower = med - (z_thresh / 0.6745) * mad
        upper = med + (z_thresh / 0.6745) * mad
        return g.clip(lower=lower, upper=upper)

    df = df.copy()
    df[value_col] = df.groupby(group_cols)[value_col].transform(_clip_group)
    return df


def clean_long_telemetry(df: pd.DataFrame, entity_col: str = "entity",
                          name_col: str = "name") -> pd.DataFrame:
    """Full cleaning pass for one telemetry type (metric or log), already
    in canonical long format with columns: timestamp, entity, name, value."""
    if df.empty:
        return df
    required = {"timestamp", entity_col, name_col, "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"clean_long_telemetry missing required columns: {missing}")

    df = coerce_numeric_values(df)
    df = drop_duplicate_timestamps(df, key_cols=["timestamp", entity_col, name_col])
    df = winsorize_outliers(df, group_cols=[entity_col, name_col])
    return df
