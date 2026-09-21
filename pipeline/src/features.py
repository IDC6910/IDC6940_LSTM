"""
features.py
------------
Chronological splitting (never shuffle-then-split -- that leaks future
telemetry into training for forecasting) and normalization fit ONLY
on the training partition.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def chronological_split(df: pd.DataFrame, time_col: str = "window_end",
                         train_frac: float = config.TRAIN_FRAC,
                         val_frac: float = config.VAL_FRAC) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by time and cut into contiguous train/val/test blocks. This is
    the correct split for time-series forecasting: random/shuffled
    splits would let the model see telemetry from after a test-set failure
    during training, which silently inflates every metric."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def fit_scaler_on_train(train_df: pd.DataFrame, feature_cols) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols].values)
    return out


def normalize_train_val_test(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                              feature_cols) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit exclusively on train_df, then transform all three splits with
    those same parameters -- this is the leakage guard called out in the
    project brief's normalization step."""
    scaler = fit_scaler_on_train(train_df, feature_cols)
    train_s = apply_scaler(train_df, feature_cols, scaler)
    val_s = apply_scaler(val_df, feature_cols, scaler)
    test_s = apply_scaler(test_df, feature_cols, scaler)
    return train_s, val_s, test_s, scaler
