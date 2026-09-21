"""
sequence_builder.py
--------------------
Slides a fixed-length lookback window (config.LOOKBACK_WINDOW_MIN) over the
regular-grid telemetry to produce:
  - X_seq: 3D array (n_windows, timesteps, n_features) for the LSTM
  - X_flat: 2D array (n_windows, n_features * k summary stats) for the
    non-sequential baselines (LogReg / RF / XGBoost), built from the SAME
    windows so the comparison in Experiment 1 is apples-to-apples -- both
    families see exactly the same information, just shaped differently.
  - y: label vector for a given horizon
  - meta: window_end timestamps + lead time, for lead-time analysis later
"""
from typing import Tuple

import numpy as np
import pandas as pd

import config


def build_windows(wide: pd.DataFrame, window_minutes: int = config.LOOKBACK_WINDOW_MIN,
                   stride_minutes: int = config.STRIDE_MIN) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Returns:
      X_seq: (n_windows, timesteps, n_features) float32 array
      window_ends: DatetimeIndex of each window's final timestamp (this is
        what gets passed to labeling.label_for_all_horizons)

    Windows that contain any leftover NaN (gaps too large to interpolate,
    per resampling.fill_small_gaps) are dropped.
    """
    freq_delta = pd.Timedelta(config.RESAMPLE_FREQ)
    steps = int(round(pd.Timedelta(minutes=window_minutes) / freq_delta))
    stride = max(1, int(round(pd.Timedelta(minutes=stride_minutes) / freq_delta)))

    values = wide.values.astype(np.float32)
    index = wide.index
    n_rows, n_features = values.shape

    starts = range(0, n_rows - steps + 1, stride)
    seqs = []
    ends = []
    for s in starts:
        e = s + steps
        chunk = values[s:e]
        if np.isnan(chunk).any():
            continue
        seqs.append(chunk)
        ends.append(index[e - 1])

    if not seqs:
        return np.empty((0, steps, n_features), dtype=np.float32), pd.DatetimeIndex([])

    X_seq = np.stack(seqs, axis=0)
    window_ends = pd.DatetimeIndex(ends)
    return X_seq, window_ends


def flatten_for_baselines(X_seq: np.ndarray, feature_names) -> Tuple[np.ndarray, list]:
    """Summarize each window's time axis into a handful of statistics per
    feature (mean, std, min, max, last value, slope). This is a standard,
    defensible way to give non-sequential models access to the same
    windowed information the LSTM sees, without literally flattening
    timesteps*features into thousands of columns."""
    if X_seq.shape[0] == 0:
        return np.empty((0, 0)), []

    mean = X_seq.mean(axis=1)
    std = X_seq.std(axis=1)
    mn = X_seq.min(axis=1)
    mx = X_seq.max(axis=1)
    last = X_seq[:, -1, :]
    first = X_seq[:, 0, :]
    slope = (last - first) / max(X_seq.shape[1] - 1, 1)

    X_flat = np.concatenate([mean, std, mn, mx, last, slope], axis=1)
    flat_names = (
        [f"{n}__mean" for n in feature_names]
        + [f"{n}__std" for n in feature_names]
        + [f"{n}__min" for n in feature_names]
        + [f"{n}__max" for n in feature_names]
        + [f"{n}__last" for n in feature_names]
        + [f"{n}__slope" for n in feature_names]
    )
    return X_flat, flat_names
