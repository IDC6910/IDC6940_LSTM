"""
labeling.py
-----------
Builds the binary "impending failure within H minutes" label. query.csv's 
ground truth holds a "prediction" field (per the OpenRCA README) that is 
a JSON-like string containing the root-cause occurrence datetime, in the 
form "[%Y-%m-%d %H:%M:%S]". We extract that occurrence datetime as each 
failure's onset. If local query.csv instead exposes it as a plain
column, `extract_failure_onsets` falls back to common column-name guesses --
check with src/inspect_schema.py and adjust `CANDIDATE_ONSET_COLUMNS` below
if neither path matches file.
"""
import ast
import re
from typing import List

import numpy as np
import pandas as pd

import config

CANDIDATE_ONSET_COLUMNS = ["occurrence_datetime", "datetime", "root_cause_datetime", "time"]
ONSET_REGEX = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def extract_failure_onsets(query_df: pd.DataFrame) -> pd.Series:
    """Return a tz-aware (Asia/Shanghai localized) Series of failure onset
    timestamps, one per row of query.csv / record.csv."""
    if "prediction" in query_df.columns:
        onsets = []
        for raw in query_df["prediction"]:
            ts = _parse_onset_from_prediction_field(raw)
            onsets.append(ts)
        s = pd.Series(onsets)
    else:
        col = next((c for c in CANDIDATE_ONSET_COLUMNS if c in query_df.columns), None)
        if col is None:
            raise ValueError(
                "Could not find a failure-onset datetime column. Inspect "
                "query.csv with src/inspect_schema.py and add the real "
                "column name to CANDIDATE_ONSET_COLUMNS in labeling.py."
            )
        s = pd.to_datetime(query_df[col], errors="coerce")

    s = pd.to_datetime(s, errors="coerce")
    # Raw OpenRCA datetimes are Asia/Shanghai local time with no offset info.
    s = s.dt.tz_localize(config.RAW_TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    s = s.dt.tz_convert("UTC")
    return s.dropna()


def _parse_onset_from_prediction_field(raw) -> pd.Timestamp:
    if pd.isna(raw):
        return pd.NaT
    text = str(raw)
    m = ONSET_REGEX.search(text)
    if m:
        return pd.Timestamp(m.group(1))
    # fall back to trying to literal_eval a dict-like string
    try:
        d = ast.literal_eval(text)
        for key in ("occurrence datetime", "occurrence_datetime", "datetime"):
            if isinstance(d, dict) and key in d:
                return pd.Timestamp(d[key])
    except (ValueError, SyntaxError):
        pass
    return pd.NaT


def label_windows(window_end_times: pd.DatetimeIndex, failure_onsets: pd.Series,
                   horizon_min: int = config.DEFAULT_HORIZON_MIN,
                   exclude_post_onset_min: int = config.EXCLUDE_POST_ONSET_MIN) -> pd.DataFrame:
    """
    For each window (identified by its END timestamp), produce:
      - label: 1 if some failure onset falls in (window_end, window_end + horizon]
      - keep:  False if window_end falls within exclude_post_onset_min minutes
               AFTER any failure onset (i.e. telemetry mid-incident), so those
               rows can be dropped rather than mislabeled as clean negatives.
      - minutes_to_failure: for positives, exact lead time (useful for
        eval / lead-time analysis later).

    Implemented with numpy broadcasting via searchsorted for speed -- this
    is the part that gets called once per horizon in Experiment 2, so it
    needs to be cheap even with hundreds of thousands of windows.
    """
    onsets = pd.DatetimeIndex(sorted(failure_onsets))
    ends = pd.DatetimeIndex(window_end_times)
    tz = ends.tz

    horizon = pd.Timedelta(minutes=horizon_min)
    exclude = pd.Timedelta(minutes=exclude_post_onset_min)

    n = len(ends)
    label = np.zeros(n, dtype=int)
    minutes_to_failure = np.full(n, np.nan)
    keep = np.ones(n, dtype=bool)

    if len(onsets) == 0:
        return pd.DataFrame({
            "window_end": ends, "label": label,
            "minutes_to_failure": minutes_to_failure, "keep": keep,
        })

    # Use pandas Series (not tz-aware DatetimeIndex + np.where, which
    # silently drops tz info) so NaT-masking keeps timezone intact.
    pos_idx = onsets.searchsorted(ends, side="right")
    has_future_onset = pos_idx < len(onsets)
    next_onset = pd.Series(
        onsets[np.clip(pos_idx, 0, len(onsets) - 1)], index=range(n)
    )
    next_onset[~has_future_onset] = pd.NaT

    lead_td = next_onset.values - ends.values  # timedelta64[ns] array, NaT-safe
    lead_min = lead_td / np.timedelta64(1, "m")
    within_horizon = has_future_onset & (lead_min <= horizon_min) & (lead_min >= 0)
    label[within_horizon] = 1
    minutes_to_failure[within_horizon] = lead_min[within_horizon]

    # exclusion: window_end within `exclude` minutes AFTER any onset
    prev_idx = onsets.searchsorted(ends, side="right") - 1
    has_past_onset = prev_idx >= 0
    prev_onset = pd.Series(
        onsets[np.clip(prev_idx, 0, len(onsets) - 1)], index=range(n)
    )
    prev_onset[~has_past_onset] = pd.NaT
    since_prev = ends.values - prev_onset.values
    since_prev_min = since_prev / np.timedelta64(1, "m")
    mid_incident = has_past_onset & (since_prev_min <= exclude_post_onset_min) & (since_prev_min >= 0)
    keep = ~mid_incident

    return pd.DataFrame({
        "window_end": ends,
        "label": label,
        "minutes_to_failure": minutes_to_failure,
        "keep": keep,
    })


def label_for_all_horizons(window_end_times: pd.DatetimeIndex, failure_onsets: pd.Series,
                            horizons: List[int] = config.PREDICTION_HORIZONS_MIN) -> pd.DataFrame:
    """Convenience wrapper for Experiment 2: one label column per horizon,
    all sharing the same 'keep' mid-incident exclusion (computed once at the
    longest horizon's exclusion window -- exclusion doesn't depend on horizon)."""
    out = pd.DataFrame({"window_end": window_end_times})
    keep_col = None
    for h in horizons:
        res = label_windows(window_end_times, failure_onsets, horizon_min=h)
        out[f"label_{h}min"] = res["label"].values
        out[f"lead_min_{h}min"] = res["minutes_to_failure"].values
        if keep_col is None:
            keep_col = res["keep"].values
    out["keep"] = keep_col
    return out
