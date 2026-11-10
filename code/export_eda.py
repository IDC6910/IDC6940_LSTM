#!/usr/bin/env python3
"""Data for the exploration figures of one system.

    python code/export_eda.py bank
    python code/export_eda.py telecom

Writes to results/<system>/eda/:
  missingness.csv    share of missing minutes per series: raw grid, and after the
                     training-day cleaning of the chronological split (before the
                     median fill)
  onset_profile.csv  average change in the telemetry around fault onsets and around
                     random fault-free times, by minute relative to the onset
  closeup.csv        four KPIs of one faulty component around one fault
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_lstm as R                                   # noqa: E402
P = R.P

CLOSEUP = {"bank": ("MG01", "2021-03-04 00:34:00"),
           "telecom": ("docker_004", "2020-04-11 01:40:00")}
BEFORE, AFTER, BASE = 60, 30, (60, 31)   # profile window and baseline (minutes before)

system = sys.argv[1] if len(sys.argv) > 1 else "bank"
OUT = R.RESULTS / system / "eda"
OUT.mkdir(parents=True, exist_ok=True)

A = P.use_dataset(system)
fold = P.chrono_folds(A)[0]
R.prepare_fold(A, fold)
X_all = A.X_ALL

# ---- missingness -------------------------------------------------------------
miss = pd.DataFrame({"feature": X_all.columns, "raw": X_all.isna().mean().to_numpy()})
miss["after_cleaning"] = miss["feature"].map(A.FFILL_MISSING)       # NaN = series dropped
miss.to_csv(OUT / "missingness.csv", index=False)

# ---- onset profile -----------------------------------------------------------
tr = np.isin(A.DAY_DATE, fold.train)
F = A.FVAL
q1, q3 = np.percentile(F[tr], 25, axis=0), np.percentile(F[tr], 75, axis=0)
iqr = np.where(q3 - q1 > 1e-9, q3 - q1, np.nan)
moving = ~np.isnan(iqr)                                 # skip constant series
Fz = F[:, moving] / iqr[moving]
day_idx = np.arange(len(A.GRID)) // A.DAY_MINUTES


def profile(t0):
    """Mean |change from the baseline| over features, for minutes -BEFORE..+AFTER."""
    lo, hi = t0 - BEFORE, t0 + AFTER
    if lo < 0 or hi >= len(F) or day_idx[lo] != day_idx[t0] or day_idx[hi] != day_idx[t0]:
        return None
    base = np.median(Fz[t0 - BASE[0]:t0 - BASE[1] + 1], axis=0)
    seg = np.clip(np.abs(Fz[lo:hi + 1] - base), 0, 10)
    return seg.mean(axis=1)


onsets = [(A.ROW_OF[int(r.onset)], r.reason) for r in A.ONSETS.itertuples()]
rows = []
for i, reason in onsets:
    p = profile(i)
    if p is not None:
        rows += [dict(kind="fault", reason=reason, onset=i, minute=m, change=v)
                 for m, v in zip(range(-BEFORE, AFTER + 1), p)]
fault_rows = np.array([i for i, _ in onsets])
rng = np.random.default_rng(P.SEED)
cands = [t for t in range(len(F))
         if np.abs(fault_rows - t).min() > BEFORE + AFTER]   # far from every fault
for t in rng.choice(cands, size=min(300, len(cands)), replace=False):
    p = profile(int(t))
    if p is not None:
        rows += [dict(kind="random", reason="", onset=int(t), minute=m, change=v)
                 for m, v in zip(range(-BEFORE, AFTER + 1), p)]
prof = pd.DataFrame(rows)
prof.to_csv(OUT / "onset_profile.csv", index=False)
print(prof.groupby(["kind"]).onset.nunique().to_dict(), "windows")

# ---- close-up of one fault ---------------------------------------------------
comp, when = CLOSEUP[system]
t0 = A.ROW_OF[int(A.ONSETS.loc[A.ONSETS["datetime"] == when, "onset"].iloc[0])]
cols = [j for j, f in enumerate(A.FEATS) if f.split("||")[0] == comp]
lo = max(t0 - BEFORE, (t0 // A.DAY_MINUTES) * A.DAY_MINUTES)   # stay inside the day
minutes = np.arange(lo - t0, BEFORE + 1)
seg = F[lo:t0 + BEFORE + 1, cols]
pre = minutes < 0
half = minutes[pre].min() // 2                                  # split the history in two
before, earlier = seg[pre & (minutes >= half)], seg[pre & (minutes < half)]
rng_ = seg.max(0) - seg.min(0)
safe = np.where(rng_ > 0, rng_, 1)
score = np.where(rng_ > 0, np.abs(before.mean(0) - earlier.mean(0)) / safe, 0)
top = np.argsort(-score)[:4]
reason = A.ONSETS.loc[A.ONSETS["datetime"] == when, "reason"].iloc[0]
pd.DataFrame([dict(component=comp, onset=when, reason=reason,
                   kpi=A.FEATS[cols[j]].split("||")[1], minute=int(m),
                   value=float((seg[k, j] - seg[:, j].min()) / safe[j]))
              for j in top for k, m in enumerate(minutes)]
             ).to_csv(OUT / "closeup.csv", index=False)
print("done", system)
