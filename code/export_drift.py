#!/usr/bin/env python3
"""Daily level of a system's top-ranked features, for the drift figure.

    python code/export_drift.py bank
    python code/export_drift.py telecom

Writes results/<system>/drift.csv: one row per (day, feature) with the day's mean value
scaled by the training-day median / IQR, plus the day's share of positive minutes.
The data are cleaned with the chronological split's training days, as in run_lstm.py.
"""
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(os.environ.get("FAILURE_PREDICTION_DIR",
                           Path.home() / "OpenRCA" / "failure_prediction"))
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P                                   # noqa: E402
import run_lstm as R                              # noqa: E402  (numpy select_topk)

H = 5
SYSTEM = sys.argv[1] if len(sys.argv) > 1 else "bank"
OUT = Path(__file__).resolve().parent.parent / "results" / SYSTEM / "drift.csv"

A = P.use_dataset(SYSTEM)
fold = P.chrono_folds(A)[0]
R.prepare_fold(A, fold)                                # training-day cleaning
y = A.Y_SYS[f"h{H}"].to_numpy()
usable = A.MASKS[f"eval_h{H}"].to_numpy().astype(bool)
train = fold.rows("train"); train = train[usable[train]]
med, iqr = P.robust_fit(train, A)
cols = R.select_topk(train, y[train], 4, A=A)

split_of = {d: s for s in ("train", "val", "test") for d in getattr(fold, s)}
rows = []
for day in A.DAYS:
    r = np.flatnonzero(A.DAY_DATE == day)
    ok = r[usable[r]]
    for c in cols:
        rows.append(dict(day=day, split=split_of[day], feature=A.FEATS[c],
                         scaled_mean=float((A.FVAL[r, c].mean() - med[c]) / iqr[c]),
                         positive_rate=float(y[ok].mean())))
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).pivot_table(index="day", columns="feature",
                                     values="scaled_mean").round(2).to_string())
