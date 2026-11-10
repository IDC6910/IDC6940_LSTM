#!/usr/bin/env python3
"""Save the raw minute x series table with every series kept.

    python code/build_all_series.py bank
    python code/build_all_series.py telecom

The pipeline's Stage 1 (build_dataset.py) keeps only series present on every day and
drops series that are mostly missing, judged over all days including the test days.
To keep test days out of those decisions, this script saves the table *before* any
filtering (build/<system>/features_all.parquet); run_lstm.py then filters and fills
it using the training days of each split only.
"""
import os, sys
from dataclasses import replace
from pathlib import Path

PIPE = Path(os.environ.get("FAILURE_PREDICTION_DIR",
                           Path.home() / "OpenRCA" / "failure_prediction"))
sys.path.insert(0, str(PIPE))
from config import get_dataset                         # noqa: E402
from build_dataset import build_features               # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "bank"
ds = replace(get_dataset(name), min_day_coverage=0.0)   # keep every series
days = sorted(p.name for p in ds.telemetry.iterdir() if p.is_dir() and p.name[:2].isdigit())
X, keep, _ = build_features(ds, days)
out = ds.out / "features_all.parquet"
X.to_parquet(out)
print(f"{name}: {X.shape[0]} minutes x {X.shape[1]} series -> {out}")
