#!/usr/bin/env python3
"""Sanity checks for a Stage 1 build: timezone alignment, label mechanics, leakage
guards, and a quick "is there signal at all?" probe before any modelling.

    python3 verify_stage1.py --dataset bank
    python3 verify_stage1.py --dataset telecom
    python3 verify_stage1.py --dataset all
"""
import argparse
from datetime import datetime
import numpy as np, pandas as pd
from config import TZ, get_dataset, DATASETS


def check(ds):
    print(f"\n{'=' * 62}\n{ds.name.upper()}  ({ds.dirname}/)  "
          f"{ds.day_minutes} min/day\n{'=' * 62}")
    F  = pd.read_parquet(ds.out / "features_filled.parquet")
    ys = pd.read_parquet(ds.out / "labels_system.parquet")
    mk = pd.read_parquet(ds.out / "masks.parquet")
    on = pd.read_csv(ds.out / "onsets.csv")
    H  = list(ds.horizons)

    print("== 1. grid / timezone ==")
    g = F.index.to_numpy()
    n_days = len(g) // ds.day_minutes
    print("  rows:", len(g), "| unique:", len(set(g.tolist())),
          "| days:", n_days, "| step:", set(np.diff(g)[:ds.day_minutes - 1].tolist()))
    print("  first:", datetime.fromtimestamp(g[0], TZ), " last:", datetime.fromtimestamp(g[-1], TZ))
    print("  all onsets land on the grid:", bool(on["in_grid"].all()),
          f"({int(on['in_grid'].sum())}/{len(on)})")
    print("  record.csv datetime == our UTC+8 rendering for every row:",
          bool((on["datetime"] == on["onset_dt_utc8"]).all()))

    print("\n== 2. label mechanics ==")
    idx = {t: i for i, t in enumerate(g.tolist())}
    pid = np.tile(np.arange(ds.day_minutes), n_days)
    bad = 0
    for _, r in on[on["in_grid"]].iterrows():
        i = idx[r["onset"]]
        for h in H:
            for lead in range(1, h + 1):
                j = i - lead
                if j >= 0 and pid[j] < pid[i]:            # same UTC+8 day
                    bad += int(ys[f"h{h}"].iloc[j] != 1)
    print("  mislabelled (onset, horizon, lead) combinations:", bad,
          f"across all {int(on['in_grid'].sum())} on-grid faults")
    print("  onset minutes flagged in_fault:",
          int(mk["in_fault"].iloc[[idx[t] for t in on.loc[on['in_grid'], 'onset']]].sum()),
          "/", int(on["in_grid"].sum()))

    hf = 5 if 5 in H else H[0]
    print(f"\n== 3. is there signal? point-biserial |corr| with the h={hf} label ==")
    m = mk[f"eval_h{hf}"].astype(bool).to_numpy()
    y = ys[f"h{hf}"].to_numpy()[m]
    X = F.loc[m].fillna(0.0)
    sd = X.std(axis=0).replace(0, np.nan)
    corr = ((X - X.mean()).mul(y - y.mean(), axis=0).mean() / (sd * y.std())).abs()
    print(corr.sort_values(ascending=False).head(8).round(3).to_string())
    print("  features with |r| > 0.05:", int((corr > 0.05).sum()), "of", len(corr))

    print("\n== 4. leakage guards ==")
    for h in H:
        m = mk[f"eval_h{h}"].astype(bool).to_numpy()
        print(f"  h{h:<3d} no evaluable row in the last {h} min of a day:",
              not m[pid >= ds.day_minutes - h].any(),
              f"| none in the first {ds.lookback - 1} min:", not m[pid < ds.lookback - 1].any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="bank | telecom | all")
    a = ap.parse_args()
    names = list(DATASETS) if a.dataset.lower() == "all" else [a.dataset]
    for n in names:
        check(get_dataset(n))


if __name__ == "__main__":
    main()
