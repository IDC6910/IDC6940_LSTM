#!/usr/bin/env python3
"""Stage 1: raw OpenRCA telemetry -> aligned minute-level matrices.

Works for any dataset registered in config.DATASETS (bank, telecom).

    python3 build_dataset.py --dataset bank
    python3 build_dataset.py --dataset telecom
    python3 build_dataset.py --dataset all

Outputs (in build/<dataset>/):
  features.parquet           minute x feature float32 matrix, index = epoch seconds
  features_filled.parquet    same, gap-filled -- the model input
  feature_index.csv          one row per feature (family, component, kpi, missingness)
  labels_system.parquet      minute x horizon binary labels (any component fails)
  labels_component.parquet   MultiIndex (horizon, component) binary labels
  masks.parquet              eval_h{H} usable-row masks + in_fault flag
  onsets.csv                 fault onsets, epoch + UTC+8 datetime, snapped to grid
  meta.json                  day list, grid shape, knobs used
  report.md                  build diagnostics
"""
import argparse, json, sys
from datetime import datetime
import numpy as np
import pandas as pd

from config import TZ, MINUTE, get_dataset, DATASETS


# ----------------------------------------------------------------- loading --
def day_grid(ds, day_name):
    """Epoch seconds for one day folder, on the dataset's own minute grid."""
    y, m, d = (int(x) for x in day_name.split("_"))
    start = int(datetime(y, m, d, 0, 0, tzinfo=TZ).timestamp()) + ds.day_offset_min * MINUTE
    return np.arange(start, start + ds.day_minutes * MINUTE, MINUTE, dtype=np.int64)


def _minutes(series, unit):
    t = pd.to_numeric(series, errors="coerce").astype("int64")
    if unit == "ms":
        t = t // 1000
    return (t // MINUTE) * MINUTE


def load_source(day_dir, src, grid):
    """Read one telemetry file into a minute x feature frame ('component||kpi')."""
    path = day_dir / src.path
    if not path.exists():
        return pd.DataFrame(index=grid)

    if src.kind == "long":
        df = pd.read_csv(path, usecols=src.required_cols(),
                         dtype={src.id_col: "category", src.kpi_col: "category"})
        df["_t"] = _minutes(df[src.ts_col], src.ts_unit)
        kpi = df[src.kpi_col].astype(str)
        if src.prefix_kpi:
            kpi = src.family + "." + kpi
        df["_col"] = df[src.id_col].astype(str) + "||" + kpi
        wide = df.pivot_table(index="_t", columns="_col", values=src.val_col,
                              aggfunc="mean", observed=True)
    else:                                            # wide
        df = pd.read_csv(path, usecols=src.required_cols())
        df["_t"] = _minutes(df[src.ts_col], src.ts_unit)
        parts = []
        for field in src.value_cols:
            w = df.pivot_table(index="_t", columns=src.id_col, values=field, aggfunc="mean")
            comp = src.component or src.family
            w.columns = [f"{comp}||{c}.{field}" for c in w.columns]
            parts.append(w)
        wide = pd.concat(parts, axis=1) if parts else pd.DataFrame()

    wide = wide.reindex(grid)
    wide.index.name = "timestamp"
    return wide


def build_features(ds, days):
    per_day, coverage = {}, {}
    for day in days:
        grid = day_grid(ds, day)
        d = ds.telemetry / day
        frames = [load_source(d, s, grid) for s in ds.sources]
        wide = pd.concat([f for f in frames if f.shape[1]], axis=1)
        wide = wide.loc[:, ~wide.columns.duplicated()]
        per_day[day] = wide
        coverage[day] = set(wide.columns[wide.notna().any()])
        print(f"  {day}: {wide.shape[1]:>5} series, "
              f"{wide.notna().to_numpy().mean():.1%} cells observed", flush=True)

    counts = pd.Series(0, index=sorted(set().union(*coverage.values())), dtype=int)
    for s in coverage.values():
        counts[sorted(s)] += 1
    keep = counts[counts >= ds.min_day_coverage * len(days)].index.tolist()
    dropped = sorted(set(counts.index) - set(keep))

    X = pd.concat([per_day[d].reindex(columns=keep) for d in days], axis=0)
    return X.sort_index().astype(np.float32), keep, dropped


def finalize(ds, X, days):
    """Drop hopeless series, forward-fill the rest within each day.

    Sampling is ragged in both datasets (Bank: many KPIs every 2 min and a few
    hourly; Telecom: ~60 s but jittered), so the minute grid starts out sparse.
    Filling happens *within* a day, so no value leaks across a day boundary, and the
    fill limit is derived from each series' own observed cadence.
    """
    miss = X.isna().mean()
    keep = miss[miss <= ds.max_missing].index
    dropped = sorted(set(X.columns) - set(keep))
    Xk = X[keep]
    day_key = np.repeat(np.arange(len(days)), ds.day_minutes)

    limits = {}
    for c in keep:
        obs = np.flatnonzero(Xk[c].notna().to_numpy())
        gaps = np.diff(obs)
        limits[c] = max(2, ds.ffill_factor * (int(np.median(gaps)) if len(gaps) else 1))

    out = {}
    lim_s = pd.Series(limits)
    for lim, cols in lim_s.groupby(lim_s.values):
        sub = Xk[list(cols.index)]
        filled = sub.groupby(day_key, group_keys=False).apply(
            lambda g: g.ffill(limit=int(lim)).bfill(limit=int(lim)))
        for c in sub.columns:
            out[c] = filled[c]
    F = pd.DataFrame(out, index=Xk.index)[list(keep)].astype(np.float32)
    return F, dropped, pd.Series(limits, name="ffill_limit")


# ---------------------------------------------------------------- labelling --
def forward_any(flags, h, day_of):
    """1 where flags[i+1 .. i+h] contains a True, never crossing a day boundary."""
    out = np.zeros(len(flags), dtype=np.int8)
    for k in range(1, h + 1):
        shifted = np.zeros(len(flags), dtype=bool); shifted[:-k] = flags[k:]
        same_day = np.zeros(len(flags), dtype=bool); same_day[:-k] = day_of[k:] == day_of[:-k]
        out |= (shifted & same_day)
    return out


def build_labels(ds, X, days):
    grid = X.index.to_numpy()
    idx = {t: i for i, t in enumerate(grid.tolist())}
    day_of = np.repeat(np.arange(len(days)), ds.day_minutes)
    pos_in_day = np.tile(np.arange(ds.day_minutes), len(days))
    assert len(day_of) == len(grid), (len(day_of), len(grid))

    rec = pd.read_csv(ds.record_csv)
    rec["onset"] = (rec["timestamp"].astype(float).astype(np.int64) // MINUTE) * MINUTE
    rec["onset_dt_utc8"] = [datetime.fromtimestamp(t, TZ).strftime("%Y-%m-%d %H:%M:%S")
                            for t in rec["onset"]]
    rec["in_grid"] = rec["onset"].isin(idx)

    comps = sorted(rec["component"].astype(str).unique())
    pos = np.zeros(len(grid), dtype=bool)
    pos_c = pd.DataFrame(False, index=grid, columns=comps)
    for _, r in rec[rec["in_grid"]].iterrows():
        i = idx[r["onset"]]
        pos[i] = True
        pos_c.iloc[i, pos_c.columns.get_loc(str(r["component"]))] = True

    y_sys = pd.DataFrame({f"h{h}": forward_any(pos, h, day_of) for h in ds.horizons},
                         index=grid)
    cols, data = [], []
    for h in ds.horizons:
        for c in comps:
            cols.append((f"h{h}", c))
            data.append(forward_any(pos_c[c].to_numpy(), h, day_of))
    y_comp = pd.DataFrame(np.array(data).T, index=grid,
                          columns=pd.MultiIndex.from_tuples(cols, names=["horizon", "component"]))

    # "already failing" blackout: onset .. onset+BLACKOUT, clipped to the day
    in_fault = np.zeros(len(grid), dtype=bool)
    for i in np.flatnonzero(pos):
        end_of_day = (i // ds.day_minutes + 1) * ds.day_minutes - 1
        in_fault[i:min(i + ds.blackout, end_of_day) + 1] = True

    masks = pd.DataFrame({"in_fault": in_fault.astype(np.int8)}, index=grid)
    for h in ds.horizons:
        incomplete = (pos_in_day >= ds.day_minutes - h) | (pos_in_day < ds.lookback - 1)
        positive = y_sys[f"h{h}"].to_numpy().astype(bool)
        masks[f"eval_h{h}"] = (((~in_fault) | positive) & ~incomplete).astype(np.int8)
    return y_sys, y_comp, masks, rec


# --------------------------------------------------------------------- main --
def build(ds, days=None):
    days = days or sorted(p.name for p in ds.telemetry.iterdir()
                          if p.is_dir() and p.name[:2].isdigit())
    print(f"\n=== {ds.name}: {len(days)} days x {ds.day_minutes} min ===")
    ds.out.mkdir(parents=True, exist_ok=True)

    print("Building feature matrix...")
    X, keep, day_dropped = build_features(ds, days)
    print(f"  kept {len(keep)} series, dropped {len(day_dropped)} not present on every day")

    print("Filling gaps...")
    F, sparse_dropped, limits = finalize(ds, X, days)
    print(f"  dropped {len(sparse_dropped)} series over {ds.max_missing:.0%} missing; "
          f"residual missing {F.isna().to_numpy().mean():.2%}")

    print("Building labels...")
    y_sys, y_comp, masks, rec = build_labels(ds, X, days)

    X.to_parquet(ds.out / "features.parquet")
    F.to_parquet(ds.out / "features_filled.parquet")
    y_sys.to_parquet(ds.out / "labels_system.parquet")
    y_comp.to_parquet(ds.out / "labels_component.parquet")
    masks.to_parquet(ds.out / "masks.parquet")
    rec.to_csv(ds.out / "onsets.csv", index=False)

    fi = pd.DataFrame({"feature": keep})
    fi["component"] = fi["feature"].str.split("||", regex=False).str[0]
    fi["kpi"] = fi["feature"].str.split("||", regex=False).str[1]
    fi["family"] = np.where(fi["kpi"].str.contains(r"^\w+\.", regex=True),
                            fi["kpi"].str.split(".", regex=False).str[0], "")
    fi["kept_after_fill"] = ~fi["feature"].isin(sparse_dropped)
    fi["missing_raw"] = fi["feature"].map(X.isna().mean())
    fi["ffill_limit"] = fi["feature"].map(limits)
    fi.to_csv(ds.out / "feature_index.csv", index=False)

    json.dump({"dataset": ds.name, "days": days, "day_minutes": ds.day_minutes,
               "rows": int(len(X)), "features_raw": int(X.shape[1]),
               "features_filled": int(F.shape[1]), "horizons": list(ds.horizons),
               "lookback": ds.lookback, "blackout": ds.blackout,
               "max_missing": ds.max_missing, "min_day_coverage": ds.min_day_coverage},
              open(ds.out / "meta.json", "w"), indent=1)

    miss = X.isna().mean()
    lines = [
        f"# Stage 1 build report — {ds.name}", "",
        f"- days: {len(days)} ({days[0]} .. {days[-1]}), {ds.day_minutes} min/day, "
        f"rows: {len(X):,} minutes",
        f"- features kept: {len(keep)} (dropped {len(day_dropped)} not present on all days)",
        f"- components: {fi['component'].nunique()}",
        f"- overall missing cells: {X.isna().to_numpy().mean():.2%}",
        f"- features >5% missing: {int((miss > 0.05).sum())}",
        f"- faults in record.csv: {len(rec)}, snapped onto the grid: {int(rec['in_grid'].sum())}",
        f"- look-back {ds.lookback} min, blackout {ds.blackout} min",
        f"- after gap-fill: {F.shape[1]} features, residual missing {F.isna().to_numpy().mean():.2%}",
        f"- dropped as too sparse (> {ds.max_missing:.0%} missing): {len(sparse_dropped)}",
        "", "## Positive rate by horizon (system-level)", "",
        "| horizon (min) | positive rows | evaluable rows | rate |", "|---|---|---|---|",
    ]
    for h in ds.horizons:
        m = masks[f"eval_h{h}"].astype(bool)
        p = int(y_sys[f"h{h}"][m].sum())
        lines.append(f"| {h} | {p:,} | {int(m.sum()):,} | {p / max(int(m.sum()), 1):.3%} |")
    if int(rec["in_grid"].sum()) < len(rec):
        miss_rows = rec[~rec["in_grid"]]
        lines += ["", f"**{len(miss_rows)} faults fall outside the telemetry grid** "
                      f"(the day folders cover only {ds.day_minutes} min/day):", ""]
        lines += [f"- {r.onset_dt_utc8}  {r.component}  {r.reason}"
                  for r in miss_rows.itertuples()][:20]
    (ds.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[1:]))
    print(f"\nWrote outputs to {ds.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="bank | telecom | all")
    ap.add_argument("--days", nargs="*", default=None)
    args = ap.parse_args()

    names = list(DATASETS) if (args.dataset or "").lower() == "all" else [args.dataset]
    for n in names:
        build(get_dataset(n), args.days)


if __name__ == "__main__":
    sys.exit(main())
