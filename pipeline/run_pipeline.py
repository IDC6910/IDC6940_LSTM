"""
run_pipeline.py
----------------
Raw OpenRCA telemetry in, a reproducible set of cleaned/labeled/normalized 
sequences and a first baseline comparison out.

Usage:
    python run_pipeline.py                       # default horizon, metrics only
    python run_pipeline.py --horizon 30
    python run_pipeline.py --sources metrics logs
    python run_pipeline.py --run-baselines

This does NOT train the LSTM -- it stops at "pipeline produces model-ready 
arrays" and, optionally, runs the threenon-sequential baselines on those 
same arrays so you have a reference number before the LSTM exists.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src import data_loading, cleaning, resampling, labeling, sequence_builder, features

if True:
    pass  # keep baseline_models import lazy/optional below


def build_dataset(sources=("metrics",), horizon_min=config.DEFAULT_HORIZON_MIN, verbose=True):
    log = print if verbose else (lambda *a, **k: None)

    log(f"[1/6] Loading raw {config.SYSTEM} telemetry ...")
    metrics_long = data_loading.load_metrics()
    metrics_long = data_loading.standardize_timestamps(metrics_long)
    log(f"  metrics rows: {len(metrics_long)}")

    logs_long = pd.DataFrame()
    if "logs" in sources:
        logs_long = data_loading.load_logs()
        logs_long = data_loading.standardize_timestamps(logs_long)
        log(f"  log rows: {len(logs_long)}")

    if metrics_long.empty:
        raise RuntimeError(
            f"No metric telemetry found under {config.DATASET_ROOT}/{config.SYSTEM}/telemetry/. "
            "Download the OpenRCA Bank system data and place it there first "
            "(see README.md), or run this against the bundled synthetic "
            "fixture with --synthetic-test."
        )

    log("[2/6] Cleaning telemetry (duplicates, invalid values, outliers) ...")
    metrics_clean = cleaning.clean_long_telemetry(metrics_long)
    frames_long = [metrics_clean]
    if not logs_long.empty:
        logs_clean = cleaning.clean_long_telemetry(logs_long)
        frames_long.append(logs_clean)
    combined_long = pd.concat(frames_long, ignore_index=True, sort=False)

    log(f"[3/6] Building common time representation @ {config.RESAMPLE_FREQ} grid ...")
    wide = resampling.build_common_time_representation(combined_long)
    log(f"  wide grid shape: {wide.shape}")
    if wide.empty:
        raise RuntimeError("Resampled telemetry is empty -- check RESAMPLE_FREQ / date range.")

    log("[4/6] Slicing lookback windows ...")
    X_seq, window_ends = sequence_builder.build_windows(wide)
    log(f"  windows: {X_seq.shape}")
    if len(window_ends) == 0:
        raise RuntimeError("No complete windows produced -- gaps may be too large, or too little data.")

    log("[5/6] Loading failure records and labeling windows ...")
    query_df = data_loading.load_failure_records()
    onsets = labeling.extract_failure_onsets(query_df)
    log(f"  failure onsets found: {len(onsets)}")
    label_df = labeling.label_for_all_horizons(window_ends, onsets)
    label_df = label_df.set_index("window_end")

    keep_mask = label_df.loc[window_ends, "keep"].values
    X_seq = X_seq[keep_mask]
    window_ends = window_ends[keep_mask]
    label_df = label_df.loc[window_ends]
    log(f"  windows after mid-incident exclusion: {X_seq.shape[0]}")

    y = label_df[f"label_{horizon_min}min"].values
    log(f"  positive rate @ {horizon_min}min horizon: {y.mean():.4f} ({int(y.sum())}/{len(y)})")

    log("[6/6] Chronological split + normalization (fit on train only) ...")
    meta = pd.DataFrame({"window_end": window_ends, "y": y}).reset_index(drop=True)
    train_meta, val_meta, test_meta = features.chronological_split(meta)
    train_idx, val_idx, test_idx = train_meta.index.values, val_meta.index.values, test_meta.index.values

    n_features = X_seq.shape[2]
    train_flat = X_seq[train_idx].reshape(len(train_idx), -1)
    # Fit per-feature scaler on TRAIN steps only, using all timesteps of
    # the training windows stacked together (not window-level stats), then
    # apply the same per-feature mean/std to every split and every timestep.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_seq[train_idx].reshape(-1, n_features))

    def _scale(X):
        shp = X.shape
        return scaler.transform(X.reshape(-1, n_features)).reshape(shp)

    X_train, X_val, X_test = _scale(X_seq[train_idx]), _scale(X_seq[val_idx]), _scale(X_seq[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    out_dir = config.PROCESSED_DIR
    np.savez_compressed(
        out_dir / f"sequences_h{horizon_min}min.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(list(wide.columns)),
    )
    log(f"Saved model-ready arrays to {out_dir / f'sequences_h{horizon_min}min.npz'}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": list(wide.columns),
    }


def maybe_run_baselines(data: dict, horizon_min: int):
    from src import baseline_models
    X_train_flat, flat_names = sequence_builder.flatten_for_baselines(data["X_train"], data["feature_names"])
    X_val_flat, _ = sequence_builder.flatten_for_baselines(data["X_val"], data["feature_names"])
    X_test_flat, _ = sequence_builder.flatten_for_baselines(data["X_test"], data["feature_names"])

    results = baseline_models.train_and_evaluate(
        X_train_flat, data["y_train"], X_val_flat, data["y_val"], X_test_flat, data["y_test"]
    )
    df = baseline_models.results_to_dataframe(results)
    print(f"\nBaseline results @ {horizon_min}-minute horizon:\n")
    print(df.to_string())
    df.to_csv(config.PROCESSED_DIR / f"baseline_results_h{horizon_min}min.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_MIN)
    parser.add_argument("--sources", nargs="+", default=["metrics"],
                         choices=["metrics", "logs"],
                         help="traces support is stubbed in config.COLUMN_MAP but not "
                              "yet wired into build_dataset -- see README 'Extending to traces'")
    parser.add_argument("--run-baselines", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(sources=args.sources, horizon_min=args.horizon)
    if args.run_baselines:
        maybe_run_baselines(dataset, args.horizon)
