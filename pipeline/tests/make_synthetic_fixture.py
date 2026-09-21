"""
make_synthetic_fixture.py
--------------------------
Generates a tiny synthetic dataset that mimics OpenRCA's Bank-system
directory layout and (assumed) column schema, purely so the pipeline can
be exercised end-to-end before you've downloaded the real 26GB dataset.
This is a smoke test, NOT a substitute for validating against real data --
rerun src/inspect_schema.py against the real files and fix config.COLUMN_MAP
as soon as you have them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

RNG = np.random.default_rng(0)


def _to_epoch_seconds(dt_index) -> np.ndarray:
    """Unit-agnostic conversion (avoids relying on int64-cast resolution,
    which has changed across pandas versions). Treats the naive input as
    Asia/Shanghai wall-clock time before converting to a true UTC epoch --
    matching the real OpenRCA convention (epoch timestamps are absolute
    UTC instants; query.csv's onset strings are naive UTC+8 wall time) so
    this synthetic fixture exercises the same tz-alignment logic the real
    pipeline depends on."""
    idx = pd.DatetimeIndex(dt_index)
    if idx.tz is None:
        idx = idx.tz_localize(config.RAW_TIMEZONE)
    idx_utc = idx.tz_convert("UTC")
    return ((idx_utc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy()


def make_fixture(system: str = "Bank", date: str = "2021_03_04", n_pods: int = 3,
                  n_hours: int = 6, failure_every_min: int = 90):
    base = config.DATASET_ROOT / system
    telemetry_dir = base / "telemetry" / date
    (telemetry_dir / "metric").mkdir(parents=True, exist_ok=True)
    (telemetry_dir / "log").mkdir(parents=True, exist_ok=True)
    (telemetry_dir / "trace").mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2021-03-04 00:00:00")
    n_seconds = n_hours * 3600
    ts = pd.date_range(start, periods=n_seconds, freq="1s")

    failure_onsets = pd.date_range(start + pd.Timedelta(minutes=30),
                                    start + pd.Timedelta(hours=n_hours),
                                    freq=f"{failure_every_min}min")

    kpis = ["cpu_util", "mem_util", "net_send", "jvm_heap"]
    rows = []
    for pod in [f"pod-{i}" for i in range(n_pods)]:
        for kpi in kpis:
            base_level = RNG.uniform(20, 50)
            noise = RNG.normal(0, 2, size=n_seconds)
            series = base_level + noise
            # inject a ramp-up preceding each failure onset (the "signal"
            # a real model would need to learn)
            for onset in failure_onsets:
                lead = pd.date_range(onset - pd.Timedelta(minutes=15), onset, freq="1s")
                mask = ts.isin(lead)
                ramp = np.linspace(0, 40, mask.sum())
                series[mask] += ramp
            # randomly drop ~2% of samples to simulate irregular sampling
            drop_mask = RNG.random(n_seconds) < 0.02
            sample_ts = ts[~drop_mask]
            sample_vals = series[~drop_mask]
            # inject a few duplicate timestamps
            dup_idx = RNG.choice(len(sample_ts), size=5, replace=False)
            dup_df = pd.DataFrame({
                "timestamp": _to_epoch_seconds(sample_ts[dup_idx]),
                "cmdb_id": pod, "kpi_name": kpi,
                "value": sample_vals[dup_idx] + RNG.normal(0, 0.5, 5),
            })
            df = pd.DataFrame({
                "timestamp": _to_epoch_seconds(sample_ts),  # epoch seconds
                "cmdb_id": pod,
                "kpi_name": kpi,
                "value": sample_vals,
            })
            df = pd.concat([df, dup_df], ignore_index=True)
            # inject a few extreme outliers
            out_idx = RNG.choice(len(df), size=3, replace=False)
            df.loc[out_idx, "value"] = df.loc[out_idx, "value"] * 50
            rows.append(df)

    metrics_df = pd.concat(rows, ignore_index=True)
    # split across a couple of files like the real export would
    metrics_df.iloc[: len(metrics_df) // 2].to_csv(telemetry_dir / "metric" / "part-0.csv", index=False)
    metrics_df.iloc[len(metrics_df) // 2:].to_csv(telemetry_dir / "metric" / "part-1.csv", index=False)

    # minimal log fixture
    log_rows = []
    for pod in [f"pod-{i}" for i in range(n_pods)]:
        n = 500
        log_ts = RNG.choice(_to_epoch_seconds(ts), size=n, replace=False)
        log_rows.append(pd.DataFrame({
            "timestamp": log_ts, "cmdb_id": pod, "log_name": "error_count",
            "value": RNG.poisson(1, n),
        }))
    pd.concat(log_rows, ignore_index=True).to_csv(telemetry_dir / "log" / "part-0.csv", index=False)

    # query.csv with the documented "prediction" field format
    query_rows = []
    for i, onset in enumerate(failure_onsets):
        pred = f"[{onset.strftime('%Y-%m-%d %H:%M:%S')}] pod-0 cpu_fault"
        query_rows.append({"case_id": i, "prediction": pred})
    pd.DataFrame(query_rows).to_csv(base / "query.csv", index=False)

    print(f"Synthetic fixture written under {base}")
    print(f"  {len(metrics_df)} metric rows, {len(failure_onsets)} synthetic failures")


if __name__ == "__main__":
    make_fixture()
