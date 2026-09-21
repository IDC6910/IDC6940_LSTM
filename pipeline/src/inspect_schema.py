"""
inspect_schema.py
------------------
Run FIRST against real downloaded data, before trusting anything
else in the pipeline. It prints the columns, dtypes, and a few sample rows
for one file of each telemetry type, plus record.csv / query.csv. 
Use to fix config.COLUMN_MAP if it doesn't match.

Usage:
    python -m src.inspect_schema
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _peek_csv(path: Path, n: int = 5):
    print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
    try:
        df = pd.read_csv(path, nrows=200)
    except Exception as e:
        print(f"  !! failed to read: {e}")
        return
    print(f"shape (first 200 rows read): {df.shape}")
    print("dtypes:")
    print(df.dtypes)
    print("\nsample rows:")
    print(df.head(n).to_string())
    for col in df.columns:
        nun = df[col].nunique()
        if nun <= 15:
            print(f"  unique values in '{col}' (<=15): {sorted(df[col].dropna().unique().tolist())}")


def main():
    system_dir = config.DATASET_ROOT / config.SYSTEM
    if not system_dir.exists():
        print(f"Dataset directory not found: {system_dir}")
        print("Download the OpenRCA Bank telemetry and place it there first.")
        return

    for meta_file in ("query.csv", "record.csv"):
        p = system_dir / meta_file
        if p.exists():
            _peek_csv(p)
        else:
            print(f"\n(missing: {p})")

    telemetry_dir = system_dir / "telemetry"
    if not telemetry_dir.exists():
        print(f"\n(missing telemetry dir: {telemetry_dir})")
        return

    date_dirs = sorted(d for d in telemetry_dir.iterdir() if d.is_dir())
    if not date_dirs:
        print("\nNo date directories found under telemetry/.")
        return

    first_date = date_dirs[0]
    print(f"\nInspecting one sample date: {first_date.name}")
    for kind in ("metric", "log", "trace"):
        kind_dir = first_date / kind
        if not kind_dir.exists():
            print(f"\n(no '{kind}' subfolder for {first_date.name})")
            continue
        files = sorted(kind_dir.glob("*.csv"))
        if not files:
            print(f"\n(no CSV files in {kind_dir})")
            continue
        _peek_csv(files[0])
        if len(files) > 1:
            print(f"  ... and {len(files) - 1} more {kind} file(s) in this date folder")


if __name__ == "__main__":
    main()
