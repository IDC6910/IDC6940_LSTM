"""
config.py
---------
Single place to configure the pipeline. Change these values instead of
hunting through the rest of the codebase.

NOTE ON SCHEMAS: OpenRCA's raw telemetry (Bank system) originates from the
AIOps Challenge series and is typically distributed as long/"melted" CSVs:
    metric files -> columns like: timestamp, cmdb_id, kpi_name, value
    log files    -> columns like: timestamp, cmdb_id, log_name, value
    trace files  -> columns like: timestamp, cmdb_id, span_id, parent_id,
                                   service_name, start_time, end_time/duration

This has NOT been verified against our actual downloaded files (schemas
have drifted across AIOps challenge years).

Run `src/inspect_schema.py` against real `dataset/Bank/telemetry/<DATE>/metric/*.csv` 
files first and update COLUMN_MAP below if the real column names differ. 
Everything downstream reads through COLUMN_MAP, so that's the only place 
to edit.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"          # dataset/{SYSTEM}/...
PROCESSED_DIR = PROJECT_ROOT / "processed"        # cleaned/labeled outputs
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"          # scalers, encoders, etc.

for d in (PROCESSED_DIR, ARTIFACT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
SYSTEM = "Bank"          # "Telecom" | "Bank" | "Market" -- expand later
DATE_DIRS = None          # None = auto-discover all YYYY_MM_DD folders under
                           # dataset/{SYSTEM}/telemetry/. Or pass an explicit
                           # list like ["2021_03_04", "2021_03_05"] to limit
                           # scope while you're getting the pipeline working.

# ---------------------------------------------------------------------------
# Raw column names -> canonical names. Edit the LEFT side after inspecting
#  real files; the RIGHT side (canonical) is what the rest of the code
# uses everywhere, so nothing else needs to change.
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "metric": {
        "timestamp": "timestamp",
        "entity": "cmdb_id",     # pod / component id
        "name": "kpi_name",      # which metric (cpu_util, mem_util, ...)
        "value": "value",
    },
    "log": {
        "timestamp": "timestamp",
        "entity": "cmdb_id",
        "name": "log_name",
        "value": "value",        # message / template id / log level count
    },
    "trace": {
        "timestamp": "timestamp",
        "entity": "cmdb_id",
        "span_id": "span_id",
        "parent_id": "parent_id",
        "duration": "duration",   # microseconds; derive from start/end if absent
        "status": "status_code",
    },
}

# Raw OpenRCA timestamps are UTC+8 (Asia/Shanghai). If extracted files
# are already tz-naive local time (typical), leave this as-is; we treat them
# as UTC+8 and convert to a consistent UTC index internally.
RAW_TIMEZONE = "Asia/Shanghai"

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
RESAMPLE_FREQ = "60s"     # 1-minute grid. Tighten to "5S"/"10S" if you have
                           # dense enough metrics and the compute budget.

# ---------------------------------------------------------------------------
# Cleaning thresholds
# ---------------------------------------------------------------------------
MAX_GAP_TO_INTERPOLATE = 5     # consecutive missing samples we'll fill via
                                 # linear interpolation; longer gaps -> ffill
                                 # capped, then flagged, never fabricated far out
OUTLIER_Z_THRESH = 6.0          # winsorize points beyond this many robust
                                 # z-scores (computed with median/MAD)

# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
PREDICTION_HORIZONS_MIN = [5, 10, 15, 30, 60]   # Experiment 2 grid
DEFAULT_HORIZON_MIN = 15                          # used when a single horizon
                                                    # is needed (e.g. Exp 1, 3)
LOOKBACK_WINDOW_MIN = 20         # length of the input sequence fed to models
                                   # (matches the "[10:00-10:20]" example)
STRIDE_MIN = 1                    # step between consecutive windows when
                                   # slicing sequences (1 = maximally overlap)

# A telemetry window is labeled 1 if a failure's ground-truth onset falls in
# (window_end, window_end + horizon]. Windows that fall inside another
# failure's *own* pre-failure buffer are still positive; windows that occur
# during active failure remediation (post-onset, before "resolved") are
# excluded entirely by default -- see EXCLUDE_POST_ONSET_MIN below.
EXCLUDE_POST_ONSET_MIN = 30       # drop windows whose end falls within this
                                    # many minutes AFTER a failure onset, so
                                    # the model isn't trained on "failure is
                                    # already happening" telemetry as if it
                                    # were a clean negative.

# ---------------------------------------------------------------------------
# Train / val / test split -- chronological, never random
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

RANDOM_SEED = 42
