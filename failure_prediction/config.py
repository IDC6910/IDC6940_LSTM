"""Dataset registry + shared configuration for the failure-prediction pipeline.

Switch dataset with the OPENRCA_DATASET environment variable, or by passing
--dataset to build_dataset.py:

    OPENRCA_DATASET=telecom jupyter lab
    python3 build_dataset.py --dataset telecom

TIMEZONE CONTRACT (read this before touching anything):
  Every file in OpenRCA is expressed in UTC+8 -- `record.csv.timestamp`, the metric
  timestamps, and the `2021_03_04/` style day-folder names alike.  We therefore:
    * join and window ONLY on raw epoch seconds (offset-free), and
    * render human-readable datetimes with TZ below and nothing else.
  Never call pandas' tz-naive `to_datetime(...)` and read it as local time.

ADDING A DATASET
  Append an entry to DATASETS below.  A `Source` describes one telemetry file:
    kind="long"  -- one row per (timestamp, component, kpi): give id/kpi/val columns
    kind="wide"  -- one row per (timestamp, entity) with several value columns
  Nothing else in the pipeline is dataset-specific.  (Market is not registered yet:
  it splits into cloudbed-1/cloudbed-2 sub-systems and needs its own entry per bed.)
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import timezone, timedelta

TZ = timezone(timedelta(hours=8))          # UTC+8 -- the dataset's clock
MINUTE = 60

ROOT = Path(__file__).resolve().parent.parent
# Dataset root: defaults to <parent of this folder>/dataset, so keeping this folder
# inside an OpenRCA checkout needs no configuration.  Point OPENRCA_DATA at the
# dataset directory instead when the code lives somewhere else (e.g. a course repo).
DATA = Path(os.environ.get("OPENRCA_DATA", ROOT / "dataset")).expanduser()
OUT_ROOT = Path(__file__).resolve().parent / "build"


@dataclass
class Source:
    """One telemetry file, and how to read it into (minute, component, kpi, value)."""
    path: str                       # relative to the day folder
    kind: str                       # "long" | "wide"
    ts_col: str
    ts_unit: str = "s"              # "s" or "ms"
    id_col: str = None              # long: component column; wide: entity column
    kpi_col: str = None             # long only
    val_col: str = None             # long only
    value_cols: tuple = ()          # wide only
    component: str = None           # wide: fixed component name for every column
    family: str = ""                # label for the modality (container/node/app/...)
    prefix_kpi: bool = False        # prepend "<family>." to the KPI name

    def required_cols(self):
        if self.kind == "long":
            return [self.ts_col, self.id_col, self.kpi_col, self.val_col]
        return [self.ts_col, self.id_col, *self.value_cols]


@dataclass
class Dataset:
    name: str
    dirname: str                    # folder under dataset/
    day_minutes: int                # minutes of telemetry per day folder
    sources: tuple
    day_offset_min: int = 0         # minutes after UTC+8 midnight the day starts
    lookback: int = 60              # look-back window fed to the sequence models
    blackout: int = 10              # minutes after onset treated as "already failing"
    horizons: tuple = (1, 3, 5, 10, 15, 30)
    min_day_coverage: float = 1.0   # keep a series only if seen on >= this share of days
    max_missing: float = 0.80       # drop series missing more than this after alignment
    ffill_factor: int = 2           # forward-fill limit = factor x the series' own cadence

    @property
    def dir(self):      return DATA / self.dirname
    @property
    def record_csv(self): return self.dir / "record.csv"
    @property
    def telemetry(self):  return self.dir / "telemetry"
    @property
    def out(self):        return OUT_ROOT / self.name


BANK = Dataset(
    name="bank", dirname="Bank", day_minutes=1440,
    sources=(
        Source("metric/metric_container.csv", "long", ts_col="timestamp", ts_unit="s",
               id_col="cmdb_id", kpi_col="kpi_name", val_col="value", family="container"),
        Source("metric/metric_app.csv", "wide", ts_col="timestamp", ts_unit="s",
               id_col="tc", value_cols=("rr", "sr", "cnt", "mrt"),
               component="app", family="app"),
    ),
)

# Telecom: millisecond timestamps, four metric families whose KPI names can collide
# across families (hence prefix_kpi), and day folders covering only 00:00-05:59 UTC+8.
TELECOM = Dataset(
    name="telecom", dirname="Telecom", day_minutes=360,
    sources=tuple(
        Source(f"metric/metric_{fam}.csv", "long", ts_col="timestamp", ts_unit="ms",
               id_col="cmdb_id", kpi_col="name", val_col="value",
               family=fam, prefix_kpi=True)
        for fam in ("node", "container", "middleware", "service")
    ) + (
        Source("metric/metric_app.csv", "wide", ts_col="startTime", ts_unit="ms",
               id_col="serviceName",
               value_cols=("avg_time", "num", "succee_num", "succee_rate"),
               component="app", family="app"),
    ),
)

DATASETS = {d.name: d for d in (BANK, TELECOM)}

def get_dataset(name=None):
    name = (name or os.environ.get("OPENRCA_DATASET", "bank")).lower()
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]

# --- back-compat aliases for the single-dataset scripts ---------------------
DATASET_NAME = os.environ.get("OPENRCA_DATASET", "bank")
DS = get_dataset(DATASET_NAME)
HORIZONS = list(DS.horizons)
LOOKBACK = DS.lookback
BLACKOUT = DS.blackout
DAY_MINUTES = DS.day_minutes
