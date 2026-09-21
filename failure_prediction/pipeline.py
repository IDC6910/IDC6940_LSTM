"""Shared core for the failure-prediction notebooks (Stages 1-3), dataset-aware.

Both notebooks do:

    from pipeline import *
    A = load_into(globals(), "bank")        # or "telecom" -- re-run the cell to switch

`load_into` loads (building first if needed) that dataset's Stage 1 outputs and
injects the familiar names -- F, GRID, DAYS, ONSETS, MASKS, ... -- into the notebook
namespace.  Datasets are cached, so switching back and forth is instant.

TIMEZONE CONTRACT: every timestamp in OpenRCA is UTC+8.  We join and window only on
raw epoch seconds, render time with TZ and nothing else, and derive day membership
from the UTC+8 date -- never from `epoch // 86400`.
"""
import os, sys, math, time, json, warnings, subprocess
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import TZ, MINUTE, get_dataset, DATASETS

warnings.filterwarnings("ignore")

SEED = 20260916
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent

PRESETS = {
    "smoke":    dict(horizons=[5],              splits=["chrono"],      epochs=4,
                     top_k=48,  neg_ratio=3, max_train=4000,  ablations=False, boot=200),
    "standard": dict(horizons=[1,3,5,10,15,30], splits=["chrono"],      epochs=15,
                     top_k=128, neg_ratio=5, max_train=12000, ablations=True,  boot=1000),
    "full":     dict(horizons=[1,3,5,10,15,30], splits=["chrono","cv"], epochs=25,
                     top_k=192, neg_ratio=8, max_train=16000, ablations=True,  boot=1000),
}
PRESET = os.environ.get("OPENRCA_PRESET", "standard")
CFG = dict(PRESETS[PRESET])

# ----------------------------------------------------------- optional deps --
def _has(name):
    try:
        __import__(name); return True
    except Exception:
        return False

HAS_SK, HAS_XGB, HAS_TORCH = _has("sklearn"), _has("xgboost"), _has("torch")
if HAS_TORCH:
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

# ------------------------------------------------------------- plot style ---
# Okabe-Ito: a fixed, colourblind-safe categorical order. Assigned in order,
# never cycled; colour follows the entity, never its rank.
PAL   = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A7A7A"]
INK, MUTED = "#1a1a1a", "#6b6b6b"
DATASET_COLOR = {"bank": "#0072B2", "telecom": "#D55E00"}

def apply_plot_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "#cfcfcf", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9,
        "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "font.size": 9,
    })

# ============================================================== STAGE 1 =====
NEEDED = ["features_filled.parquet", "labels_system.parquet",
          "labels_component.parquet", "masks.parquet", "onsets.csv"]

_CACHE = {}
_ACTIVE = None

def _load(name):
    ds = get_dataset(name)
    if not all((ds.out / f).exists() for f in NEEDED):
        print(f"build/{ds.name} incomplete -- running build_dataset.py --dataset {ds.name} ...")
        subprocess.run([sys.executable, "build_dataset.py", "--dataset", ds.name],
                       cwd=HERE, check=True)

    A = SimpleNamespace()
    A.DS, A.DATASET = ds, ds.name
    A.BUILD   = ds.out
    A.RESULTS = HERE / "results" / ds.name; A.RESULTS.mkdir(parents=True, exist_ok=True)
    A.FIGS    = A.RESULTS / "figures";      A.FIGS.mkdir(parents=True, exist_ok=True)

    A.F      = pd.read_parquet(ds.out / "features_filled.parquet")
    A.F_RAW  = pd.read_parquet(ds.out / "features.parquet")
    A.Y_SYS  = pd.read_parquet(ds.out / "labels_system.parquet")
    A.Y_COMP = pd.read_parquet(ds.out / "labels_component.parquet")
    A.MASKS  = pd.read_parquet(ds.out / "masks.parquet")
    A.ONSETS = pd.read_csv(ds.out / "onsets.csv")
    A.FEATIX = pd.read_csv(ds.out / "feature_index.csv")
    A.META   = json.loads((ds.out / "meta.json").read_text())

    A.DAY_MINUTES = ds.day_minutes
    A.LOOKBACK    = ds.lookback
    A.HORIZONS    = list(ds.horizons)
    A.GRID        = A.F.index.to_numpy()
    A.DAY_DATE    = np.array([datetime.fromtimestamp(int(t), TZ).strftime("%Y-%m-%d")
                              for t in A.GRID])
    A.DAYS        = list(dict.fromkeys(A.DAY_DATE.tolist()))
    A.POS_IN_DAY  = np.tile(np.arange(A.DAY_MINUTES), len(A.DAYS))
    A.ROW_OF      = {int(t): i for i, t in enumerate(A.GRID.tolist())}
    A.FEATS       = list(A.F.columns)
    A.COMPONENT_OF = pd.Series(A.FEATS).str.split("||", regex=False).str[0].to_numpy()
    A.FVAL        = A.F.fillna(0.0).to_numpy(dtype=np.float32)

    # --- integrity assertions: this is the timezone regression test ---------
    assert len(A.GRID) == len(A.DAYS) * A.DAY_MINUTES
    assert set(np.diff(A.GRID)[:A.DAY_MINUTES - 1].tolist()) == {60}
    assert (A.ONSETS["datetime"] == A.ONSETS["onset_dt_utc8"]).all(), \
        f"{ds.name}: UTC+8 rendering disagrees with record.csv -- do NOT proceed"
    return A

def use_dataset(name=None):
    """Load (and cache) a dataset and make it the active one."""
    global _ACTIVE
    name = (name or os.environ.get("OPENRCA_DATASET", "bank")).lower()
    if name not in _CACHE:
        _CACHE[name] = _load(name)
    _ACTIVE = _CACHE[name]
    return _ACTIVE

def active():
    return _ACTIVE if _ACTIVE is not None else use_dataset()

def load_into(ns, name=None):
    """Load a dataset and inject its names (F, GRID, DAYS, ONSETS, ...) into `ns`.

    Re-run the cell with a different name to switch datasets -- no kernel restart.
    """
    A = use_dataset(name)
    ns.update({k: v for k, v in vars(A).items()})
    return A

# ============================================================== STAGE 2 =====
@dataclass
class Fold:
    """A day-grouped split.  Never split by row: adjacent minutes are duplicates."""
    name:  str
    train: list
    val:   list
    test:  list
    dataset: str = None
    def _A(self):
        return _CACHE[self.dataset] if self.dataset else active()
    def rows(self, which):
        A = self._A()
        return np.flatnonzero(np.isin(A.DAY_DATE, list(getattr(self, which))))
    def n_events(self):
        A = self._A()
        return int(A.ONSETS["datetime"].str[:10].isin(self.test).sum())
    def __repr__(self):
        return (f"Fold({self.name}: train={len(self.train)}d val={len(self.val)}d "
                f"test={[d[5:] for d in self.test]}, {self.n_events()} events)")

def chrono_folds(A=None):
    """Deployment-realistic: train on the early block, test on the latest ~30 % of days."""
    A = A or active()
    days = A.DAYS
    n_test = max(2, round(0.30 * len(days)))
    train, val, test = days[:-n_test - 1], days[-n_test - 1:-n_test], days[-n_test:]
    return [Fold("chrono", train, val, test, A.DATASET)]

def cv_folds(block=2, A=None):
    """Leave-`block`-days-out, so every day is tested exactly once."""
    A = A or active()
    days, out = A.DAYS, []
    for k in range(0, len(days), block):
        test = days[k:k + block]
        rest = [d for d in days if d not in test]
        out.append(Fold(f"cv{k // block + 1}", rest[:-1], rest[-1:], test, A.DATASET))
    return out

def build_folds(splits=None, A=None):
    A = A or active()
    splits = splits or CFG["splits"]
    folds = []
    if "chrono" in splits: folds += chrono_folds(A)
    if "cv"     in splits: folds += cv_folds(2, A)
    return folds

# ============================================================== STAGE 3 =====
def robust_fit(rows, A=None):
    """median / IQR scaler -- fitted on TRAIN rows only.  KPIs are heavy-tailed."""
    A = A or active()
    M = A.FVAL[rows]
    med = np.nanmedian(M, axis=0)
    q1, q3 = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
    iqr = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return med.astype(np.float32), iqr.astype(np.float32)

def select_topk(rows, y, k, restrict=None, A=None):
    """ANOVA F between each feature and the label -- TRAIN rows only."""
    from sklearn.feature_selection import f_classif
    A = A or active()
    cand = np.arange(A.FVAL.shape[1]) if restrict is None else np.asarray(restrict)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, _ = f_classif(A.FVAL[np.ix_(rows, cand)], y)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0)
    return cand[np.argsort(-f)[:min(k, len(cand))]]

def make_windows(centers, cols, stats, L=None, A=None):
    """(n, L, |cols|) float32.  Window t covers [t-L+1, t], scaled with `stats`.

    Stage 1's eval masks guarantee the whole window sits inside one UTC+8 day, so a
    window can never straddle a gap between day folders.
    """
    A = A or active()
    L = L or A.LOOKBACK
    centers = np.asarray(centers)
    idx = centers[:, None] + np.arange(-L + 1, 1)[None, :]
    W = A.FVAL[np.ix_(idx.ravel(), cols)].reshape(len(centers), L, len(cols))
    med, iqr = stats[0][cols], stats[1][cols]
    return np.clip((W - med) / iqr, -10, 10).astype(np.float32)

def window_stats(W):
    """Flatten a window to 6 summary stats per feature, for the tabular models."""
    L = W.shape[1]
    tc = np.arange(L, dtype=np.float32); tc -= tc.mean()
    slope = np.einsum("t,ntf->nf", tc, W) / float((tc ** 2).sum())
    return np.concatenate([W.mean(1), W.std(1), W.min(1), W.max(1), W[:, -1, :], slope],
                          axis=1).astype(np.float32)

def sample_rows(rows, y, neg_ratio, cap, rng):
    """Keep every positive, subsample negatives.  TRAINING ONLY -- test sees all."""
    pos, neg = rows[y == 1], rows[y == 0]
    n_neg = min(len(neg), max(int(len(pos) * neg_ratio), 500))
    out = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    if len(out) > cap:
        out = rng.choice(out, size=cap, replace=False)
    return np.sort(out)

def fold_data(fold, h, k=None, lookback=None, restrict=None, neg_ratio=None,
              max_train=None, A=None):
    """Everything a model needs for one (fold, horizon).

    Scaler and feature selection are refit inside the fold on training rows only.
    """
    A = A or (_CACHE[fold.dataset] if fold.dataset else active())
    k = k or CFG["top_k"]
    lookback = lookback or A.LOOKBACK
    neg_ratio = CFG["neg_ratio"] if neg_ratio is None else neg_ratio
    max_train = CFG["max_train"] if max_train is None else max_train
    rng = np.random.default_rng(SEED)
    y_all  = A.Y_SYS[f"h{h}"].to_numpy()
    usable = A.MASKS[f"eval_h{h}"].to_numpy().astype(bool)

    parts = {w: (lambda r: r[usable[r]])(fold.rows(w)) for w in ("train", "val", "test")}
    stats = robust_fit(parts["train"], A)
    cols  = select_topk(parts["train"], y_all[parts["train"]], k, restrict, A)
    tr    = sample_rows(parts["train"], y_all[parts["train"]], neg_ratio, max_train, rng)

    d = {"cols": cols, "stats": stats, "h": h, "fold": fold, "lookback": lookback,
         "dataset": A.DATASET}
    for name, rows in (("train", tr), ("val", parts["val"]), ("test", parts["test"])):
        d[f"rows_{name}"] = rows
        d[f"X_{name}"]    = make_windows(rows, cols, stats, lookback, A)
        d[f"y_{name}"]    = y_all[rows].astype(np.float32)
    return d

def summary(A=None):
    A = A or active()
    return "\n".join([
        f"dataset   {A.DATASET.upper()}  ({A.DS.dirname}/)",
        f"preset    {PRESET}  horizons={CFG['horizons']} splits={CFG['splits']} "
        f"epochs={CFG['epochs']} top_k={CFG['top_k']}",
        f"grid      {len(A.DAYS)} days x {A.DAY_MINUTES} min = {A.F.shape[0]:,} minutes",
        f"features  {A.F.shape[1]:,} series over "
        f"{pd.unique(A.COMPONENT_OF).size} components "
        f"({A.F.isna().to_numpy().mean():.2%} still missing)",
        f"days      {', '.join(A.DAYS)}",
        f"faults    {len(A.ONSETS)} across {A.ONSETS['component'].nunique()} components, "
        f"{A.ONSETS['reason'].nunique()} reasons",
        f"deps      sklearn={HAS_SK} xgboost={HAS_XGB} torch={HAS_TORCH}",
        "timezone + grid assertions passed",
    ])

def dataset_overview():
    """One row per registered dataset -- the side-by-side the notebooks open with.

    Loads every dataset (cached) and restores whichever one was active before.
    """
    global _ACTIVE
    keep, rows = _ACTIVE, []
    for name in DATASETS:
        A = use_dataset(name)
        rows.append(dict(
            dataset=name, days=len(A.DAYS), min_per_day=A.DAY_MINUTES,
            minutes=A.F.shape[0], features=A.F.shape[1],
            components=int(pd.unique(A.COMPONENT_OF).size),
            faults=len(A.ONSETS), reasons=A.ONSETS["reason"].nunique(),
            faults_per_1k_min=round(1000 * len(A.ONSETS) / A.F.shape[0], 2),
            residual_missing=round(float(A.F.isna().to_numpy().mean()), 4)))
    _ACTIVE = keep if keep is not None else _ACTIVE
    return pd.DataFrame(rows).set_index("dataset")

def positive_rates(name):
    """System-level positive rate per horizon for one dataset (used in comparisons)."""
    A = _CACHE.get(name) or _load(name)
    _CACHE[name] = A
    return pd.Series({h: float(A.Y_SYS[f"h{h}"][A.MASKS[f"eval_h{h}"].astype(bool)].mean())
                      for h in A.HORIZONS}, name=name)
