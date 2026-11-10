#!/usr/bin/env python3
"""Train the LSTM on an OpenRCA system and export the CSVs the report reads.

    python code/run_lstm.py --system bank                    # chronological, level inputs
    python code/run_lstm.py --system telecom --split cv      # leave-two-days-out folds
    python code/run_lstm.py --system bank --inputs change    # window-relative inputs
    python code/run_lstm.py --system bank --model logreg     # logistic regression baseline

--inputs level   features scaled by the training median / IQR (the default in Methods)
--inputs change  each window minus its own median, and features ranked on
                 within-day variation, so slow day-to-day drift is removed

Uses the data pipeline in OpenRCA/failure_prediction (set FAILURE_PREDICTION_DIR if it
lives elsewhere) and writes to results/<system>/<split>-<inputs>/:

  metrics.csv      one row per horizon: PR-AUC, ROC-AUC, precision/recall/F1, events
  history.csv      training loss and validation PR-AUC per epoch
  predictions.csv  test-set probability for every usable minute
  events.csv       one row per test fault: detected or not, lead time
  split.csv        days and class counts per split

Metrics are computed with numpy so the script does not need scipy/scikit-learn.
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PIPE = Path(os.environ.get("FAILURE_PREDICTION_DIR",
                           Path.home() / "OpenRCA" / "failure_prediction"))
sys.path.insert(0, str(PIPE))
import pipeline as P                                   # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

HORIZONS = [1, 3, 5, 10, 15, 30]
ALARM_SHARE = 30 / 1440    # threshold spends 30 alarm-minutes per 24 h of telemetry
EPOCHS, PATIENCE, BATCH, LR = 15, 4, 128, 1e-3


# ------------------------------------------------------------------ metrics --
def anova_f(X, y):
    """One-way ANOVA F per column (two classes); same as sklearn's f_classif."""
    X = X.astype(np.float64); y = y.astype(bool)
    m = X.mean(0)
    ss_b = sum(g.sum() * (X[g].mean(0) - m) ** 2 for g in (y, ~y))
    ss_w = sum(((X[g] - X[g].mean(0)) ** 2).sum(0) for g in (y, ~y))
    with np.errstate(divide="ignore", invalid="ignore"):
        return ss_b / (ss_w / (len(y) - 2))


WITHIN_DAY = False         # rank features on day-demeaned values (--inputs change)


def select_topk(rows, y, k, restrict=None, A=None):
    A = A or P.active()
    cand = np.arange(A.FVAL.shape[1]) if restrict is None else np.asarray(restrict)
    X = A.FVAL[np.ix_(rows, cand)].astype(np.float64)
    if WITHIN_DAY:
        day = A.DAY_DATE[rows]
        for d in np.unique(day):
            X[day == d] -= X[day == d].mean(0)
    f = np.nan_to_num(anova_f(X, y), nan=0.0, posinf=0.0)
    return cand[np.argsort(-f)[:min(k, len(cand))]]


P.select_topk = select_topk                            # pipeline's version needs sklearn


def average_precision(y, s):
    """Area under the precision-recall curve (step-wise, as in sklearn)."""
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]   # end of each tied block
    tp = np.cumsum(y)[last]; fp = (last + 1) - tp
    prec, rec = tp / (tp + fp), tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def roc_auc(y, s):
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_bootstrap_ap(y, s, n_boot=300, block=60, seed=P.SEED):
    """95 % CI for PR-AUC, resampling 60-minute blocks (minutes are autocorrelated)."""
    if y.sum() in (0, len(y)):
        return np.array([np.nan, np.nan])
    rng = np.random.default_rng(seed)
    starts = np.arange(0, max(len(y) - block, 1))
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(p, min(p + block, len(y)))
                              for p in rng.choice(starts, max(len(y) // block, 1))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(average_precision(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


# -------------------------------------------------------------------- model --
class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden=64, layers=1, p=0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                              # x: (batch, L, k)
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def predict(model, X, bs=512):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(X[i:i + bs]))).numpy()
                               for i in range(0, len(X), bs)])


def train(d, h, hidden=64, seed=P.SEED):
    torch.manual_seed(seed)
    Xtr, ytr = d["X_train"], d["y_train"]
    model = LSTMNet(Xtr.shape[2], hidden=hidden)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=BATCH, shuffle=True)
    best, best_state, bad, hist = -1.0, None, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        val_ap = np.nan_to_num(average_precision(d["y_val"], predict(model, d["X_val"])))
        hist.append(dict(horizon=h, epoch=epoch, train_loss=np.mean(losses), val_pr_auc=val_ap))
        if val_ap > best:
            best, bad = val_ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    for r in hist:
        r["best_epoch"] = max(hist, key=lambda r: r["val_pr_auc"])["epoch"]
    return model, hist


# --------------------------------------------------------- logistic baseline --
C_GRID = [0.001, 0.01, 0.1, 1.0]


def fit_logreg(X, y, C, iters=200):
    """L2 logistic regression with balanced class weights, fitted with L-BFGS.

    Minimises 0.5 * ||w||^2 + C * sum_i s_i * logloss_i (intercept not penalised),
    the same objective as sklearn's LogisticRegression(class_weight="balanced").
    """
    X = torch.from_numpy(X.astype(np.float64)); y = torch.from_numpy(y.astype(np.float64))
    n_pos = y.sum(); n_neg = len(y) - n_pos
    s = torch.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, tolerance_grad=1e-7,
                            tolerance_change=1e-10, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = 0.5 * (w @ w) + C * (s * nn.functional.binary_cross_entropy_with_logits(
            z, y, reduction="none")).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def train_logreg(d):
    """Window summaries -> logistic regression; C chosen on validation PR-AUC."""
    Xtr, Xva, Xte = (P.window_stats(d[f"X_{p}"]) for p in ("train", "val", "test"))
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Xtr, Xva, Xte = ((X - mu) / sd for X in (Xtr, Xva, Xte))
    best = None
    for C in C_GRID:
        w, b = fit_logreg(Xtr, d["y_train"], C)
        s_val = 1 / (1 + np.exp(-(Xva @ w + b)))
        ap = np.nan_to_num(average_precision(d["y_val"], s_val))
        if best is None or ap > best[0]:
            best = (ap, C, w, b, s_val)
    ap, C, w, b, s_val = best
    s_test = 1 / (1 + np.exp(-(Xte @ w + b)))
    return s_val, s_test, dict(C=C, val_pr_auc=ap, n_inputs=Xtr.shape[1])


# --------------------------------------------------------------- evaluation --
def evaluate(A, fold, d, h, s_val, s_test):
    y = d["y_test"].astype(int)
    # threshold: the validation-score quantile that alarms on ALARM_SHARE of minutes
    thr = float(np.quantile(s_val, 1 - ALARM_SHARE))
    alarm = s_test >= thr
    tp, fp = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum())
    fn, tn = int((~alarm & (y == 1)).sum()), int((~alarm & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    lo, hi = block_bootstrap_ap(y, s_test)

    # event level: a fault is caught if any minute in the H minutes before it alarms
    score_of = dict(zip(d["rows_test"].tolist(), s_test.tolist()))
    events = []
    for _, r in A.ONSETS[A.ONSETS["datetime"].str[:10].isin(fold.test)].iterrows():
        i = A.ROW_OF[int(r["onset"])]
        lead = next((k for k in range(h, 0, -1)
                     if score_of.get(i - k, -1) >= thr), None)
        events.append(dict(horizon=h, onset=r["datetime"], component=r["component"],
                           reason=r["reason"], detected=lead is not None, lead_min=lead))
    ev = pd.DataFrame(events, columns=["horizon", "onset", "component", "reason",
                                       "detected", "lead_min"])

    # false-alarm episodes: runs of consecutive alarmed negative minutes
    fa = alarm & (y == 0); rows = d["rows_test"]
    starts = fa & ~np.r_[False, fa[:-1] & (np.diff(rows) == 1)]
    n_days = len(fold.test)
    m = dict(horizon=h, base_rate=y.mean(), pr_auc=average_precision(y, s_test),
             pr_auc_lo=lo, pr_auc_hi=hi, roc_auc=roc_auc(y, s_test), threshold=thr,
             tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec,
             f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
             n_events=len(ev), events_detected=int(ev["detected"].sum()),
             event_recall=ev["detected"].mean() if len(ev) else np.nan,
             median_lead_min=ev["lead_min"].median() if len(ev) else np.nan,
             false_alarms_per_day=starts.sum() / n_days, test_days=n_days)
    m["pr_lift"] = m["pr_auc"] / m["base_rate"]
    return m, ev


# ------------------------------------------------- train-only data cleaning --
MAX_MISSING, FFILL_FACTOR = 0.80, 2
_PREP = {}


def prepare_fold(A, fold):
    """Clean the raw series table using the TRAINING days of `fold` only.

    1. keep series observed on every training day
    2. drop series more than 80 % missing on the training days
    3. forward-fill within each day, up to twice the series' own sampling interval
       (estimated on the training days); never backward, so no value from the future
    4. fill what is left with the training median
    Sets A.FVAL / A.FEATS / A.COMPONENT_OF for the pipeline and returns a summary.
    """
    key = (A.DATASET, fold.name)
    if key not in _PREP:
        if not hasattr(A, "X_ALL"):
            A.X_ALL = pd.read_parquet(A.BUILD / "features_all.parquet").reindex(A.GRID)
        X = A.X_ALL
        day = A.DAY_DATE
        tr = np.isin(day, fold.train)
        obs = X.notna().to_numpy()

        on_every_day = np.all([obs[day == d].any(0) for d in fold.train], axis=0)
        missing_tr = 1 - obs[tr].mean(0)
        keep = on_every_day & (missing_tr <= MAX_MISSING)

        limits = {}
        for j in np.flatnonzero(keep):
            gaps = [np.diff(np.flatnonzero(obs[day == d, j])) for d in fold.train]
            gaps = np.concatenate(gaps)
            limits[j] = max(2, FFILL_FACTOR * (int(np.median(gaps)) if len(gaps) else 1))
        Xk = X.iloc[:, list(limits)]
        lim = pd.Series(list(limits.values()), index=Xk.columns)
        filled = pd.concat(
            [Xk[cols.index].groupby(day, sort=False).ffill(limit=int(l))
             for l, cols in lim.groupby(lim.values)], axis=1)[Xk.columns]
        F = filled.to_numpy(dtype=np.float32)
        left_tr, left_all = np.isnan(F[tr]).mean(), np.isnan(F).mean()
        A.FFILL_MISSING = pd.Series(np.isnan(F).mean(0), index=Xk.columns)
        med = np.nanmedian(F[tr], axis=0)
        F = np.where(np.isnan(F), np.nan_to_num(med)[None, :], F).astype(np.float32)
        feats = list(Xk.columns)
        _PREP[key] = (F, feats, dict(
            fold=fold.name, series_raw=X.shape[1],
            dropped_not_every_train_day=int((~on_every_day).sum()),
            dropped_over_80pct_missing=int((on_every_day & ~keep).sum()),
            series_kept=len(feats),
            missing_after_ffill_train=float(left_tr),
            missing_after_ffill_all=float(left_all)))
    F, feats, info = _PREP[key]
    A.FVAL, A.FEATS = F, feats
    A.COMPONENT_OF = pd.Series(feats).str.split("||", regex=False).str[0].to_numpy()
    return info


def to_change(X):
    """Window-relative inputs: subtract each window's own median per feature."""
    return np.clip(X - np.median(X, axis=1, keepdims=True), -10, 10).astype(np.float32)


def main():
    global WITHIN_DAY
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["bank", "telecom"], default="bank")
    ap.add_argument("--split", choices=["chrono", "cv"], default="chrono")
    ap.add_argument("--inputs", choices=["level", "change"], default="level")
    ap.add_argument("--model", choices=["lstm", "logreg"], default="lstm")
    args = ap.parse_args()
    WITHIN_DAY = args.inputs == "change"
    suffix = "" if args.model == "lstm" else f"-{args.model}"
    out = RESULTS / args.system / f"{args.split}-{args.inputs}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    A = P.use_dataset(args.system)
    folds = P.chrono_folds(A) if args.split == "chrono" else P.cv_folds(2, A)
    metrics, history, preds, events, split, prep = [], [], [], [], [], []
    for h in HORIZONS:
        t0 = time.time()
        fold_m, fold_preds, fold_ev = [], [], []
        for fold in folds:
            info = prepare_fold(A, fold)
            if h == HORIZONS[0]:
                prep.append(info)
            d = P.fold_data(fold, h)
            for part in ("train", "val", "test"):
                X = d[f"X_{part}"]
                split.append(dict(horizon=h, fold=fold.name, split=part,
                                  days=", ".join(getattr(fold, part)),
                                  windows=len(d[f"y_{part}"]),
                                  positive=int(d[f"y_{part}"].sum()),
                                  shape=" x ".join(map(str, X.shape)),
                                  clipped_share=float((np.abs(X) >= 10).mean())))
                if args.inputs == "change":
                    d[f"X_{part}"] = to_change(X)
            if args.model == "lstm":
                model, hist = train(d, h)
                s_val, s_test = predict(model, d["X_val"]), predict(model, d["X_test"])
            else:
                s_val, s_test, info_lr = train_logreg(d)
                hist = [dict(horizon=h, **info_lr)]
            for r in hist:
                r["fold"] = fold.name
            m, ev = evaluate(A, fold, d, h, s_val, s_test)
            m["fold"], m["epochs_run"] = fold.name, len(hist)
            fold_m.append(m); history += hist; fold_ev.append(ev.assign(fold=fold.name))
            fold_preds.append(pd.DataFrame(dict(horizon=h, fold=fold.name,
                                                timestamp=A.GRID[d["rows_test"]],
                                                y=d["y_test"].astype(int), score=s_test,
                                                alarm=s_test >= m["threshold"])))
        p = pd.concat(fold_preds); ev = pd.concat(fold_ev)
        if len(folds) == 1:
            m = fold_m[0]
        else:   # pool the out-of-fold predictions; thresholds stay per fold
            y, sc, al = p["y"].to_numpy(), p["score"].to_numpy(), p["alarm"].to_numpy()
            tp, fp = int((al & (y == 1)).sum()), int((al & (y == 0)).sum())
            fn, tn = int((~al & (y == 1)).sum()), int((~al & (y == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            lo, hi = block_bootstrap_ap(y, sc)
            m = dict(horizon=h, fold="pooled", base_rate=y.mean(),
                     pr_auc=average_precision(y, sc), pr_auc_lo=lo, pr_auc_hi=hi,
                     roc_auc=roc_auc(y, sc), threshold=np.nan, tp=tp, fp=fp, fn=fn, tn=tn,
                     precision=prec, recall=rec,
                     f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                     n_events=len(ev), events_detected=int(ev["detected"].sum()),
                     event_recall=ev["detected"].mean(),
                     median_lead_min=ev["lead_min"].median(),
                     false_alarms_per_day=sum(f["false_alarms_per_day"] * f["test_days"]
                                              for f in fold_m)
                     / sum(f["test_days"] for f in fold_m),
                     test_days=sum(f["test_days"] for f in fold_m),
                     epochs_run=np.mean([f["epochs_run"] for f in fold_m]))
            m["pr_lift"] = m["pr_auc"] / m["base_rate"]
        m["seconds"] = round(time.time() - t0)
        metrics.append(m); events.append(ev); preds.append(p)
        print(f"{args.split}-{args.inputs} H={h:<2d} PR-AUC {m['pr_auc']:.3f} "
              f"(base {m['base_rate']:.3f}, lift {m['pr_lift']:.2f}x) "
              f"ROC {m['roc_auc']:.3f} events {m['events_detected']}/{m['n_events']} "
              f"FA/day {m['false_alarms_per_day']:.1f}  {m['seconds']}s", flush=True)

    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
    pd.concat(events).to_csv(out / "events.csv", index=False)
    pd.DataFrame(split).to_csv(out / "split.csv", index=False)
    pd.DataFrame(prep).to_csv(out / "prep.csv", index=False)


if __name__ == "__main__":
    main()
