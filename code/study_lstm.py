#!/usr/bin/env python3
"""Study how the LSTM behaves on one system (chronological split, level inputs).

    python code/study_lstm.py bank
    python code/study_lstm.py telecom

Writes to results/<system>/study/:
  lookback.csv          test/validation PR-AUC for look-back L in LOOKBACKS
  hidden.csv            the same for hidden size d in HIDDEN
  seeds.csv             the default model trained with five random seeds
  gates.csv             mean forget / input / output gate value per window minute
  saliency_time.csv     mean |input x gradient| per window minute
  saliency_feature.csv  mean |input x gradient| per feature (top 15)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_lstm as R                                   # noqa: E402
P = R.P

H = 10
LOOKBACKS = [1, 5, 15, 30, 60, 120]
HIDDEN = [8, 16, 32, 64, 128]
SEEDS = [P.SEED + i for i in range(5)]

system = sys.argv[1] if len(sys.argv) > 1 else "bank"
OUT = R.RESULTS / system / "study"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

A = P.use_dataset(system)
fold = P.chrono_folds(A)[0]
R.prepare_fold(A, fold)


def run(lookback=60, hidden=64, seed=P.SEED):
    d = P.fold_data(fold, H, lookback=lookback)
    model, hist = R.train(d, H, hidden=hidden, seed=seed)
    s_val, s_test = R.predict(model, d["X_val"]), R.predict(model, d["X_test"])
    m, _ = R.evaluate(A, fold, d, H, s_val, s_test)
    val_base = d["y_val"].mean()
    row = dict(lookback=lookback, hidden=hidden, seed=seed,
               params=sum(p.numel() for p in model.parameters()),
               val_pr_auc=R.average_precision(d["y_val"], s_val), val_base=val_base,
               test_pr_auc=m["pr_auc"], test_lo=m["pr_auc_lo"], test_hi=m["pr_auc_hi"],
               test_base=m["base_rate"], test_roc=m["roc_auc"],
               best_epoch=hist[0]["best_epoch"])
    row["val_lift"], row["test_lift"] = row["val_pr_auc"] / val_base, m["pr_lift"]
    print(row, flush=True)
    return row, model, d


pd.DataFrame([run(lookback=L)[0] for L in LOOKBACKS]).to_csv(OUT / "lookback.csv", index=False)
pd.DataFrame([run(hidden=h)[0] for h in HIDDEN]).to_csv(OUT / "hidden.csv", index=False)
rows = [run(seed=s) for s in SEEDS]
pd.DataFrame([r[0] for r in rows]).to_csv(OUT / "seeds.csv", index=False)

# ---- inside the default model (first seed): gates and saliency on the test days --
_, model, d = rows[0]
model.eval()
X, y = torch.from_numpy(d["X_test"]), d["y_test"].astype(int)
rnn = model.rnn
W_ih, W_hh = rnn.weight_ih_l0.detach(), rnn.weight_hh_l0.detach()
b = (rnn.bias_ih_l0 + rnn.bias_hh_l0).detach()
hid = W_hh.shape[1]
with torch.no_grad():
    h_t = torch.zeros(len(X), hid); c_t = torch.zeros(len(X), hid)
    gates = []
    for t in range(X.shape[1]):
        z = X[:, t, :] @ W_ih.T + h_t @ W_hh.T + b       # PyTorch order: i, f, g, o
        i, f, g, o = z.split(hid, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)
        for name, v in (("forget", f), ("input", i), ("output", o)):
            for cls in (0, 1):
                gates.append(dict(minute=t - X.shape[1] + 1, gate=name,
                                  label="pre-fault" if cls else "other",
                                  mean=float(v[y == cls].mean())))
    # sanity check: the manual pass reproduces the model's own output
    ref = model.head(h_t).squeeze(-1)
    out = model(X)
    assert torch.allclose(ref, out, atol=1e-4), "manual LSTM pass does not match"
pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False)

Xg = X.clone().requires_grad_(True)
model(Xg).sum().backward()
sal = (Xg.grad * Xg).abs().detach().numpy()             # n x L x k
t_rows = []
for cls in (0, 1):
    m = sal[y == cls].mean(axis=(0, 2))
    t_rows += [dict(minute=t - len(m) + 1, label="pre-fault" if cls else "other",
                    saliency=float(v)) for t, v in enumerate(m)]
pd.DataFrame(t_rows).to_csv(OUT / "saliency_time.csv", index=False)
feat = sal.mean(axis=(0, 1))
order = np.argsort(-feat)[:15]
pd.DataFrame(dict(feature=[A.FEATS[d["cols"][j]] for j in order],
                  saliency=feat[order])).to_csv(OUT / "saliency_feature.csv", index=False)
print("done", system)
