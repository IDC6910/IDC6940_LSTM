# Modeling plan: section 3 "Modeling and Results"

Brainstorm for the modeling part of the report, built on the structure of the
reference project
([Predicting Plant Extinctions Using the Random Forest Algorithm](https://uwfgit.github.io/STA6257_random_forest_project/)).
Nothing here is in `index.qmd` yet.

## 1. How the reference project structures its modeling

| Reference section | What it does |
|---|---|
| Methods → Evaluation | Defines the metrics (accuracy, recall, precision, F1) with formulas *before* any results |
| Statistical Modeling → Packages | One short paragraph per package and what it is used for |
| Data Preparation | Encodes variables, shows the data before/after, splits train/test 7:3 |
| Class Balancing | Explains the imbalance, gives the resampling formula (ROSE), prints class counts before/after (`490 148 23` → `490 490 490`) |
| Normalization | Applies 5 scaling methods, each with its formula, and trains one model per method |
| Training Steps for Each Model | Lists the steps, then a small metric table per model |
| Prediction | Combines the 5 models by hard voting, final confusion matrix and per-class metrics |
| Conclusion | Which variant won and why |

The pattern: **explain one step → show its formula → show what it did to the data → move on.** Every step leaves a small table or figure.

## 2. Proposed outline for our report

Our sections 3.1–3.3 (Dataset Description, Load Data, Data Preparation and Labeling)
already cover the reference's "Data and Visualization". The new part goes after
`@fig-overview` and before `## References`.

### Add to Methods first: "Evaluation Metrics"

The reference defines its metrics in Methods. We removed our evaluation subsection,
so section 3 currently uses PR-AUC without defining it. Add a short subsection after
Parameter Estimation:

- Confusion matrix terms (TP, FP, FN, TN) for "failure within $H$ minutes"
- Precision, recall, F1 (same formulas as the reference)
- **PR-AUC** and its baseline (= positive rate), **ROC-AUC**
- Event-level metrics the reference doesn't have but we need:
  - **event recall**: share of faults with at least one alarm before onset
  - **false alarms per day**
  - **lead time**: minutes between first correct alarm and the fault
- Why not accuracy: at 1 % positives, "never alarm" scores 99 %

### 3.4 Modeling and Results

| # | Subsection | Content | Table / figure | Reference parallel |
|---|---|---|---|---|
| 3.4.1 | **Packages** | Python: `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`. R (report only): `tidyverse`, `knitr` | — | Packages |
| 3.4.2 | **Train / Validation / Test Split** | Split by whole day, oldest → newest. ~30 % of days for test, the day before for validation (early stopping + threshold), the rest for training. Explain why not a random 7:3 split of minutes: neighbouring windows overlap 59 of 60 minutes | Table of days per split and faults per split; existing `figures/*_03_splits.png` | 7:3 split |
| 3.4.3 | **Class Imbalance Handling** | Keep all positive windows, subsample negatives to 5:1, plus `pos_weight` in the loss. Explain why we *don't* use ROSE/SMOTE: interpolating between two 60-minute windows produces telemetry that never happened | Positive/negative counts before → after per horizon (like `490 148 23 → 490 490 490`) | ROSE + class counts |
| 3.4.4 | **Scaling** | Formulas for min-max, z-score and robust (median/IQR, clip ±10), all fitted on training days only. **Experiment:** train the LSTM once per scaling method and compare. Heavy-tailed KPIs should favour robust scaling | One window before/after scaling (heatmap); PR-AUC per scaling method | 5 normalization methods |
| 3.4.5 | **Windowing and Feature Selection** | Build $N \times 60 \times k$ tensors; keep top $k = 128$ features by ANOVA $F$ (give the formula) on training days only | Tensor shapes per split; top-10 features table (component, KPI, F) | Encoding / data processing |
| 3.4.6 | **Training** | Steps list (like the reference's "Training Steps"): build loader → weighted BCE → Adam → clip → validate each epoch → early stop → restore best | Learning curves: training loss and validation PR-AUC by epoch, Bank and Telecom | Training steps |
| 3.4.7 | **Results by Prediction Horizon** | Main result. One LSTM per horizon $H \in \{1,3,5,10,15,30\}$ per system | Table per system: H × (base rate, PR-AUC, lift, ROC-AUC, precision, recall, F1, event recall, false alarms/day, lead time). Figure: PR-AUC lift vs H, both systems | Per-model score tables |
| 3.4.8 | **Focus Horizon: $H = 5$** | Threshold chosen on validation; confusion matrix on test; which fault types are caught | Confusion matrix; detection rate by fault type; a timeline plot of predicted probability around 2–3 test faults | Final confusion matrix |
| 3.4.9 | **Ensemble** | Our version of hard voting: train 3 seeds (or the 3 scaling variants) and average probabilities. Does it beat a single model? | Single vs ensemble PR-AUC | Hard voting classifier |
| 3.4.10 | **Ablations** | Look-back $L \in \{15,30,60,120\}$; feature count $k \in \{32,128,384\}$; telemetry source (container only / app only / both). The source ablation answers the intro's "does combining telemetry sources help?" | One small table per ablation (Bank; Telecom if time) | — (beyond the reference) |

### 3.5 Conclusion

- How far ahead can the LSTM warn us, and how reliably (lift, event recall, lead time)?
- Does it hold on both systems?
- Limits: injected faults, short recordings, 136 and 51 events, Telecom only 00:00–05:59.

## 3. Decisions for the team

1. **Baselines or LSTM alone?** "How effectively" is hard to judge with no reference
   point. Proposal: keep the LSTM as the only model we explain, but add one line per
   table for the **base rate** (random guessing) and optionally **logistic
   regression** on window summaries. The pipeline already has both, plus gradient
   boosting, 1D-CNN and BiLSTM, if we want more later.
2. **Focus horizon:** $H = 5$ (about 5 % positives, still a real warning) or $H = 10$?
3. **Splits:** chronological only (fast, realistic) or also leave-two-days-out
   cross-validation (more test faults, much slower)? Proposal: chronological for
   the report, CV only if time allows.
4. **Scaling experiment (3.4.4) and ensemble (3.4.9):** keep both? They mirror the
   reference closely but each adds a few training runs.

## 4. How the code fits the report

Training takes minutes to hours, so it should **not** run when the report is
rendered. Proposal:

1. Run the models offline with the existing pipeline in
   `OpenRCA/failure_prediction` (`pipeline.py` + `02_models_evaluation.ipynb`;
   `torch` 2.5 is installed in the `openrca` conda env).
2. Save small outputs into this repo: `results/<system>/results.csv`,
   `history.csv` (loss per epoch), `ablations.csv`, and figures under `figures/`.
3. In `index.qmd`, R chunks read those CSVs and print tables with `kable()`, the same
   way Load Data reads `data/*_record.csv`. Show the Python training code as
   non-running code blocks (`code-fold: true`).

What the pipeline already provides vs what is missing:

| Needed | Status |
|---|---|
| Day-based split, scaling, top-k, windows, negative subsampling | ✅ `pipeline.py` |
| LSTM + training loop with early stopping | ✅ notebook 02 |
| PR-AUC, ROC-AUC, precision/recall/F1, event recall, false alarms, lead time | ✅ notebook 02 |
| Ablations (look-back, k, modality) | ✅ notebook 02 |
| Results saved to CSV | ✅ (no run saved yet: `results/*/results.csv` doesn't exist) |
| Per-epoch training history | ❌ add to `train_torch` |
| Scaling comparison (min-max / z-score / robust) | ❌ add a `scaler=` option to `fold_data` |
| Seed ensemble | ❌ small loop around `train_torch` |
| Class counts before/after subsampling | ❌ one print in `fold_data` |

## 5. Suggested order of work

1. Agree on the decisions in section 3.
2. Add the Evaluation Metrics subsection to Methods.
3. Write 3.4.1–3.4.5 (no results needed).
4. Add the missing pieces to the pipeline, run a `smoke` preset to check, then
   `standard` for Bank and Telecom.
5. Copy the results into this repo; write 3.4.6–3.4.10 around the numbers.
6. Conclusion.
