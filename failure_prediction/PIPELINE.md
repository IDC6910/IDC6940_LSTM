# Predicting impending failures in OpenRCA — **Bank** and **Telecom** — from multivariate telemetry

**Research question.** How effectively can LSTM networks predict impending
software-system failures from multivariate telemetry, and do they beat classical
models that see the same features?

**Short answer to "can we?":** yes, the data supports it — but it is a *small-event*
problem dressed up as a big-data problem. Tens of GB of telemetry reduce to 14,400
labelled minutes / **136 events** (Bank) and 5,400 minutes / **51 events** (Telecom).
Everything below is designed around that: the events, not the minutes, are the
effective sample size.

Both systems run through one pipeline. Switch with `DATASET = "bank" | "telecom"` in
either notebook, or `OPENRCA_DATASET=telecom` in the environment; Stage 1 outputs go to
`build/<dataset>/` and results to `results/<dataset>/`, so the two never collide.

---

## 1. The two systems

| | **Bank** | **Telecom** |
|---|---|---|
| folder | `dataset/Bank` | `dataset/Telecom` |
| days | 10 (2021-03-04…03-25, **not contiguous**) | 15 (2020-04-11…05-31, **not contiguous**) |
| coverage per day | 1,440 min (full day) | **360 min (00:00–05:59 UTC+8 only)** |
| labelled minutes | 14,400 | 5,400 |
| timestamp unit | seconds | **milliseconds** |
| metric files | `metric_container`, `metric_app` | `metric_node`, `metric_container`, `metric_middleware`, `metric_service`, `metric_app` |
| components | 18 raw → **15** retained | 51 raw → **47** retained (`os_*`, `docker_*`, `db_*`, `redis_*`) |
| features after Stage 1 | **1,565** | **1,222** |
| faults | **136**, 8 reasons | **51**, 5 reasons |
| fault density | 9.44 per 1,000 min | 9.44 per 1,000 min |
| residual missingness | 4.5 % | 1.1 % |
| other modalities (out of scope) | `log_service`, `trace_span` | `trace_span` |

Bank fault reasons: high CPU (33), network packet loss (32), network latency (27),
high disk I/O read (19), high memory (10), JVM OOM heap (7), high disk space (5),
high JVM CPU load (3).
Telecom fault reasons: CPU fault (19), network delay (13), db connection limit (7),
network loss (7), db close (5).

Facts that shape the design:

* **Sampling is ragged in both.** Bank samples many container KPIs every 2 minutes and
  a few hourly (~48 % of the 1-minute grid empty); Telecom samples at roughly 60 s but
  jittered by seconds (~26 % empty). After dropping series above 80 % missing and
  forward-filling within a day by each series' own cadence, residual missingness is
  4.5 % / 1.1 %.
* **Components come and go.** Bank's `dockerA1/A2/B1/B2` are present early and mostly
  gone by 03-24, so they fail the present-on-every-day rule and are dropped (248
  series). Telecom drops 430 series the same way. Say so in the thesis; silently
  keeping them breaks the late folds.
* **Telecom KPI names collide across families** (`node` and `service` both have
  `CPU_Used_Pct`), so its features are keyed `component||family.kpi`. Bank's KPI names
  already carry their family, so they stay `component||kpi`.
* **Telecom's day is six hours and always the same six hours.** A time-of-day feature
  means something very different there — only 360 clock positions, all in the small
  hours — which is why the `time_of_day` control model is less informative on Telecom
  than on Bank.

### Timezone — the thing that silently ruins results

Every timestamp in OpenRCA (`record.csv`, metric files, the `2021_03_04/` folder
names) is **UTC+8**. If you render epochs in US Central you land 14 hours off,
faults appear on the wrong day and on components that look innocent, and your
labels quietly decorrelate from the telemetry. The rule this pipeline follows:

* join and window **only on raw epoch seconds** — never on a formatted string;
* render human-readable time with `TZ = UTC+8` and nothing else;
* derive the day a row belongs to from `epoch → UTC+8 date`, never `epoch // 86400`.

`verify_stage1.py` asserts that our UTC+8 rendering reproduces `record.csv`'s own
`datetime` column for all 136 faults. That assertion is your regression test.

---

## 2. Target definition

Two framings, both built:

* **System-level (headline).** `y_sys[t, H] = 1` if *any* fault onset falls in
  `(t, t+H]`. Answers "is the system about to break?"
* **Per-component (secondary).** `y_comp[t, H, c]` for each of the 14 components —
  supports a localization claim, but is far sparser.

Six horizons `H ∈ {1, 3, 5, 10, 15, 30}` minutes. Realised system-level rates — close
enough across the two systems to make the horizon comparison fair:

| H (min) | Bank positives / evaluable | rate | Telecom positives / evaluable | rate |
|---|---|---|---|---|
| 1 | 129 / 12,382 | 1.04 % | 42 / 4,033 | 1.04 % |
| 3 | 386 / 12,370 | 3.12 % | 126 / 4,003 | 3.15 % |
| 5 | 642 / 12,363 | 5.19 % | 210 / 3,974 | 5.28 % |
| 10 | 1,279 / 12,358 | 10.35 % | 420 / 3,904 | 10.76 % |
| 15 | 1,902 / 12,368 | 15.38 % | 630 / 3,839 | 16.41 % |
| 30 | 3,573 / 12,459 | 28.68 % | 1,217 / 3,812 | 31.93 % |

**Two guards that make these numbers honest:**

1. *Blackout.* `record.csv` gives an onset but no duration. Minutes from onset to
   onset+10 are flagged `in_fault` and excluded from the negative pool, so the
   model is never rewarded for "predicting" an incident already in progress.
   `BLACKOUT` is a config knob — run 5/10/20 as a robustness check.
2. *Window completeness.* A row is evaluable only if its 60-minute look-back and
   its H-minute look-ahead both fit inside the same UTC+8 day. Days are not
   contiguous; letting a window straddle 03-12 → 03-23 would be nonsense.

---

## 3. Pipeline

```
Stage 0  raw CSVs (per day: metric_container, metric_app)
Stage 1  align      -> minute grid, pivot to wide, fill, label      [BUILT]
Stage 2  split      -> day-grouped folds, fit scaler on train only
Stage 3  window     -> (N, 60, F) tensors + y, sample weights
Stage 4  model      -> 6 model families x 6 horizons
Stage 5  evaluate   -> row metrics + event metrics + bootstrap CIs
Stage 6  ablate     -> lookback, feature set, blackout, class weighting
```

### Stage 1 — alignment and labelling *(implemented: `build_dataset.py`)*

Pivots each day to `timestamp × (component||kpi)`, reindexes onto the exact
1,440-minute UTC+8 grid, appends the 44 app-metric series, keeps only series
present on all ten days, drops >80 %-missing series, forward-fills within a day,
and emits labels plus masks. Outputs land in `build/`:

| file | shape / contents |
|---|---|
| `features.parquet` | raw aligned (NaNs preserved) — 14,400 × 1,685 / 5,400 × 1,643 |
| `features_filled.parquet` | gap-filled — **model input** — 14,400 × 1,565 / 5,400 × 1,222 |
| `labels_system.parquet` | minutes × 6 (`h1 … h30`) |
| `labels_component.parquet` | minutes × (6 × components) MultiIndex |
| `masks.parquet` | `in_fault`, `eval_h1 … eval_h30` |
| `feature_index.csv` | component / family / kpi / missingness / fill limit per feature |
| `onsets.csv` | 136 / 51 onsets, epoch + UTC+8 datetime, grid-snapped |
| `meta.json` | days, grid shape, knobs used for this build |
| `report.md` | build diagnostics |

(Bank values first, Telecom second. Outputs live in `build/<dataset>/`.)

### Stage 2 — splitting

**Group by day, never shuffle minutes.** Adjacent minutes are near-duplicates; a
random row split inflates AUC into the 0.99s and means nothing.

* Primary protocol: **leave-2-days-out, 5 folds** over the 10 days, stratified so
  each test fold holds roughly 20–30 faults.
* Secondary protocol: **chronological** — train 03-04…03-10, validate 03-12,
  test 03-23…03-25. This is the deployment-realistic number and will be *worse*;
  report both and say why they differ (the March 23–25 block has a different
  component inventory).
* Standardize per feature with train-fold statistics only (median/IQR — these KPIs
  are heavy-tailed). Persist the scaler with the fold.

### Stage 3 — windowing

* Look-back `L = 60` minutes, stride 1 → `(N, 60, 1565)` float32.
* 12.4 k usable rows × 60 × 1,565 × 4 B ≈ **4.6 GB** if materialized (Bank; Telecom is
  ~1.2 GB). Don't. Keep
  the day matrix in memory and slice windows in a `Dataset.__getitem__`.
* Optional feature reduction *inside the fold* (this is where the LSTM story is
  won or lost): per-component aggregation (mean/max over that component's KPIs),
  variance filtering, or top-k by train-fold mutual information. Report the
  full-feature and reduced-feature results — reviewers will ask.
* Class imbalance: `pos_weight` in `BCEWithLogitsLoss` (≈ 95:1 at H=1), or
  negative subsampling at a fixed ratio. Never oversample by duplicating windows
  across the fold boundary.

### Stage 4 — models (six families, all fed the same windows)

| # | Family | What it tests |
|---|---|---|
| 1 | **Persistence / prior baseline** — predict the train-fold base rate; plus a "last-value threshold" rule on CPU/mem | The floor. PR-AUC below this means the model learned nothing |
| 2 | **Logistic regression** on window summary stats (mean, std, min, max, last, slope per feature) | Does anything beyond a linear read of the current state help? |
| 3 | **Random forest / gradient boosting (XGBoost or LightGBM)** on the same summary stats | The classical champion; usually very hard to beat on tabular telemetry |
| 4 | **1D-CNN** over the raw window | Local shape detection without recurrence |
| 5 | **LSTM** (1–2 layers, 64–128 hidden, dropout 0.2) on the raw window | The hypothesis under test |
| 6 | **BiLSTM + attention** (or GRU) | Does temporal context *plus* a learned focus add anything over plain LSTM? |

Keep the hyperparameter budget identical across families (e.g. 30 random-search
trials each, selected on validation PR-AUC) — otherwise "LSTM wins" just means
"LSTM got more tuning".

### Stage 5 — evaluation

Accuracy is meaningless here. Report, per (model × horizon):

*Row-level*
* **PR-AUC (primary)** and the base rate next to it, always.
* ROC-AUC (secondary — flattering under imbalance, report for comparability).
* Precision/recall/F1 at a threshold chosen on the *validation* fold, not the test fold.
* Brier score / reliability curve if you want a calibration claim.

*Event-level (this is what an SRE cares about, and what makes the thesis land)*
* **Event recall @ H**: fraction of the 136 faults with at least one alarm in the
  H minutes before onset.
* **False alarms per day** at the operating threshold.
* **Mean / median lead time** of the first true alarm before onset.
* Event recall broken down by the 8 fault reasons — expect CPU/memory faults to be
  predictable and network packet loss to be near-unpredictable. That breakdown is
  a more interesting finding than the headline AUC.

*Uncertainty*
* **Bootstrap over events (not rows)**, 1,000 resamples, for 95 % CIs. With 136
  events, differences under ~0.05 PR-AUC will not be significant — say so rather
  than ranking noise.

### Stage 6 — ablations

Look-back `L ∈ {15, 30, 60, 120}` · feature set (container-only / app-only / both,
full / reduced) · blackout `{5, 10, 20}` · imbalance handling (weighting vs
subsampling) · per-component vs system-level target · split protocol
(grouped CV vs chronological).

---

## 4. What to expect, honestly

* At **H = 1–5** the signal should be strongest and the class rate lowest; at
  **H = 30** the task drifts from "prediction" toward "is this a busy period",
  and a high AUC there may reflect diurnal load rhythm rather than foresight.
  Guard against it: add a "time-of-day only" control model. If it scores near
  your LSTM at H=30, that horizon's result is a rhythm detector.
* Expect gradient boosting on window statistics to be competitive with or ahead
  of the LSTM. That is a legitimate, publishable finding, and it is the reason
  the classical arm has to be tuned as hard as the neural one.
* Faults inject at ~1 % of minutes and are synthetic injections; do not
  generalize to production incident prediction without a caveat paragraph.
* **A cross-system claim needs both numbers.** "The LSTM works" is a finding only if it
  holds on Bank *and* Telecom. If it holds on one and not the other, that difference is
  the result — lead with it rather than with the better of the two. And remember
  Telecom has 51 events to Bank's 136, so its CIs are roughly 1.6× wider: a 0.05 PR-AUC
  edge there is noise.

---

## 5. Running it

Two notebooks, sharing one core module so nothing is computed twice:

| file | what it is |
|---|---|
| `01_data_pipeline.ipynb` | Stages 1–3: build/load, EDA, day-grouped splits, windowing + leakage guards |
| `02_models_evaluation.ipynb` | Stages 4–6: the seven-model zoo, evaluation, ablations |
| `pipeline.py` | shared core both notebooks import (`from pipeline import *`) |
| `build_dataset.py` | Stage 1 builder — run automatically on first import, then cached |
| `verify_stage1.py` | standalone timezone / label / leakage assertions |

```bash
cd ~/OpenRCA/failure_prediction
pip install pandas numpy pyarrow matplotlib scikit-learn   # required
pip install xgboost torch                                  # notebook 02; degrades without them
jupyter lab                                                # then run 01, then 02
```

Switch systems inside either notebook by editing one line — `DATASET = "bank"` or
`"telecom"` — and re-running; both stay cached in memory, so it is instant after the
first load. From the shell instead:

```bash
python3 build_dataset.py --dataset all      # build both (~3 min Bank, ~1 min Telecom)
python3 verify_stage1.py --dataset all      # timezone / label / leakage assertions
OPENRCA_DATASET=telecom jupyter lab
```

In notebook 02, set `DATASETS_TO_RUN = ["bank", "telecom"]` to fit both in one pass;
its last section then compares them from `results/<dataset>/results.csv`.

**Presets** (set `OPENRCA_PRESET` before importing, or edit `pipeline.py`):

| preset | what it runs | rough time |
|---|---|---|
| `smoke` | H=5 only, 4 epochs, top-48 | ~1 min |
| `standard` | all 6 horizons × 7 models, chronological split, 15 epochs, top-128 | ~10–15 min, ~2 GB RAM |
| `full` | adds leave-2-days-out CV and ablations | ~1–2 h |

Stage 1 knobs are **per dataset**, in the `Dataset` entries in `config.py`
(`lookback`, `blackout`, `max_missing`, `min_day_coverage`, `ffill_factor`, plus the
`TZ` contract). Change any and delete `build/<dataset>/` to force a rebuild. Stage 2–6
knobs live in `pipeline.py` (`PRESETS`).

Outputs land in `results/<dataset>/`: `results.csv`, `ablations.csv`, and seven figures
under `results/<dataset>/figures/`, plus two cross-system figures at `results/`.

**Adding a third system** is one `Dataset` entry in `config.py` listing its telemetry
files as `Source(...)` rows — long format (timestamp/component/kpi/value) or wide
(timestamp/entity + value columns), seconds or milliseconds. Nothing else in the
pipeline is dataset-specific. Market is not registered yet: it splits into
`cloudbed-1` / `cloudbed-2` sub-systems and needs one entry per bed.
