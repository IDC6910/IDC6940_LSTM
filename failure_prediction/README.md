# Predicting impending failures from multivariate telemetry (OpenRCA)

**Research question.** How effectively can LSTM networks predict impending
software-system failures from multivariate telemetry — and do they beat classical
models given the same features?

Metrics-only, two systems (**Bank** and **Telecom**), six prediction horizons, seven
model families. `PIPELINE.md` is the full design document; this README is the
orientation.

## Layout

| file | what it is |
|---|---|
| `01_data_pipeline.ipynb` | Stages 1–3: build/load, EDA, day-grouped splits, windowing + leakage guards |
| `02_models_evaluation.ipynb` | Stages 4–6: the model zoo, row- and event-level evaluation, ablations |
| `pipeline.py` | shared core both notebooks import (`from pipeline import *`) |
| `config.py` | dataset registry (Bank, Telecom) + Stage 1 knobs |
| `build_dataset.py` | Stage 1 builder — raw CSVs → aligned minute-level matrices |
| `verify_stage1.py` | standalone timezone / label / leakage assertions |
| `PIPELINE.md` | the design document: data, labelling, splits, models, metrics, ablations |

Derived data (`build/`) and model runs (`results/`) are gitignored — they are
reproducible from the raw dataset in about four minutes.

## Getting the data

This code reads the [OpenRCA](https://github.com/microsoft/OpenRCA) dataset, which is
not redistributed here. Download it from the link in that repo's `dataset/README.md`
and lay it out as:

```
<somewhere>/OpenRCA/
├── dataset/
│   ├── Bank/      { record.csv, query.csv, telemetry/<YYYY_MM_DD>/... }
│   └── Telecom/   { record.csv, query.csv, telemetry/<YYYY_MM_DD>/... }
└── failure_prediction/      <- this folder
```

`config.py` resolves the dataset relative to its own parent, so keeping this folder
inside the OpenRCA checkout needs no configuration. To keep it here in the course repo
instead, point `OPENRCA_DATA` at the dataset directory:

```bash
export OPENRCA_DATA=~/OpenRCA/dataset
```

## Running it

```bash
cd failure_prediction
pip install -r requirements.txt
python3 build_dataset.py --dataset all     # ~3 min Bank, ~1 min Telecom
python3 verify_stage1.py  --dataset all    # assertions, not prose
jupyter lab                                # run 01, then 02
```

Switch systems by editing one line in either notebook — `DATASET = "bank"` or
`"telecom"` — and re-running; both stay cached, so it is instant after the first load.
Presets are set with `OPENRCA_PRESET`: `smoke` (~1 min), `standard` (~10–15 min, the one
to report), `full` (adds day-grouped CV and ablations).

## The two systems

| | Bank | Telecom |
|---|---|---|
| coverage | 10 days × 1,440 min | 15 days × 360 min (00:00–05:59 UTC+8) |
| labelled minutes | 14,400 | 5,400 |
| features after Stage 1 | 1,565 | 1,222 |
| components | 15 | 47 |
| fault events | 136, 8 reasons | 51, 5 reasons |

## Two things to know before reading any result

**Timezone.** Every timestamp in OpenRCA — `record.csv`, the metric files, the
`2021_03_04/` folder names — is **UTC+8**. Render epochs in local time and faults land
on the wrong day, on innocent components, and the labels quietly decorrelate from the
telemetry. This pipeline joins and windows only on raw epoch seconds and derives day
membership from the UTC+8 date. `verify_stage1.py` asserts our rendering reproduces
`record.csv`'s own `datetime` column for all 187 faults; it raises if it doesn't.

**This is a small-event problem.** Tens of GB of telemetry reduce to 136 (Bank) and 51
(Telecom) fault events. The events, not the minutes, are the effective sample size:
splits are grouped by day, PR-AUC is always reported against the base rate it must beat,
and confidence intervals bootstrap over events. Differences under ~0.05 PR-AUC are noise.
