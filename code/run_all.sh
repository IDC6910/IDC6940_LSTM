#!/usr/bin/env bash
# Rebuild every result the report reads, from the raw OpenRCA files.
#
#   bash code/run_all.sh                 # uses $PYTHON or `python`
#   PYTHON=/path/to/python bash code/run_all.sh
#
# Needs the OpenRCA dataset and the failure_prediction pipeline next to each other
# (set FAILURE_PREDICTION_DIR if the pipeline lives somewhere else).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
PIPE=${FAILURE_PREDICTION_DIR:-$HOME/OpenRCA/failure_prediction}

for sys in bank telecom; do
  echo "=== $sys: stage 1 (grid, labels, masks) ==="
  (cd "$PIPE" && "$PY" build_dataset.py --dataset "$sys")
  echo "=== $sys: unfiltered series table ==="
  "$PY" code/build_all_series.py "$sys"
  for split in chrono cv; do
    for inputs in level change; do
      echo "=== $sys: LSTM $split-$inputs ==="
      "$PY" -u code/run_lstm.py --system "$sys" --split "$split" --inputs "$inputs"
    done
  done
  echo "=== $sys: drift, exploration, LSTM study ==="
  "$PY" code/export_drift.py "$sys" > /dev/null
  "$PY" code/export_eda.py "$sys"
  "$PY" -u code/study_lstm.py "$sys"
done
echo "all done"
