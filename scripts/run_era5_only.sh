#!/usr/bin/env bash
# ERA5-only experiments (target era5_swh), one phase per invocation so that
# Phase 3 (trees, CPU) and Phase 4 (LSTM) can run side by side:
#
#   PHASE=3 nohup caffeinate -i bash scripts/run_era5_only.sh > results/logs/phase3.log 2>&1 &
#   PHASE=4 nohup caffeinate -i bash scripts/run_era5_only.sh > results/logs/phase4.log 2>&1 &
#
# Budget knobs: TRIALS / TIMEOUT_MIN (Optuna, per study), EPOCHS / PATIENCE (LSTM).
# Progress lines start with "=== ".
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BUOYS="${BUOYS:-BD08 BD10 BD11}"
PHASE="${PHASE:-3}"
TRIALS="${TRIALS:-40}"
TIMEOUT_MIN="${TIMEOUT_MIN:-15}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-6}"
export TF_CPP_MIN_LOG_LEVEL=2

for b in $BUOYS; do
  echo "=== phase$PHASE $b start $(date '+%F %T')"
  if [ "$PHASE" = 3 ]; then
    $PY scripts/run_phase3.py --buoy "$b" --target era5_swh --trials "$TRIALS" --timeout-min "$TIMEOUT_MIN"
  else
    $PY scripts/run_phase4.py --buoy "$b" --target era5_swh --stride 3 --epochs "$EPOCHS" --patience "$PATIENCE"
  fi
  status=$?
  if [ $status -eq 0 ]; then echo "=== phase$PHASE $b done $(date '+%F %T')"; else echo "=== phase$PHASE $b FAILED $(date '+%F %T')"; fi
done
echo "=== phase$PHASE all done $(date '+%F %T')"
