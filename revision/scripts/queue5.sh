#!/bin/bash
# Queue 5 (28 Oct): after queue 4 -> E7 evaluation of the 10-D RLDAS models.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue5.log; }
log "waiting for queue 4"
while pgrep -f "revision/scripts/queue4.sh" > /dev/null; do sleep 60; done
log "E7 eval start"
python -u experiments/e7_eval.py --runs results_rev/e7 --out results_rev/e7_eval.csv --workers 30 > logs_rev/e7_eval.log 2>&1
log "E7 eval exit $?"
log "queue 5 done"
