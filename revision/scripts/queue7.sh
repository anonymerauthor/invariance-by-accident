#!/bin/bash
# Queue 7 (28 Oct): after queue 6 -> E7 evaluation of the 30-D RLDAS models (budget 50,000 as in
# MetaBox's 30-D protocol), 24 workers.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue7.log; }
log "waiting for queue 6"
while pgrep -f "revision/scripts/queue6.sh" > /dev/null; do sleep 60; done
python -u experiments/e7_eval.py --runs results_rev/e7 --dim 30 --budget 50000 \
    --out results_rev/e7_eval_30D.csv --workers 24 > logs_rev/e7_eval_30D.log 2>&1
log "E7 eval 30-D exit $?"
log "queue 7 done"
