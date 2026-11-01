#!/bin/bash
# Queue 1 (27 Oct): after E1 finishes -> E2 inference -> E5b knock-outs -> E2 calibration.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue1.log; }
log "waiting for E1"
while pgrep -f "experiments/exp2_metabox_audit.py --family audit --seeds 20" > /dev/null; do sleep 60; done
log "E1 finished: $(wc -l < results_rev/e1_audit_20seeds.csv) lines"
python revision/scripts/e2_audit_inference.py --inp results_rev/e1_audit_20seeds.csv \
    --out results_rev/e2_audit_inference.csv --workers 30 > logs_rev/e2_inference.log 2>&1
log "E2 inference exit $?"
python -u experiments/e5_knockouts.py --workers 30 --out results_rev/e5_knockouts.csv > logs_rev/e5_knockouts.log 2>&1
log "E5b exit $?"
python revision/scripts/e2_calibration.py --reps 5000 --workers 30 > logs_rev/e2_calibration.log 2>&1
log "E2 calibration exit $?"
log "queue 1 done"
