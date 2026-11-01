#!/bin/bash
# Queue 6 (28 Oct): after queue 5 -> E6: the 16 distributed agents with the objective layer at
# test time (audit grid) and held-out functions with / without the layer, 20 seeds.
# 24 workers, leaving cores for the three DEDDQN trainings.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue6.log; }
log "waiting for queue 5"
while pgrep -f "revision/scripts/queue5.sh" > /dev/null || pgrep -f "run_e7_eval_early.sh" > /dev/null; do sleep 60; done
for F in audit_layer ood ood_layer; do
  python -u experiments/exp2_metabox_audit.py --family $F --seeds 20 --workers 24 \
      --out results_rev/e6_${F}_20seeds.csv > logs_rev/e6_$F.log 2>&1
  log "E6 $F exit $?"
done
log "queue 6 done"
