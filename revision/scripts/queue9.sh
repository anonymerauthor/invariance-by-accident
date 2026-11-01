#!/bin/bash
# Queue 9 (30 Oct): anchors, beside the DEDDQN training (10 workers).
#   exp1  pod rerun at 10-D (baseline for the harness cost, one platform)
#   exp6  harness on the anchors at 10-D, with order-embedding instrumentation (R2.4)
#   E1b   anchors on the MetaBox instances, 20 seeds, audit + held-out
#   E3    anchors at 30 / 50 / 100-D, budget 2000 x D, one probe per group action (R3.2)
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${W:-10}
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue9.log; }
log "start"
python -u experiments/anchor_grid.py --dim 10 --budget 20000 --probes exp1 --workers $W \
    --out results_rev/exp1_anchors_10D.csv > logs_rev/exp1_10D.log 2>&1
log "exp1 10D exit $?"
python -u experiments/anchor_grid.py --dim 10 --budget 20000 --probes exp6 --harness --workers $W \
    --out results_rev/exp6_harness_10D.csv > logs_rev/exp6_10D.log 2>&1
log "exp6 10D exit $?"
python -u experiments/exp3_anchors_on_bbob.py --family audit --seeds 20 --workers $W \
    --out results_rev/e1b_anchors_audit.csv > logs_rev/e1b_audit.log 2>&1
log "E1b audit exit $?"
python -u experiments/exp3_anchors_on_bbob.py --family ood --seeds 20 --workers $W \
    --out results_rev/e1b_anchors_ood.csv > logs_rev/e1b_ood.log 2>&1
log "E1b ood exit $?"
for D in 30 50 100; do
  python -u experiments/anchor_grid.py --dim $D --probes e3 --workers $W \
      --out results_rev/e3_anchors_${D}D.csv > logs_rev/e3_${D}D.log 2>&1
  log "E3 ${D}D exit $?"
done
log "queue 9 done"
