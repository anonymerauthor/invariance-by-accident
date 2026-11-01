#!/bin/bash
# Queue 2 (28 Oct): after queue 1 -> E7 timing smoke (1 epoch, layer condition) for RLDAS and DEDDQN, in parallel.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue2.log; }
log "waiting for queue 1"
while pgrep -f "revision/scripts/queue1.sh" > /dev/null; do sleep 60; done
log "queue 1 finished; starting E7 smoke"
for A in RLDAS DEDDQN; do
  ( python -u experiments/e7_train.py --agent $A --condition layer --epochs 1 \
      --outdir results_rev/e7_smoke/${A}_10D_layer > logs_rev/e7_smoke_$A.log 2>&1
    log "E7 smoke $A exit $?" ) &
done
wait
log "queue 2 done"
