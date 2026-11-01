#!/bin/bash
# Queue 3 (28 Oct): E7 RLDAS retraining. 10-D: {control, layer} x 5 seeds, two runs at a time
# (each run uses 16 env subprocesses). Then 30-D: {control, layer} x seed 3849 in parallel.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue3.log; }
train() {  # agent cond seed suite
  out=results_rev/e7/$1_${4#bbob-}_$2_$3
  if [ -f $out/run.json ] && grep -q '"end"' $out/run.json; then log "skip $out (done)"; return; fi
  rm -rf $out
  python -u experiments/e7_train.py --agent $1 --condition $2 --seed $3 --suite $4 \
      --outdir $out > logs_rev/e7_$1_${4#bbob-}_$2_$3.log 2>&1
  log "E7 $out exit $?"
}
log "queue 3 start"
for S in 3849 3850 3851 3852 3853; do
  train RLDAS control $S bbob-10D & train RLDAS layer $S bbob-10D & wait
done
train RLDAS control 3849 bbob-30D & train RLDAS layer 3849 bbob-30D & wait
log "queue 3 done"
