#!/bin/bash
# Queue 4 (28 Oct): after queue 3 -> E7 RLDAS condition layer_rawreward (codes observed,
# reward from raw values): 10-D x 5 seeds, two at a time; then 30-D seed 3849.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue4.log; }
train() {  # agent cond seed suite
  out=results_rev/e7/$1_${4#bbob-}_$2_$3
  if [ -f $out/run.json ] && grep -q '"end"' $out/run.json; then log "skip $out (done)"; return; fi
  rm -rf $out
  python -u experiments/e7_train.py --agent $1 --condition $2 --seed $3 --suite $4 \
      --outdir $out > logs_rev/e7_$1_${4#bbob-}_$2_$3.log 2>&1
  log "E7 $out exit $?"
}
log "waiting for queue 3"
while pgrep -f "revision/scripts/queue3.sh" > /dev/null; do sleep 60; done
log "queue 4 start"
train RLDAS layer_rawreward 3849 bbob-10D & train RLDAS layer_rawreward 3850 bbob-10D & wait
train RLDAS layer_rawreward 3851 bbob-10D & train RLDAS layer_rawreward 3852 bbob-10D & wait
train RLDAS layer_rawreward 3853 bbob-10D & train RLDAS layer_rawreward 3849 bbob-30D & wait
log "queue 4 done"
