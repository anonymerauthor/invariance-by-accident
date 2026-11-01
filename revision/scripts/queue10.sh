#!/bin/bash
# Queue 10 (30 Oct): E3 agents, zero-shot at 30 / 50-D with the pretrained 10-D models (R3.2).
# B2OPT, RLDAS, RLPSO and RNNOPT are locked to 10-D (input layers sized for D = 10; smoke
# test 30 Oct) and are excluded; RLDAS is covered at 30-D by the E7 retraining.
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${W:-8}
AG=DEDDQN,DEDQN,GLEET,GLHF,LDE,LES,NRLPSO,QLPSO,RLDEAFL,RLEPSO,RLHPSDE,SYMBOL
PR=ambient_translation_diag,rotation,coordinate_permutation,domain_scaling_up,fitness_rescale,monotone_log
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue10.log; }
log "start"
python -u experiments/exp2_metabox_audit.py --family audit --agents $AG --probes $PR --seeds 20 \
    --dim 30 --budget 50000 --workers $W --out results_rev/e3_agents_30D.csv > logs_rev/e3_agents_30D.log 2>&1
log "E3 agents 30D exit $?"
python -u experiments/exp2_metabox_audit.py --family audit --agents $AG --probes $PR --seeds 10 \
    --dim 50 --budget 100000 --workers $W --out results_rev/e3_agents_50D.csv > logs_rev/e3_agents_50D.log 2>&1
log "E3 agents 50D exit $?"
log "queue 10 done"
