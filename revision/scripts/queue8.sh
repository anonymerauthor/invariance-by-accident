#!/bin/bash
# Queue 8 (29 Oct): runs beside the DEDDQN training (which uses ~11 of 32 cores).
#   E12  reference DE variants: box effect of the rotation probe (R3.4)
#   E12  16 agents, identity + rotation on the training functions, with exposure
#   E8   16 agents, all probes on the held-out functions, with exposure (R1.2, R3.5)
cd /workspace/invaudit
source /root/venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> logs_rev/queue8.log; }
log "start"
python -u experiments/e12_rotation_control.py --out results_rev/e12_rotation_control.csv \
    --workers 20 > logs_rev/e12_control.log 2>&1
log "E12 control exit $?"
python -u experiments/exp2_metabox_audit.py --family rot --seeds 20 --workers 20 \
    --out results_rev/e12_rot_20seeds.csv > logs_rev/e12_rot.log 2>&1
log "E12 rot exit $?"
python -u experiments/exp2_metabox_audit.py --family ood_probes --seeds 20 --workers 20 \
    --out results_rev/e8_ood_probes_20seeds.csv > logs_rev/e8_ood_probes.log 2>&1
log "E8 ood_probes exit $?"
log "queue 8 done"
