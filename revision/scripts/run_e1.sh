#!/bin/bash
# E1: main audit of the 16 MetaBox agents, 20 seeds, on the CPU pod.
cd /workspace/invaudit
source /root/venv/bin/activate
mkdir -p results_rev logs_rev
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python -u experiments/exp2_metabox_audit.py --family audit --seeds 20 --workers ${1:-30} \
     --out results_rev/e1_audit_20seeds.csv >> logs_rev/e1_audit.log 2>&1
