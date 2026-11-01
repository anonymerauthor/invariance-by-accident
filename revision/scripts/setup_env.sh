#!/bin/bash
# Pod environment: Python 3.11 venv in /root/venv with versions pinned to the
# original Windows environment (revision/constraints_linux.txt).
set -e
python3.11 -m venv /root/venv || { apt-get update -qq && apt-get install -y -qq python3.11-venv python3.11-dev && python3.11 -m venv /root/venv; }
source /root/venv/bin/activate
pip install -q --upgrade "pip<25"
C=/workspace/invaudit/revision/constraints_linux.txt
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip download -q --no-deps metaevobox==2.0.2 -d /root/pkgs
sha256sum /root/pkgs/metaevobox-2.0.2*
pip install -q metaevobox==2.0.2 "scipy==1.11.1" matplotlib pytest -c <(grep -v -E "^(torch|torchvision)==" $C)
pip freeze > /workspace/invaudit/revision/pip_freeze_linux.txt
python -c "import metaevobox, torch, numpy, scipy, pandas; print('OK', torch.__version__, numpy.__version__, scipy.__version__, pandas.__version__)"
