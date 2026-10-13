#!/usr/bin/env bash
set -euo pipefail
cd /workspace/proteine-legand
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/proteine-legand/.cache/huggingface
python3 -m venv /workspace/.venvs/proteine-legand
python=/workspace/.venvs/proteine-legand/bin/python
"$python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1'
"$python" -m pip install -r requirements.txt -c requirements-cpu.lock
"$python" -m pip check
OMP_NUM_THREADS=2 "$python" -m pytest -q
