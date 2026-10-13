#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
python3 -m venv /workspace/.venvs/protein-dynamic
python=/workspace/.venvs/protein-dynamic/bin/python
"$python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1'
"$python" -m pip install -r requirements-md.txt -c requirements-dynamic-cpu.lock
"$python" -m pip check
OMP_NUM_THREADS=2 "$python" -m pytest -q
