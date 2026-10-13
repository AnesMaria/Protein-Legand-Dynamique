#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:-davis}"
variant="${2:-classical}"
split="${3:-random}"
for seed in 0 1 2 3 4; do
  python train.py --dataset "$dataset" --model "$variant" --split "$split" --seed "$seed"
done
python aggregate.py --root "runs/$dataset/$variant/$split"
