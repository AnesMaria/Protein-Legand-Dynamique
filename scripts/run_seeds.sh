#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:-davis}"
variant="${2:-classical}"
split="${3:-random}"
projector="${4:-mlp}"
sparsity="${5:-l1}"
for seed in 0 1 2 3 4; do
  python train.py --dataset "$dataset" --projector "$projector" --model "$variant" --split "$split" \
    --seed "$seed" --sparsity-mode "$sparsity" --no-pretrained --no-freeze-sae
done
label="$projector"
if [[ "$projector" == "sae" ]]; then label="sae_$sparsity"; fi
python aggregate.py --root "runs/$dataset/$label/$variant/$split"
