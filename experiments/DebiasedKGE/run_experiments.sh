#!/bin/bash

# Exit immediately if any command fails
set -e

SEED=55

# Standard datasets
DATASETS=(
  "FACT"
  "RELATION"
  "HYBRID"
  "graph_equal"
  "graph_higher"
  "graph_lower"
)

# Run standard datasets
for DATASET in "${DATASETS[@]}"; do
  python main.py -dataset "$DATASET" -seed "$SEED"
done

# PS-CKGE requires an additional argument
python main.py -dataset PS-CKGE -seed "$SEED" -snapshot_num 3

# ENTITY requires custom hyperparameters
python main.py \
  -dataset ENTITY \
  -seed "$SEED" \
  -learning_rate 0.0001 \
  --num_factors 3 \
  --alignment_weight 0.05