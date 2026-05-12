#!/bin/bash

# Exit immediately if any command fails
set -e

# Common embedding dimensions
EMB_ARGS=(
  -emb_dim 160
  -emb2_dim 170
  -emb3_dim 180
  -emb4_dim 190
  -emb_dim5 200
)

# Datasets
DATASETS=(
  "ENTITY"
  "RELATION"
  "HYBRID"
  "FACT"
  "graph_equal"
  "graph_higher"
  "graph_lower"
)

# Run standard datasets
for DATASET in "${DATASETS[@]}"; do
  python main.py -dataset "$DATASET" "${EMB_ARGS[@]}"
done

# PS-CKGE requires an additional argument
python main.py \
  -dataset PS-CKGE \
  -snapshot_num 3 \
  "${EMB_ARGS[@]}"