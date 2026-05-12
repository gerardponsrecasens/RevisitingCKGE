#!/bin/bash

# Exit immediately if any command fails
set -e

# Common arguments
COMMON_ARGS=(
  -lifelong_name GKGEL
  -using_embedding_transfer True
)

# Datasets with custom hyperparameters
python main.py -dataset ENTITY \
  "${COMMON_ARGS[@]}" \
  -regular_weight 3.0 \
  -reconstruct_weight 0.1

python main.py -dataset RELATION \
  "${COMMON_ARGS[@]}" \
  -regular_weight 0.09 \
  -reconstruct_weight 0.1

python main.py -dataset HYBRID \
  "${COMMON_ARGS[@]}" \
  -regular_weight 0.05 \
  -reconstruct_weight 0.2

python main.py -dataset FACT \
  "${COMMON_ARGS[@]}" \
  -regular_weight 0.04 \
  -reconstruct_weight 0.7

# Datasets using default GKGEL hyperparameters
python main.py -dataset graph_equal "${COMMON_ARGS[@]}"
python main.py -dataset graph_higher "${COMMON_ARGS[@]}"
python main.py -dataset graph_lower "${COMMON_ARGS[@]}"

# PS-CKGE requires an additional argument
python main.py -dataset PS-CKGE \
  "${COMMON_ARGS[@]}" \
  -snapshot_num 3