#!/bin/bash

# Exit immediately if any command fails
set -e

# Common arguments shared by all runs
COMMON_ARGS=(
  -lifelong_name double_tokened
  -using_token_distillation_loss True
  -use_multi_layers False
  -without_multi_layers True
  -use_two_stage False
  -batch_size 3072
  -learning_rate 0.001
  -patience 3
)

# ENTITY
python main.py -dataset ENTITY \
  "${COMMON_ARGS[@]}" \
  -token_distillation_weight 5000 10000 10000 10000 \
  -token_num 2 \
  -div_loss_weight 0.2

# FACT
python main.py -dataset FACT \
  "${COMMON_ARGS[@]}" \
  -token_distillation_weight 1000 10000 10000 10000 \
  -token_num 4 \
  -div_loss_weight 0.6

# HYBRID
python main.py -dataset HYBRID \
  "${COMMON_ARGS[@]}" \
  -token_distillation_weight 10000 3000 800 200000 \
  -token_num 10 \
  -div_loss_weight 0.2

# RELATION
python main.py -dataset RELATION \
  "${COMMON_ARGS[@]}" \
  -token_distillation_weight 3000 15000 80000 80000 \
  -token_num 2 \
  -div_loss_weight 0.2

# Datasets using only common arguments
python main.py -dataset graph_equal  "${COMMON_ARGS[@]}"
python main.py -dataset graph_higher "${COMMON_ARGS[@]}"
python main.py -dataset graph_lower  "${COMMON_ARGS[@]}"

# PS-CKGE requires an additional argument
python main.py -dataset PS-CKGE \
  "${COMMON_ARGS[@]}" \
  -snapshot_num 3