#!/bin/bash

# Exit immediately if any command fails
set -e

# Fixed experiment settings
SEED=55

# Hyperparameter ranges
ENT_R_VALUES=(150 200)
NUM_LAYER_VALUES=(1)
LEARNING_RATES=(0.1 0.2 0.3)

# Datasets
DATASETS=(
  "ENTITY"
  "graph_equal"
  "graph_higher"
  "graph_lower"
  "FACT"
  "RELATION"
  "HYBRID"
  "PS-CKGE"
)

# Run experiments
for DATASET in "${DATASETS[@]}"; do
  for ENT_R in "${ENT_R_VALUES[@]}"; do
    for NUM_LAYER in "${NUM_LAYER_VALUES[@]}"; do
      for LR in "${LEARNING_RATES[@]}"; do

        CMD=(
          python main.py
          -dataset "$DATASET"
          -ent_r "$ENT_R"
          -num_layer "$NUM_LAYER"
          -learning_rate "$LR"
          -random_seed "$SEED"
        )

        # PS-CKGE requires an additional argument
        if [[ "$DATASET" == "PS-CKGE" ]]; then
          CMD+=(-snapshot_num 3)
        fi

        echo "Running: ${CMD[*]}"
        "${CMD[@]}"

      done
    done
  done
done