

# Datasets to process
DATASETS=(
  "graph_higher"
  "graph_lower"
  "graph_equal"
  "ENTITY"
  "FACT"
  "RELATION"
  "HYBRID"
)

# Run main.py for standard datasets
for DATASET in "${DATASETS[@]}"; do
    python main.py -dataset "$DATASET" 
done

# Run PS-CKGE with an additional argument
python main.py -dataset "PS-CKGE" -snapshot_num 3