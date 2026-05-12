for dataset in ENTITY RELATION FACT HYBRID graph_equal graph_higher graph_lower
  do
    for method in finetune EWC EMR LKGE
      do
        for seed in 11 22 33 44 55
          do
            note=$seed
            python -u main.py -dataset $dataset -gpu 0 -lifelong_name $method -learning_rate 0.0001 -batch_size 2048 -seed $seed -note $note
          done
      done
  done


for dataset in PS-CKGE
  do
    for method in finetune EWC EMR LKGE
      do
        for seed in 11 22 33 44 55
          do
            note=$seed
            python -u main.py -dataset $dataset -gpu 0 -lifelong_name $method -learning_rate 0.0001 -batch_size 2048 -seed $seed -note $note
          done
      done
  done