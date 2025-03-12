#!/bin/bash

target_dir="simulated_annealing"

for i in {0..0}; do

    python scripts/chaotic_fractals.py \
    --dir "$target_dir" \
    --config "ablation_sa" \
    --trainer_type "sa" \
    --gt_name "fdb_$i" 
    # &

done
# Wait for all background jobs to finish
wait