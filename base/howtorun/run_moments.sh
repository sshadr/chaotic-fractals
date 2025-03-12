#!/bin/bash

target_dir="moments"

for i in {0..0}; do

    python scripts/chaotic_fractals.py \
    --dir "$target_dir" \
    --config "ablation_moments" \
    --trainer_type "moments" \
    --gt_name "fdb_$i"
    # &

done
# Wait for all background jobs to finish
wait