#!/bin/bash

target_dir="cf_t"

for i in {0..0}; do

    python scripts/chaotic_fractals.py \
    --dir "$target_dir" \
    --config "test_suite_ours" \
    --trainer_type "cf" \
    --gt_name "fdb_$i"
    # &

    wait
done
