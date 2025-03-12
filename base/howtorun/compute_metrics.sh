#!/bin/bash

root_folder="ROOT" # provide full path
gt_folder="gt" # just folder name
opt_folder="ours" # just folder name

python scripts/compute_metrics.py \
--root "$root_folder" \
--gt_folder $gt_folder \
--opt_folder "$opt_folder" \