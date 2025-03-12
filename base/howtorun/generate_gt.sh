#!/bin/bash

# save_dir="SAVE_DIR" # change to full path
save_dir="./dataset/test"
index=0

python data_gen/ground_truth_generator.py \
--dir $save_dir \
--num_f 10 \
--gt_name "$index" \
--method "imp"