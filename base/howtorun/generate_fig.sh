#!/bin/bash

# trajectory file (.txt) contains the views to be rendered.
# This file is generated from the openGL viewer's trajectory capturing functionality.
trajectory_file="TRAJECTORY_FILE" # Trajectory file name

# Use fractal name to access the corresponding log files in the log_dirs
index=0
fractal_name="fdb_$index"

gt_log_dir="GT_CODE_PATH"           # Directory containing groundtruth fractal codes
ours_log_dir="OURS_LOG_DIR"        # Directory for 'ours' method logs
moments_log_dir="MOMENTS_LOG_DIR"  # Directory for 'moments' method logs
evol_pcov_log_dir="EVOL_PCOV_LOG_DIR" # Directory for 'evol_pcov' method logs
cuckoo_log_dir="CUCKOO_LOG_DIR"    # Directory for 'cuckoo' method logs
nr_log_dir="CNN_LOG_DIR"          # Directory for 'neural regression' method logs
lf32_log_dir="LF32_LOG_DIR"        # Directory for 'lf_32' method logs
lf256_log_dir="LF256_LOG_DIR"      # Directory for 'lf_256' method logs

python scripts/fractal_zoomer.py \
--trajectory_file $trajectory_file \
--gt_log_dir $gt_log_dir \
--ours_log_dir $ours_log_dir \
--lf32_log_dir $lf32_log_dir \
--lf256_log_dir $lf256_log_dir \
--moments_log_dir $moments_log_dir \
--evol_pcov_log_dir $evol_pcov_log_dir \
--cuckoo_log_dir $cuckoo_log_dir \
--nr_log_dir $nr_log_dir \
--method "ours" \
--resume_idx 0