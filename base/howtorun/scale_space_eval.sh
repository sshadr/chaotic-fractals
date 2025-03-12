#!/bin/bash

# the log_dir passed as arguments to the script have a root path 
# and the following are the names of the folder where their corresponding method's log files exist.
# we use the optimized code from these log dirs to generate images in the scale space for our numerical evaluation.

save_root="SAVE_ROOT" # Where all the images generated using this script are saved, arranged according to folders("method") 
gt_code_path="GT_PATH" # Where all groundtruth fractal codes are stored
ours_log_dir="OURS_LOG_DIR"   # Directory for 'ours' method logs
moments_log_dir="MOMENTS_LOG_DIR" # Directory for 'moments' method logs
evol_pcov_log_dir="EVOL_PCOV_LOG_DIR" # Directory for 'evol_pcov' method logs
cuckoo_log_dir="CUCKOO_LOG_DIR"  # Directory for 'cuckoo' method logs
nr_log_dir="NR_LOG_DIR"    # Directory for 'neural regression' method logs
lf32_log_dir="LF32_LOG_DIR"  # Directory for 'lf_32' method logs
lf256_log_dir="LF256_LOG_DIR" # Directory for 'lf_256' method logs

# -- method flag has options for which method to evaluate
python scripts/scale_space_eval.py \
--idx $index \
--method "ours" \
--save_root $save_root \
--gt_log_dir $gt_code_path \
--ours_log_dir $ours_log_dir \
--lf32_log_dir $lf32_log_dir \
--lf256_log_dir $lf256_log_dir \
--moments_log_dir $moments_log_dir \
--evol_pcov_log_dir $evol_pcov_log_dir \
--cuckoo_log_dir $cuckoo_log_dir \
--nr_log_dir $nr_log_dir
