#!/bin/bash

index=0 # name of the fractal

# Evolutionary
# method="evolutionary"
# fitness_fn="pointcov" # or "hamming" or "ourloss"
# GENERATIONS=1000 # 1000

# # Cuckoo search
method="cuckoosearch" #
fitness_fn="hamming" 
GENERATIONS=2500 #2500

python scripts/stochastic.py \
    --gt_name fdb_$index \
    --optimizer $method \
    --fitness $fitness_fn \
    --num_gens $GENERATIONS \
    --gen_size 100\
    --ifs_size 10