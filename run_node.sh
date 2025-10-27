#!/bin/bash
unset DISPLAY
export OMP_NUM_THREADS=$(( $(nproc) / 1 ))
source /home/incheol/.miniconda3/etc/profile.d/conda.sh
conda activate gpytorch
python main.py -F 500 800 100 -s 1 -S 1 -n 3 --add 3 -i 3 -p ./data/VWT-data/sphere -v
