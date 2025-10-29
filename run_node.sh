#!/bin/bash
unset DISPLAY
export OMP_NUM_THREADS=$(( $(nproc) / 1 ))
source /home/incheol/.miniconda3/etc/profile.d/conda.sh
conda activate gpytorch
python main.py -F 500 1500 1 -s 1 -S 1 -n 2 --add 2 -i 80 -t 5 -p ./data/VWT-data/sphere -v
