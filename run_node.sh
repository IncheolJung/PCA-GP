#!/bin/bash
unset DISPLAY
export OMP_NUM_THREADS=$(( $(nproc) / 1 ))
source /home/incheol/.miniconda3/etc/profile.d/conda.sh
conda activate gpytorch
python main.py -F 500 1500 1 -s 1 -S 1 -n 8 --add 4 -i 80 -p ./data/VWT-data/sphere -v
