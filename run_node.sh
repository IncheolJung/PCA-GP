#!/bin/bash
unset DISPLAY
export OMP_NUM_THREADS=$(( $(nproc) / 4 ))
source /home/incheol/.miniconda3/etc/profile.d/conda.sh
conda activate gpytorch
python main.py -F 500 1500 301 -s 1 -S 1 -n 2 --add 1 -i 100 -p ./data/VWT-data/sphere -v -x 2
