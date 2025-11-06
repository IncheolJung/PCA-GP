#!/bin/bash
unset DISPLAY
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
# source $(conda info --base)/etc/profile.d/conda.sh
source /home/incheol/.miniconda3/etc/profile.d/conda.sh
conda activate gpytorch
# python main.py -F 230 360 1 -s 1 -S 1 -n 4 --add 1 -i 100 -t 5 -T 5e-3 -p ./data/VWT-data/circylinder-200MHz -v
python main.py -F 10000 15000 10 -s 1 -S 0 -n 4 --add 1 -i 100 -t 5 -T 5e-3 -p ./data/VWT-data/prime -v
