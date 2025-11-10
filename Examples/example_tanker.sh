#!/bin/bash
cd ..
unset DISPLAY
export OMP_NUM_THREADS=$(nproc)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gpytorch
python main.py -F 100 300 1 -A 0 10 0.1 -s 0 -S 0 -n 4 --add 1 -i 100 -t 5 -T 5e-3 -p ./data/VWT-data/tanker
