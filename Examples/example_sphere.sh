#!/bin/bash
cd ..
unset DISPLAY
export OMP_NUM_THREADS=$(nproc)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gpytorch
python main.py -F 100 300 1 -i 30 -t 5 -T 1e-3 -p ./data/VWT-data/sphere
