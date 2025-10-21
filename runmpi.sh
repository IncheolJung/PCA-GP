#!/bin/bash
unset DISPLAY
source /home/incheol/.miniconda3/etc/profile.d/conda.sh
conda activate gpytorch
python main.py -F 500 1500 301 -s 1 -S 1 -p ./data/VWT-data/sphere

