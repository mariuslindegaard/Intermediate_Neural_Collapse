#!/bin/bash

source ~/.conda_init
conda activate nc

# /bin/false
# while [ $? -ne 0 ]; do
echo "~~~ RUNNING EXPERIMENT ~~~"
python3 ../../main.py --config ../../config/base_runs/${SLURM_ARRAY_TASK_ID}.yaml

