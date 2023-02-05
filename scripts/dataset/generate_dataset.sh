#!/usr/bin/env bash

mkdir -p experiments/dataset_creation/data_20220515/logs/
PYTHONPATH=src python -u src/tempel_creation/s03_final_dataset_creator.py \
    --config_file experiments/dataset_creation/data_20220515/config/s03_dataset_creator.json \
     --nr_threads 4 2>&1 \
     | tee experiments/dataset_creation/data_20220515/logs/s03_dataset_creator.log