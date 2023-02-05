#!/usr/bin/env bash

experiment_name=$1

mkdir -p experiments/stats/results_metrics/${experiment_name}/logs/
PYTHONPATH=. python -u src/stats/s07_calculate_results_and_metrics_blink.py \
    --config_file experiments/stats/results_metrics/${experiment_name}/config/s07_config_metrics.json 2>&1 \
    | tee experiments/stats/results_metrics/${experiment_name}/logs/s07_calculate_metrics.log
