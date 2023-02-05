#!/usr/bin/env bash

experiment_name=$1

# evaluating year 2013
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2013.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2013.log

# evaluating year 2014
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2014.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2014.log

# evaluating year 2015
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2015.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2015.log

# evaluating year 2016
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2016.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2016.log

# evaluating year 2017
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2017.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2017.log

# evaluating year 2018
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2018.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2018.log

# evaluating year 2019
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2019.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2019.log

# evaluating year 2020
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2020.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2020.log

# evaluating year 2021
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2021.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2021.log

# evaluating year 2022
mkdir -p experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s06_main_eval_faiss_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_parent.json \
    --config_path experiments/models/blink/biencoder/evaluate/${experiment_name}/config/s06_config_evaluate_2022.json 2>&1 \
    | tee experiments/models/blink/biencoder/evaluate/${experiment_name}/logs/s06_evaluate_biencoder_2022.log
