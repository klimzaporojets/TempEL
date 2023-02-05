#!/usr/bin/env bash

experiment_name=$1

mkdir -p experiments/models/blink/biencoder/train/${experiment_name}/logs/
#python -u -m src.models.s04_main_train_biencoder_blink \
PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2013.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2013.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2014.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2014.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2015.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2015.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2016.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2016.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2017.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2017.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2018.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2018.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2019.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2019.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2020.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2020.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2021.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2021.log

PYTHONPATH=. python -u src/models/s04_main_train_biencoder_blink.py \
    --config_parent_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_parent.json \
    --config_path experiments/models/blink/biencoder/train/${experiment_name}/config/s04_config_train_2022.json 2>&1 \
    | tee experiments/models/blink/biencoder/train/${experiment_name}/logs/s04_train_biencoder_2022.log
