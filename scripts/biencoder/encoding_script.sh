#!/usr/bin/env bash

experiment_name=$1

mkdir -p experiments/models/blink/biencoder/encode/${experiment_name}/logs/
PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2013.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2013.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2014.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2014.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2015.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2015.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2016.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2016.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2017.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2017.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2018.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2018.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2019.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2019.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2020.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2020.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2021.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2021.log

PYTHONPATH=. python -u src/models/s05_wikipedia_encode.py \
    --config_parent_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_parent.json \
    --config_path experiments/models/blink/biencoder/encode/${experiment_name}/config/s05_config_encode_2022.json 2>&1 \
    | tee experiments/models/blink/biencoder/encode/${experiment_name}/logs/s05_wikipedia_encode_2022.log
