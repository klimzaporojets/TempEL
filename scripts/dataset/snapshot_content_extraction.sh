#!/usr/bin/env bash

mkdir -p experiments/snapshot_extraction/snap_20220515/logs/
PYTHONPATH=. python -u src/tempel_creation/s01_wikipedia_clean_and_tokenize.py \
    --config_file experiments/snapshot_extraction/snap_20220515/config/s01_config_content.json \
    --nr_threads_processor 40 --nr_threads_reader 20 2>&1 | \
    tee experiments/snapshot_extraction/snap_20220515/logs/s01_wikipedia_clean_and_tokenize.log
