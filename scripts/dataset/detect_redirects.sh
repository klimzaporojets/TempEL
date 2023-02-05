#!/usr/bin/env bash

mkdir -p experiments/snapshot_extraction/snap_20220515/logs/
PYTHONPATH=src python -u src/tempel_creation/s02_redirect_detector.py \
    --config_file experiments/snapshot_extraction/snap_20220515/config/s02_redirect_detector.json 2>&1 \
    | tee experiments/snapshot_extraction/snap_20220515/logs/s02_redirect_detector.log
