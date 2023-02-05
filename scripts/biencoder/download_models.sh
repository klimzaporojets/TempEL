#!/usr/bin/env bash

mkdir -p experiments/models/blink/biencoder/train/train_bi_20220630/output/models_ep_9_only/
wget -O experiments/models/blink/biencoder/train/train_bi_20220630/output/models_ep_9_only/download.zip https://cloud.ilabt.imec.be/index.php/s/i7dWxc4Fkw7cKcC/download
unzip experiments/models/blink/biencoder/train/train_bi_20220630/output/models_ep_9_only/download.zip -d experiments/models/blink/biencoder/train/train_bi_20220630/output/
rm experiments/models/blink/biencoder/train/train_bi_20220630/output/models_ep_9_only/download.zip
