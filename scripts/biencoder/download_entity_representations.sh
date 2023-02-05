#!/usr/bin/env bash

mkdir -p experiments/models/blink/biencoder/encode/20220630/output/faiss/
wget -O experiments/models/blink/biencoder/encode/20220630/output/faiss/download.zip https://cloud.ilabt.imec.be/index.php/s/DSiKXQwWpQacRJ6/download
unzip experiments/models/blink/biencoder/encode/20220630/output/faiss/download.zip -d experiments/models/blink/biencoder/encode/20220630/output/
rm experiments/models/blink/biencoder/encode/20220630/output/faiss/download.zip