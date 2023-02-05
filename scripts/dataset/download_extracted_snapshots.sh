#!/usr/bin/env bash

sudo apt-get -y install lz4
mkdir -p experiments/snapshot_extraction/snap_20220515/
wget -O experiments/snapshot_extraction/snap_20220515/download.zip https://cloud.ilabt.imec.be/index.php/s/Ytt2WDTJH5r3w4z/download
unzip experiments/snapshot_extraction/snap_20220515/download.zip -d experiments/snapshot_extraction/snap_20220515/
rm experiments/snapshot_extraction/snap_20220515/download.zip
cd experiments/snapshot_extraction/snap_20220515/output/
cat splitted_wikipedia_evol/wikipedia_evolution_content.jsonl.lz4.a \
splitted_wikipedia_evol/wikipedia_evolution_content.jsonl.lz4.b \
splitted_wikipedia_evol/wikipedia_evolution_content.jsonl.lz4.c \
splitted_wikipedia_evol/wikipedia_evolution_content.jsonl.lz4.d > wikipedia_evolution_content.jsonl.lz4
lz4 -d wikipedia_evolution_content.jsonl.lz4
rm wikipedia_evolution_content.jsonl.lz4
cd -
