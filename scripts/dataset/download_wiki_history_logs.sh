#!/usr/bin/env bash

mkdir -p data/wikipedia_dump/enwiki_20220201/pages_meta_history/
wget -O data/wikipedia_dump/enwiki_20220201/zipped_enwiki.zip https://cloud.ilabt.imec.be/index.php/s/8KtT3HxDHybsNmt/download
unzip data/wikipedia_dump/enwiki_20220201/zipped_enwiki.zip -d data/wikipedia_dump/
rm data/wikipedia_dump/enwiki_20220201/zipped_enwiki.zip