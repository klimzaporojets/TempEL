#!/usr/bin/env bash

mkdir -p experiments/dataset_creation/data_20230112_from_cloud
cd experiments/dataset_creation/data_20230112_from_cloud
wget -O tempel_v1.0_only_bert_tokenized.tar.gz https://cloud.ilabt.imec.be/index.php/s/aHGj3NAGWsYF78t/download/tempel_v1.0_only_bert_tokenized.tar.gz
tar -xf tempel_v1.0_only_bert_tokenized.tar.gz
cd -
