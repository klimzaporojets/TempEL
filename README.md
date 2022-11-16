# TempEL
Repository for Temporal Entity Linking (TempEL), accepted to NeurIPS 2022 Dataset and Benchmarks 

## Dataset 
The dataset can be downloaded from [this link](https://cloud.ilabt.imec.be/index.php/s/RinXy8NgqdW58RW) 

Currently, the dataset is available in the following two formats: 
1. ```tempel_v1.0_all```: the dataset files in this directory include the complete text of Wikipedia anchor and target pages
for each of the instances. 
2. ```tempel_v1.0_only_bert_tokenized```: the dataset files in this directory only include the truncated bert tokenization
used in our baseline model. 

## Dataset creation, code and models
### Dataset creation
__Step 0: download the necessary Wikipedia tables and history logs__

The TempEL creation pipeline begins with history logs of Wikipedia as well as 
Wikipedia sql tables with auxiliary information (e.g., tables listing all the redirect pages in Wikipedia), 
which can be downloaded from our cloud storage using the following commands: 
```
mkdir -p data/wikipedia_dump/enwiki_20220201/pages_meta_history/
wget -O data/wikipedia_dump/enwiki_20220201/zipped_enwiki.zip https://cloud.ilabt.imec.be/index.php/s/8KtT3HxDHybsNmt/download
unzip data/wikipedia_dump/enwiki_20220201/zipped_enwiki.zip data/wikipedia_dump/enwiki_20220201/
rm data/wikipedia_dump/enwiki_20220201/pages_meta_history/zipped_history.zip 
```
The cloud console with all the Wikipedia files can also be accessed at the following link: 
[https://cloud.ilabt.imec.be/index.php/s/8KtT3HxDHybsNmt](https://cloud.ilabt.imec.be/index.php/s/8KtT3HxDHybsNmt). 

__Step 1: snapshot content extraction__ 

The [wikitext](https://en.wikipedia.org/wiki/Help:Wikitext) content is cleaned
 for for each of the temporal snapshots. 
The hyperparameter configuration is located in 
```experiments/snapshot_extraction/snap_20220515/config/s01_config_content.json``` file. Execute: 
```
mkdir -p experiments/snapshot_extraction/snap_20220515/logs/
PYTHONPATH=src python -u src/tempel_creation/s01_wikipedia_clean_and_tokenize.py \
    --config_file experiments/snapshot_extraction/snap_20220515/config/s01_config_content.json 2>&1 \
    --nr_threads_processor 40 --nr_threads_reader 20 | \
    tee experiments/snapshot_extraction/snap_20220515/logs/s01_wikipedia_clean_and_tokenize.log
```
The script above will generate the ```experiments/snapshot_extraction/snap_20220515/output/wikipedia_evolution_content.jsonl``` 
file with cleaned wikipedia content for each of the entities in snapshots. 
It will also create a number of files in ```experiments/snapshot_extraction/snap_20220515/output/stats/snapshot_stats/``` 
directory that we later use to 
calculate statistics (such as the _mention prior_) to be used during the creation of TempEL dataset. 


__Step 2.a: inverse dictionary and mention-driven statistics generation__

In this step we generate inverse dictionary index that maps a particular target entity to wikipedia
 pages with mentions linked to that entity. Additional mention-based statistics (e.g., 
 mention prior, edit distance between mention and target entity title, etc.) are also generated and 
 serialized to disc as .csv files. The detailed configuration of the hyperparameters used for this 
 step is located in the ```experiments/snapshot_extraction/snap_20220515/config/s02_alias_table_generator.json``` 
 file. 
 The following is the python command to generate the above mentioned inverse dictionary and mention-driven 
 statistics: 
 ```
mkdir -p experiments/snapshot_extraction/snap_20220515/logs/
PYTHONPATH=src python -u src/tempel_creation/s02_alias_table_generator.py \
    --config_file experiments/snapshot_extraction/snap_20220515/config/s02_alias_table_generator.json 2>&1 \
    | tee experiments/snapshot_extraction/snap_20220515/logs/s02_alias_table_generator.log 
 ```

__Step 2.b: detecting redirects in page history__

Some Wikipedia pages are redirect pages in one or more of the snapshots. We do not include these page in 
TempEL dataset, since we are interested in pages with actual content and not redirects pointing to other 
pages. The following python command detects these pages and generates an output file with the corresponding
Wikidata QID and Wikipedia title:
```
mkdir -p experiments/snapshot_extraction/snap_20220515/logs/
PYTHONPATH=src python -u src/tempel_creation/s02_redirect_detector.py \
    --config_file experiments/snapshot_extraction/snap_20220515/config/s02_redirect_detector.json 2>&1 \
    | tee experiments/snapshot_extraction/snap_20220515/logs/s02_redirect_detector.log
```  

__Alternative: download files from the cloud__ 

It can take multiple days for the scripts in __Step 1__ and __Step 2__
 to complete and generate all the files. Alternatively, these files can also be downloaded from our cloud 
executing (the lz4 decompressor has to be 
installed first: ```sudo apt-get -y install lz4```):

```
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
```
The files can also be accessed via the following link: 
[https://cloud.ilabt.imec.be/index.php/s/Ytt2WDTJH5r3w4z](https://cloud.ilabt.imec.be/index.php/s/Ytt2WDTJH5r3w4z)

__Step 3: TempEL dataset generation__

```
mkdir -p experiments/dataset_creation/data_20220515/logs/
PYTHONPATH=src python -u src/tempel_creation/s03_final_dataset_creator.py \
    --config_file experiments/dataset_creation/data_20220515/config/s03_dataset_creator.json \
     --nr_threads 4 2>&1 \
     | tee experiments/dataset_creation/data_20220515/logs/s03_dataset_creator.log 

```

__Note 1__ set ```nr_threads``` to be close to the number of the available CPUs.

__Note 2__ the selection of the entities and mentions to produce TempEL dataset is random. Therefore, the produced
version of TempEL dataset will differ to the one used in the paper available in 
[this link](https://cloud.ilabt.imec.be/index.php/s/RinXy8NgqdW58RW).

 


### Models
The baseline models (as well as the code to train and evaluate) will be available shortly, stay tuned!