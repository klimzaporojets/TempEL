# TempEL
Repository for Temporal Entity Linking (TempEL), accepted to NeurIPS 2022 Dataset and Benchmarks 

## 1 Dataset 
The dataset can be downloaded from [this link](https://cloud.ilabt.imec.be/index.php/s/RinXy8NgqdW58RW) 

Alternatively, the TempEL dataset can be downloaded using ```wget``` linux command:
```
wget -O tempel_dataset.zip https://cloud.ilabt.imec.be/index.php/s/RinXy8NgqdW58RW/download
```

Currently, the dataset is available in the following two formats: 
1. ```tempel_v1.0_all``` (78.4 GB): the dataset files in this directory include the complete text of Wikipedia anchor and target pages
for each of the instances. 
2. ```tempel_v1.0_only_bert_tokenized``` (8.6 GB): the dataset files in this directory only include the truncated bert tokenization
used in our baseline model. 

## 2 Dataset creation, code and models
Below we describe details to re-generate the TempEL dataset, which can be used to 
create other TempEL versions using different hyperparameters, such as the time span between the 
temporal snapshots. 

### 2.1 TempeEL dataset creation
<!--The disk space requirement of the steps described below is of xx GB.--> 

__Step 0: download the necessary Wikipedia tables and history logs__
The disk space requirement of this step is of 400 GB. 

The TempEL creation pipeline begins with history logs of Wikipedia as well as 
Wikipedia sql tables with auxiliary information (e.g., tables listing all the redirect pages in Wikipedia), 
which can be downloaded from our cloud storage using the following commands: 
```
./scripts/dataset/download_wiki_history_logs.sh
```
The cloud console with all the Wikipedia files can also be accessed at the following link: 
[https://cloud.ilabt.imec.be/index.php/s/8KtT3HxDHybsNmt](https://cloud.ilabt.imec.be/index.php/s/8KtT3HxDHybsNmt). 

__Step 1: snapshot content extraction__ 

The [wikitext](https://en.wikipedia.org/wiki/Help:Wikitext) content is parsed
 for each of the temporal snapshots. 
The hyperparameter configuration is located in 
```experiments/snapshot_extraction/snap_20220515/config/s01_config_content.json``` file. Execute: 
```
./scripts/dataset/snapshot_content_extraction.sh
```
The script above will generate the ```experiments/snapshot_extraction/snap_20220515/output/wikipedia_evolution_content.jsonl``` 
file with cleaned wikipedia content for each of the entities in snapshots. 
It will also create a number of _stat_ files in ```experiments/snapshot_extraction/snap_20220515/output/stats/snapshot_stats/``` 
directory that are later use to 
calculate attributes (such as the _mention prior_) used during the creation of TempEL dataset. 
These files are divided in three categories:
1. ```page_info_yyyy-01-01T00:00:00.csv```: information necessary to connect Wikipedia _page_id_ to _wikipedia_title_ and 
_wikidata_qid_ with some additional attributes such as the creation date, the last revision date and the content_length
of the page for snapshot of the year ```yyyy```. 
2. ```page_link_stats_yyyy-01-01T00:00:00.csv```: contains the details of _anchor_ mention and _target_ entities
these mentions are linked to (using both Wikipedia title as well as Wikidata QID identifiers) for the snapshot of the year 
```yyyy```. 
3. ```title_changes.csv```: contains the temporal title changes of Wikipedia pages. This allows to 
identify mentions linked to the same entity even if the title of this one is changed at some point in time.

__--nr_threads_processor__ and __--nr_threads_reader__: these parameters define the nr of parallel threads used to 
process the files in ```data/wikipedia_dump/enwiki_20220201/pages_meta_history/```. The _--nr_threads_reader_ defines
the number of parallel threads that read each of the files in the input directory. The _--nr_threads_processor_ defines the 
number of parallel threads that process the entity history. 
For ideal performance, the sum of _--nr_threads_processor_ and _--nr_threads_reader_ should not exceed the total 
number of available CPUs. 

__Step 2.a: inverse dictionary and mention-driven statistics generation__

In this step, we generate inverse dictionary index that maps a target entities to Wikipedia
 pages with mentions linked to those target entities. Additional mention-driven statistics (e.g., 
 mention prior, edit distance between mention and target entity title, etc.) are also generated and 
 serialized to disc as .csv files. The detailed configuration of the hyperparameters used for this 
 step is located in the ```experiments/snapshot_extraction/snap_20220515/config/s02_alias_table_generator.json``` 
 file. 
 The following is the python command to generate the above mentioned inverse dictionary and mention-driven 
 statistics: 
 ```
./scripts/dataset/detect_mentions.sh
 ```

__Step 2.b: detecting redirects in page history__

Some Wikipedia pages are redirect pages in one or more of the snapshots. We do not include these page in 
TempEL dataset, since we are interested in pages with actual content and not redirects pointing to other 
pages. The following python command generates output file containing Wikidata QID and Wikipedia title
of such redirect pages:
```
./scripts/dataset/detect_redirects.sh
```  

__Alternative: download files from the cloud__ 

It can take multiple days for the scripts in __Step 1__ and __Step 2__
 to complete and generate all the files. Alternatively, these files can also be downloaded from our cloud 
executing:

```
./scripts/dataset/download_extracted_snapshots.sh
```
The files can also be accessed via the following link: 
[https://cloud.ilabt.imec.be/index.php/s/Ytt2WDTJH5r3w4z](https://cloud.ilabt.imec.be/index.php/s/Ytt2WDTJH5r3w4z)

__Step 3: TempEL dataset generation__

Finally, TempEL dataset is generated with the parameters defined in 
```experiments/dataset_creation/data_20220515/config/s03_dataset_creator.json```. 
The following is the python command:

```
./scripts/dataset/generate_dataset.sh
```

__Note 1__ set ```nr_threads``` to be close to the number of the available CPUs.

__Note 2__ the selection of the entities and mentions to produce TempEL dataset is random. Therefore, the produced
version of TempEL dataset will differ to the one used in the paper available in 
[this link](https://cloud.ilabt.imec.be/index.php/s/RinXy8NgqdW58RW).


### 2.2 Models
We use the bi-encoder BLINK model [Wu et al., 2020](https://aclanthology.org/2020.emnlp-main.519/) as a baseline 
in our paper. We use [faiss](https://github.com/facebookresearch/faiss) facebook library for fast 
candidate entities retrieval.

#### 2.2.1 Training
The training was performed on TempEL dataset, which can be downloaded using the following
command:
```
./scripts/dataset/download_tempel.sh
```

We train a separate bi-encoder model for each of the 10 temporal snapshots of TempEL. 
The training is performed on 4 parallel V100 GPUs. The following command will start the training process and 
expand automatically to all the available GPUs:

```
./scripts/biencoder/training_script.sh train_20230112_from_cloud
```
The shared training hyperparameters in ```experiments/models/blink/biencoder/train/train_20230112_from_cloud/config/```
are divided in:
 1. Hyperparameters common to all the snapshots: ```s04_config_train_parent.json```. 
 2. Hyperaparameters specific to each of the snapshots: ```s04_config_train_yyyy.json```, with ```yyyy```
being the year of the snapshot. 

The trained models used to report the results in our work can be downloaded from 
[this link](https://cloud.ilabt.imec.be/index.php/s/i7dWxc4Fkw7cKcC), 
or alternatively using the following command, which will copy the models into 
```experiments/models/blink/biencoder/train/train_bi_20220630/output/models_ep_9_only/``` directory:

```
./scripts/biencoder/download_models.sh
```

#### 2.2.2 Encoding 
Each of the 10 trained models (one for each temporal snapshot of TempEL) in the previous subsection, is used to 
encode the entities from all the 10 Wikipedia snapshots in order to compare the temporal drift in performance 
of the models (see Table 2 of the [paper](https://openreview.net/pdf?id=vrnqr3PG4yB)). This results in a total of
100 encoded entity representation
tables (__2.2 TB__). The following command will start the process, executing the models configured in the
hyperparameter files located in ```experiments/models/blink/biencoder/encode/20220630/config``` directory
(the command below uses the models in 
```experiments/models/blink/biencoder/train/train_bi_20220630/output/models_ep_9_only/```
directory):
```
./scripts/biencoder/encoding_script.sh 20220630
```

The encodings can also be downloaded from [this link](https://cloud.ilabt.imec.be/index.php/s/DSiKXQwWpQacRJ6), 
or alternatively using the following command, 
which will copy the encoded entities into ```experiments/models/blink/biencoder/encode/20220630/output/faiss/``` directory:
<!---
# mkdir -p experiments/models/blink/biencoder/encode/20220630_test/output/faiss/
# TODO - here try https://cloud.ilabt.imec.be/index.php/apps/files/?dir=/copies/cope_nlu/el-copenhagen-cleaned/experiments/models/blink/biencoder/encode/encode_test_download/output/faiss&fileid=1121661483
# wget -O experiments/models/blink/biencoder/encode/20220630_test/output/faiss/download.zip https://cloud.ilabt.imec.be/index.php/s/C9CQ7TTq4FidB5c/download
# where a small sized encoding (maybe only the first encoded snapshot of the model of the year 2014) has to be copied. 
# unzip experiments/models/blink/biencoder/encode/20220630_test/output/faiss/download.zip -d experiments/models/blink/biencoder/encode/20220630_test/output/ 
-->
```
./scripts/biencoder/download_entity_representations.sh
```

__Note:__ The 100 encoded entity representation tables mentioned above are 2.2 TB in size. 

#### 2.2.3 Evaluation
During the evaluation step, each of the models (one for each temporal snapshot) is evaluated on TempEL dataset. 
The predictions are saved in json files inside ```models/blink/biencoder/evaluate/20220630/output/predictions/```
directory (configured in ```experiments/models/blink/biencoder/evaluate/20220630/config``` files). 
The following script will run the evaluation, 
taking the entity encodings from ```experiments/models/blink/biencoder/encode/20220630/output/faiss/``` directory
created in the previous step.
<!--
./scripts/biencoder/evaluation_script.sh 20220630_test
-->
```
./scripts/biencoder/evaluation_script.sh 20220630
```
We tested the script above on one 32 GB V100 GPU. 

#### 2.2.4 Metrics
The following script calculates accuracy@64 metric of the model predictions: 
<!--
./scripts/stats/metrics_script.sh 20220630_test  
-->
```
./scripts/stats/metrics_script.sh 20220630  
```

#### 2.2.5 Statistics
The script to calculate the statistics reported in the paper will be available shortly, 
stay tuned!