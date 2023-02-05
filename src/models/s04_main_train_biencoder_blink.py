import argparse
import json
import logging
import os

import git
import torch
import torch.multiprocessing as mp
from pytorch_transformers.tokenization_bert import BertTokenizer

from src.models.biencoder.data_process import process_mention_data
from src.models.utils.s04_main_train_biencoder_blink_utils import main_train_biencoder
from src.models.utils.utils import read_dataset
from src.utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_parent_path', required=False, type=str,
                        default='experiments/models/blink/biencoder/train/train_20230112_from_cloud/config/'
                                's04_config_train_parent.json',
                        help='Config file with generic experiment hyperparameters shared among all the temporal '
                             'snapshots')
    parser.add_argument('--config_path', required=True, type=str,
                        default='experiments/models/blink/biencoder/train/train_20230112_from_cloud/config/'
                                's04_config_train_2013.json',
                        help='Config file with a particular experiment hyperparameters')

    parser.add_argument('--nr_gpus', type=int, default=-1, help='Nr. of gpus to run on.')

    args = parser.parse_args()
    logger.info('training biencoder blink: %s' % str(args))
    # arg_params = json.load(open(args.config_path, 'rt'))

    repo = git.Repo(search_parent_directories=True)
    git_commit_hash = repo.head.object.hexsha
    dict_args = vars(args)
    nr_gpus = args.nr_gpus

    if nr_gpus == -1:
        nr_gpus = torch.cuda.device_count()

    config_parent_params = dict()
    config_file_path = dict_args['config_path']
    config_params = json.load(open(config_file_path))

    if 'config_parent_path' in dict_args and dict_args['config_parent_path'] is not None:
        config_file_path = dict_args['config_parent_path']
        config_parent_params = json.load(open(config_file_path))

    # config_params = config_params.update(config_parent_params)
    config_params = {**config_params, **config_parent_params}

    repo = git.Repo(search_parent_directories=True)
    git_commit_hash = repo.head.object.hexsha
    os.makedirs(config_params['output_path'], exist_ok=True)
    config_params['git_commit_hash'] = git_commit_hash

    config_params['is_training'] = True

    base_experiment_path = config_params['base_experiment_path']

    train_data_path = os.path.join(base_experiment_path, config_params['train_data_path'])
    valid_data_path = os.path.join(base_experiment_path, config_params['val_data_path'])
    output_path = os.path.join(base_experiment_path, config_params['output_path'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_dir_logger = os.path.dirname(output_path)

    train_samples = read_dataset(train_data_path, config_params['train_file'])
    logger.info('Read %d train samples.' % len(train_samples))
    tokenizer = BertTokenizer.from_pretrained(
        config_params['bert_model'], do_lower_case=config_params['lowercase'], cache_dir=config_params['bert_cache_dir']
    )

    train_data, train_tensor_data = process_mention_data(
        train_samples,
        tokenizer,
        config_params['max_context_length'],
        config_params['max_cand_length'],
        silent=config_params['silent'],
        debug=config_params['debug'],
    )
    valid_samples = read_dataset(valid_data_path, config_params['val_file'])
    logger.info('Read %d valid samples. ' % len(valid_samples))

    valid_data, valid_tensor_data = process_mention_data(
        valid_samples,
        tokenizer,
        config_params['max_context_length'],
        config_params['max_cand_length'],
        silent=config_params['silent'],
        debug=config_params['debug'],
    )

    if config_params['data_parallel']:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        world_size = nr_gpus
        mp.spawn(main_train_biencoder,
                 args=(config_params, world_size, train_tensor_data, valid_tensor_data, base_experiment_path,
                       tokenizer),
                 nprocs=world_size,
                 join=True)
    else:
        main_train_biencoder(gpu_id=0, params=config_params, world_size=0, train_tensor_data=train_tensor_data,
                             valid_tensor_data=valid_tensor_data,
                             base_experiment_path=base_experiment_path,
                             tokenizer=tokenizer)
