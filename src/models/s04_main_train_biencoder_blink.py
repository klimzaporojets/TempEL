import argparse
import json
import logging
import os

import git
import torch
import torch.multiprocessing as mp
from pytorch_transformers.tokenization_bert import BertTokenizer

from models.biencoder.data_process import process_mention_data
from models.utils.s04_main_train_biencoder_blink_utils import main_train_biencoder
from models.utils.utils import read_dataset
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # import pdb;
    #
    # pdb.set_trace()

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        # TODO 
                        default='TODO',
                        help='The config file that contains all the parameters')

    parser.add_argument('--nr_gpus', type=int, default=-1, help='Nr. of gpus to run on.')

    args = parser.parse_args()
    logger.info('training biencoder blink: %s' % str(args))
    params = json.load(open(args.config_file, 'rt'))

    repo = git.Repo(search_parent_directories=True)
    git_commit_hash = repo.head.object.hexsha
    nr_gpus = args.nr_gpus

    if nr_gpus == -1:
        nr_gpus = torch.cuda.device_count()

    if 'path_to_config' in params and params['path_to_config'] is not None:
        config_file_path = params['path_to_config']
        params = json.load(open(config_file_path))

    repo = git.Repo(search_parent_directories=True)
    git_commit_hash = repo.head.object.hexsha
    os.makedirs(params['output_path'], exist_ok=True)
    params['git_commit_hash'] = git_commit_hash

    params['is_training'] = True

    base_experiment_path = params['base_experiment_path']

    data_path = os.path.join(base_experiment_path, params['data_path'])
    output_path = os.path.join(base_experiment_path, params['output_path'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_dir_logger = os.path.dirname(output_path)

    train_samples = read_dataset(params['train_dataset'], data_path, params['time_cut'])
    logger.info('Read %d train samples.' % len(train_samples))
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=params['lowercase'], cache_dir=params['bert_cache_dir']
    )

    train_data, train_tensor_data = process_mention_data(
        train_samples,
        tokenizer,
        params['max_context_length'],
        params['max_cand_length'],
        silent=params['silent'],
        debug=params['debug'],
    )
    valid_samples = read_dataset(params['valid_dataset'], data_path, params['time_cut'])
    logger.info('Read %d valid samples. ' % len(valid_samples))

    valid_data, valid_tensor_data = process_mention_data(
        valid_samples,
        tokenizer,
        params['max_context_length'],
        params['max_cand_length'],
        silent=params['silent'],
        debug=params['debug'],
    )

    if params['data_parallel']:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        world_size = nr_gpus
        mp.spawn(main_train_biencoder,
                 args=(params, world_size, train_tensor_data, valid_tensor_data, base_experiment_path,
                       tokenizer),
                 nprocs=world_size,
                 join=True)
    else:
        main_train_biencoder(gpu_id=0, params=params, world_size=0, train_tensor_data=train_tensor_data,
                             valid_tensor_data=valid_tensor_data,
                             base_experiment_path=base_experiment_path,
                             tokenizer=tokenizer)
