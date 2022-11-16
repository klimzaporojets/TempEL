import argparse
import json
import logging
import os
import pickle
import time
import traceback
from typing import List

import git
import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import SequentialSampler, DataLoader

from models.biencoder.build_faiss_index_passages import faiss_index_passages_single_date_cut
from models.biencoder.data_process import get_candidate_passage_representation, get_mentions_from_dataset_wiki_cuts
from models.biencoder.misc_utils import get_passages_bert_tokens_input, Stats, generate_and_save_json
from models.biencoder.nn_prediction import get_topk_faiss_predictions_passages
from tempel_creation.misc.article_queue import ArticleReadingQueue
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def display_d(g_candidate_encodings: List, process_candidate_encoders: List, v_nr_entities, arq: ArticleReadingQueue,
              start_time):
    while True:
        try:
            logger.info('\tDISPLAY Size of queue process_files_queue: %s' % arq.process_files_queue.qsize())
            logger.info('\tDISPLAY Size of queue gatherer_encodings_queue: %s' % arq.gatherer_encodings_queue.qsize())

            is_alive = 0
            is_dead = 0
            for curr_process in process_candidate_encoders:
                if curr_process.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1

            logger.info('\tDISPLAY NR candidate encoders processors alive: %s and dead: %s' % (is_alive, is_dead))

            is_alive = 0
            is_dead = 0
            for curr_process in g_candidate_encodings:
                if curr_process.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1

            logger.info('\tDISPLAY NR gather processes alive: %s and dead: %s' % (is_alive, is_dead))
            # print('\tDISPLAY TOT encoded entities (if in 0, uncomment the code that sums!!): ', v_nr_entities)

            logger.info('nr of encoded entities: %s' % v_nr_entities.value)
            time_lapse_mins = (time.time() - start_time) / 60
            logger.info('Avg. nr of entities per minute: %s' % ((v_nr_entities.value / time_lapse_mins)))
            logger.info('minutes that passed: %s' % time_lapse_mins)

            # time.sleep(60 * 1)
            time.sleep(1 * 30)
        except Exception as e:
            logger.error('type error in display: %s with stack: %s' % (str(e)), traceback.format_exc())
            continue


class Gatherer:

    def __init__(self, dim_vector, initial_capacity=1000000, increment_capacity=100000000):
        self.capacity = initial_capacity
        self.data = np.zeros((self.capacity,), dtype=np.float32)
        self.increment_capacity = increment_capacity
        self.size = 0
        self.dim_vector = dim_vector

    def update(self, row):
        # for r in row:
        #     self.add(r)
        self.add(row)

    def add(self, to_add):
        if self.size + self.dim_vector >= self.capacity:
            self.capacity += self.increment_capacity
            newdata = np.zeros((self.capacity,), dtype=np.float32)
            newdata[:self.data.shape[0]] = self.data
            self.data = newdata

        self.data[self.size:self.size + self.dim_vector] = to_add
        self.size += self.dim_vector

    def finalize(self):
        data = self.data[:self.size]
        return np.reshape(data, newshape=(int(len(data) / self.dim_vector), self.dim_vector))


def get_token_ids(parsed_token_file, curr_date_cut, tokenizer, config):
    if not 'bert_tokenized' in parsed_token_file['evolution'][curr_date_cut]:
        curr_version_text = parsed_token_file['evolution'][curr_date_cut]['cleaned_text']
        b_curr_version_text = tokenizer.tokenize(curr_version_text)
    else:
        b_curr_version_text = parsed_token_file['evolution'][curr_date_cut]['bert_tokenized']
    len_title = 0
    b_curr_version_title = []
    if 'bert_wikipedia_title' not in parsed_token_file['evolution'][curr_date_cut]:
        curr_version_title = parsed_token_file['evolution'][curr_date_cut]['wikipedia_title']
        b_curr_version_title = tokenizer.tokenize(curr_version_title)
    else:
        b_curr_version_title = parsed_token_file['evolution'][curr_date_cut][
            'bert_wikipedia_title']
        len_title = len(b_curr_version_title)

    text_passages = get_passages_bert_tokens_input(b_curr_version_text,
                                                   config['max_cand_length'] - 2 - len_title)
    rep = get_candidate_passage_representation(text_passages[0], tokenizer, config['max_cand_length'],
                                               b_curr_version_title)
    tensor_rep = torch.LongTensor(rep['ids'])
    tensor_length = torch.IntTensor([rep['length']])
    # return tensor_rep, rep['length']
    return tensor_rep, tensor_length


def get_candidate_encodings_parallel(params, start_time, v_nr_entities, arq: ArticleReadingQueue, v_shutdown_encoders,
                                     tokenizer):
    while not (v_shutdown_encoders.value == 1 and arq.process_files_queue.empty()):
        try:
            curr_filename, curr_file_path, curr_token_path = arq.process_files_queue.get(block=True, timeout=1)
            parsed_file = json.load(open(curr_file_path, 'rt'))
            parsed_token_file = json.load(open(curr_token_path, 'rt'))
            wikidata_qid = parsed_file['wikidata_qid']
            for curr_time_cut, curr_representation in parsed_file['evolution'].items():
                bert_tokens, bert_tokens_length = \
                    get_token_ids(parsed_token_file, curr_time_cut, tokenizer, params)

                curr_tensor = torch.FloatTensor(curr_representation)
                arq.gatherer_encodings_queue.put((curr_time_cut, wikidata_qid, curr_tensor.unsqueeze(0).
                                                  detach().cpu().numpy(),
                                                  bert_tokens.detach().cpu().numpy(),
                                                  bert_tokens_length.detach().cpu().numpy()))
            v_nr_entities.value += 1
        except Exception as e:
            logger.error('type error get_candidate_encodings_parallel: %s with stack: %s' %
                         (str(e), traceback.format_exc()))


if __name__ == '__main__':
    # gets the current git hash and puts it in the directory
    # import pdb;
    #
    # pdb.set_trace()
    start = time.time()

    gpu_id = 0

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        # TODO
                        default='TODO',
                        help='The config file that contains all the parameters')
    args = parser.parse_args()
    logger.info('evaluating faiss biencoder blink: %s' % str(args))
    config = json.load(open(args.config_file, 'rt'))

    repo = git.Repo(search_parent_directories=True)
    git_commit_hash = repo.head.object.hexsha
    config['git_commit_hash'] = git_commit_hash

    base_experiment_path = config['base_experiment_path']
    output_dir_predictions = os.path.join(base_experiment_path, config['output_dir_predictions'])
    os.makedirs(output_dir_predictions, exist_ok=True)
    nr_dump = 0

    output_config_path = os.path.join(output_dir_predictions,
                                      'config_eval_faiss_biencoder_blink_%04d.json' % (nr_dump,))

    while os.path.isfile(output_config_path):
        nr_dump += 1
        output_config_path = os.path.join(output_dir_predictions,
                                          'config_eval_faiss_biencoder_blink_%04d.json' % (nr_dump,))

    with open(output_config_path, 'wt') as out_json:
        json.dump(config, out_json, indent=4, sort_keys=True)

    nr_threads = config['nr_threads']
    initial_capacity = config['initial_capacity']
    increment_capacity = config['increment_capacity']

    logger.info('Loading the candidates and creating them if necessary')

    tokenizer = BertTokenizer.from_pretrained(
        config['bert_model'], do_lower_case=config['lowercase'], cache_dir=config['bert_cache_dir']
    )

    already_serialized = True
    candidate_serialization_path = None  # TODO!!

    cand_faiss_path = os.path.join(base_experiment_path, config['cand_faiss_path'])

    # print('---------FINISHED SERIALIZING-------------')

    # loads the serialized tensors into faiss
    date_cut_to_label_id_to_wikidata_qid = dict()
    date_cut_to_wikidata_qid_to_label_id = dict()
    date_cut_to_tensor_shape = dict()
    date_cut_to_candidate_encodings = dict()
    date_cut_to_candidate_token_ids = dict()

    logger.info('loading mappings ... ')
    begin_mappings = time.time()
    for curr_date_cut in config['time_cut_list']:

        curr_output_token_ids_path = os.path.join(cand_faiss_path,
                                                  '{}_tensors_token_ids.pt'.format(
                                                      curr_date_cut['entities_cut']))
        curr_cut_input_path = os.path.join(cand_faiss_path,
                                           '{}_label_id_to_wikidata_qid_mappings.pt'.format(
                                               curr_date_cut['entities_cut']))

        if config['debug']:
            # just creates/emulates the content locally, no need of files to exist on the disk
            loaded_candidate_encodings = dict()
            curr_cut_mappings = dict()
            debug_size = config['debug_size']
            curr_cut_mappings['label_id_to_wikidata_qid'] = {k: 'Q{}'.format(k) for k in range(debug_size)}
            curr_cut_mappings['wikidata_qid_to_label_id'] = {'Q{}'.format(k): k for k in range(debug_size)}
            # curr_cut_mappings['embeddings_tensor_shape'] = (debug_size, 1024)
            curr_cut_mappings['embeddings_tensor_shape'] = (debug_size, 768)
            tensors_tok_ids = torch.randint(low=0, high=1000, size=(debug_size, 128))
            date_cut_to_candidate_token_ids[curr_date_cut['entities_cut']] = tensors_tok_ids
        else:
            logger.info('loading mappings from %s' % curr_cut_input_path)
            loaded_candidate_encodings = pickle.load(open(curr_output_token_ids_path, 'rb'))
            tensors_tok_ids = loaded_candidate_encodings['tensors_tok_ids'].astype(int)
            tensors_tok_ids = torch.from_numpy(tensors_tok_ids).int()
            date_cut_to_candidate_token_ids[curr_date_cut['entities_cut']] = tensors_tok_ids
            curr_cut_mappings = pickle.load(open(curr_cut_input_path, 'rb'))

        # TODO - this has to be loaded FROM A SEPARATE FILE Inside the faiss loading function ONLY if faiss wasn't
        #  loaded
        # candidate_per_date_cut_encodings[curr_date_cut] = curr_cut_encodings['tensors']
        date_cut_to_label_id_to_wikidata_qid[curr_date_cut['entities_cut']] = curr_cut_mappings[
            'label_id_to_wikidata_qid']
        date_cut_to_wikidata_qid_to_label_id[curr_date_cut['entities_cut']] = curr_cut_mappings[
            'wikidata_qid_to_label_id']
        date_cut_to_tensor_shape[curr_date_cut['entities_cut']] = curr_cut_mappings['embeddings_tensor_shape']

        # print('shape of tensors for date cut ', curr_date_cut, ' is: ', curr_cut_mappings['tensors'].shape)
        logger.info('length of label_id_to_wikidata_qid for date cut %s is: %s' %
                    (curr_date_cut, len(curr_cut_mappings['label_id_to_wikidata_qid'])))
        logger.info('length of wikidata_qid_to_label_id for date cut %s is: %s' %
                    (curr_date_cut, len(curr_cut_mappings['wikidata_qid_to_label_id'])))
    end_mappings = time.time()
    logger.info('mappings loaded in ... %s minutes. ' % ((end_mappings - begin_mappings) / 60))

    begin_model = time.time()
    dict_samples = dict()
    dict_tensor_data = dict()

    for curr_date_cut in config['time_cut_list']:
        logger.info('getting mentions from %s' % curr_date_cut)
        dict_samples[curr_date_cut['mentions_cut']] = dict()
        dict_tensor_data[curr_date_cut['mentions_cut']] = dict()
        for curr_subset in config['evaluation_subsets']:
            test_samples, test_tensor_data = \
                get_mentions_from_dataset_wiki_cuts(subset_name=curr_subset,
                                                    label_id_to_wikidata_qid=date_cut_to_label_id_to_wikidata_qid,
                                                    wikidata_qid_to_label_id=date_cut_to_wikidata_qid_to_label_id,
                                                    date_cut=curr_date_cut,
                                                    # biencoder=biencoder,
                                                    tokenizer=tokenizer,
                                                    params=config,
                                                    is_training=False, gpu_id=gpu_id)

            dict_samples[curr_date_cut['mentions_cut']][curr_subset] = test_samples
            dict_tensor_data[curr_date_cut['mentions_cut']][curr_subset] = test_tensor_data
            # test_samples --> list as loaded from dataset files not encoded in any way
            #
            #
            # test_tensor_data --> TensorDataset object consisting of tuples
            #
            #
        logger.info('END getting mentions for subsets from %s' % curr_date_cut)

    # TODO - too memory heavy to load all the faiss embeddings of ALL the wikipedia versions (cuts), around 500 GB in prod
    # faiss_date_cut_to_index = faiss_index_passages(config, candidate_per_date_cut_encodings, 0)

    stats = dict()
    candidate_per_date_cut_encodings = dict()
    for curr_date_cut in config['time_cut_list']:
        logger.info('indexing passages')

        # do the loading of each cut individuall since it is too memory heavy to load all the faiss embeddings of ALL
        # the wikipedia versions (cuts), around 500 GB in prod
        # So, first erase the dictionary in each loop
        faiss_date_cut_to_index = dict()  # renew the dict, too memory heavy if not
        start_loading_candidates = time.time()

        faiss_date_cut_to_index[curr_date_cut['entities_cut']] = \
            faiss_index_passages_single_date_cut(config, curr_date_cut['entities_cut'],
                                                 date_cut_to_tensor_shape[curr_date_cut['entities_cut']])
        end_loading_candidates = time.time()

        logger.info('%s minutes to load the candidate encodings' %
                    ((end_loading_candidates - start_loading_candidates) / 60))
        stats[curr_date_cut['mentions_cut']] = dict()
        for curr_subset in config['evaluation_subsets']:
            test_sampler = SequentialSampler(dict_tensor_data[curr_date_cut['mentions_cut']][curr_subset])

            test_dataloader = DataLoader(
                dict_tensor_data[curr_date_cut['mentions_cut']][curr_subset],
                sampler=test_sampler,
                batch_size=config['eval_batch_size']
            )
            save_predictions = config.get('save_topk_result')
            stats[curr_date_cut['mentions_cut']][curr_subset] = Stats(config['top_k'])
            topk_predictions = get_topk_faiss_predictions_passages(faiss_date_cut_to_index,
                                                                   # date_cut_to_candidate_encodings,
                                                                   test_dataloader, config,
                                                                   date_cut=curr_date_cut,
                                                                   subset=curr_subset,
                                                                   label_id_to_candidate_token_ids=date_cut_to_candidate_token_ids,
                                                                   stats=stats,
                                                                   save_predictions=save_predictions)
            if save_predictions:
                save_data_dir = os.path.join(output_dir_predictions, 'encoded', curr_subset)
                os.makedirs(save_data_dir, exist_ok=True)

                save_predictions_path = os.path.join(save_data_dir,
                                                     'm%s_e%s_top%d_candidates.t7' %
                                                     (curr_date_cut['mentions_cut'],
                                                      curr_date_cut['entities_cut'],
                                                      config['top_k']))
                # '%s_top%d_candidates.t7' % (curr_date_cut, config['top_k']))
                logger.info('saving tensors to %s' % save_predictions_path)
                torch.save(topk_predictions, save_predictions_path)

                if config['json_output']:
                    generate_and_save_json(topk_predictions,
                                           dict_samples[curr_date_cut['mentions_cut']][curr_subset],
                                           config, curr_subset, curr_date_cut,
                                           date_cut_to_label_id_to_wikidata_qid)
