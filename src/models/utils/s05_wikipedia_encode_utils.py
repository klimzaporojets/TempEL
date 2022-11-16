import json
import logging
import os
import pickle
import random
import time
import traceback
from typing import List, Set, Dict

import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import SequentialSampler, DataLoader

from models.biencoder.biencoder import BiEncoder, to_bert_input, NULL_IDX
from models.biencoder.data_process import get_candidate_passage_representation
from models.biencoder.misc_utils import load_model, get_passages_bert_tokens_input
from tempel_creation.misc.article_queue import ArticleReadingQueue
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def copy_tensors(candidate_embeddings_per_date_cut, candidate_token_ids_per_date_cut,
                 cand_faiss_path, label_id_to_wikidata_qid, wikidata_qid_to_label_id, nr_gathered,
                 initial_capacity, increment_capacity, nr_gathered_per_date_cut):
    logger.info('saving checkpoint at nr_gathered of %s' % nr_gathered)
    start_copy_tensors = time.time()
    date_cuts = list(candidate_embeddings_per_date_cut.keys())
    for curr_date_cut in date_cuts:
        curr_output_embeddings_path = os.path.join(cand_faiss_path, '{}_tensors.pt'.format(curr_date_cut))
        curr_output_token_ids_path = os.path.join(cand_faiss_path, '{}_tensors_token_ids.pt'.format(curr_date_cut))
        curr_output_label_mappings_path = os.path.join(cand_faiss_path,
                                                       '{}_label_id_to_wikidata_qid_mappings.pt'.format(curr_date_cut))
        start = time.time()
        curr_embedding_tensor = candidate_embeddings_per_date_cut[curr_date_cut].finalize()
        curr_token_ids_tensor = candidate_token_ids_per_date_cut[curr_date_cut].finalize()
        end = time.time()
        logger.info('finalize finished in %s minutes' % ((end - start) / 60))

        # concatenates with already existing embeddings
        if os.path.exists(curr_output_embeddings_path):
            loaded_embeddings = pickle.load(open(curr_output_embeddings_path, 'rb'))
            loaded_embeddings = loaded_embeddings['tensors']
            logger.info('-----------')
            logger.info('%s dimensions of existing loaded_embeddings.shape: %s' %
                        (curr_date_cut, str(loaded_embeddings.shape)))
            logger.info('%s dimensions of current candidate_embeddings_per_date_cut.shape: %s' %
                        (curr_date_cut, str(curr_embedding_tensor.shape)))
            curr_accumulated_candidate_embeddings = np.concatenate([loaded_embeddings, curr_embedding_tensor])
            logger.info('%s curr_accumulated_candidate_embeddings.shape (result of np.concatenate): %s' %
                        (curr_date_cut, str(curr_accumulated_candidate_embeddings.shape)))
            logger.info('-----------')
        else:
            curr_accumulated_candidate_embeddings = curr_embedding_tensor

        candidate_embeddings_per_date_cut[curr_date_cut] = Gatherer(dim_vector=curr_embedding_tensor.shape[1],
                                                                    initial_capacity=initial_capacity,
                                                                    increment_capacity=increment_capacity)

        logger.info('%s expected length of curr_accumulated_candidate_embeddings.shape: %s' %
                    (curr_date_cut, str(nr_gathered_per_date_cut[curr_date_cut])))
        logger.info('%s length of wikidata_qid_to_label_id[curr_date_cut]: %s ' %
                    (curr_date_cut, len(wikidata_qid_to_label_id[curr_date_cut])))

        if nr_gathered_per_date_cut[curr_date_cut] != curr_accumulated_candidate_embeddings.shape[0]:
            logger.warning('!!!WARNING V10!!!: nr_gathered_per_date_cut[curr_date_cut] != '
                           'curr_accumulated_candidate_embeddings.shape[0] %s vs %s' %
                           (nr_gathered_per_date_cut[curr_date_cut], curr_accumulated_candidate_embeddings.shape[0]))

        if nr_gathered_per_date_cut[curr_date_cut] != len(wikidata_qid_to_label_id[curr_date_cut]):
            logger.warning('!!!WARNING V11!!!: nr_gathered_per_date_cut[curr_date_cut] != '
                           'len(wikidata_qid_to_label_id[curr_date_cut]) %s vs %s' %
                           (nr_gathered_per_date_cut[curr_date_cut], len(wikidata_qid_to_label_id[curr_date_cut])))

        # concatenates with already existing token ids
        if os.path.exists(curr_output_token_ids_path):
            loaded_tok_ids = pickle.load(open(curr_output_token_ids_path, 'rb'))
            loaded_tok_ids = loaded_tok_ids['tensors_tok_ids']
            logger.info('-----------')
            logger.info('%s dimensions of existing loaded_tok_ids.shape: %s' %
                        (curr_date_cut, str(loaded_tok_ids.shape)))
            logger.info('%s dimensions of current candidate_token_ids_per_date_cut.shape: %s' %
                        (curr_date_cut, str(curr_token_ids_tensor.shape)))
            curr_accumulated_candidate_token_ids = np.concatenate([loaded_tok_ids, curr_token_ids_tensor])
            logger.info('%s curr_accumulated_candidate_token_ids.shape (result of np.concatenate): %s' %
                        (curr_date_cut, str(curr_accumulated_candidate_token_ids.shape)))
            logger.info('-----------')
        else:
            curr_accumulated_candidate_token_ids = curr_token_ids_tensor

        candidate_token_ids_per_date_cut[curr_date_cut] = Gatherer(dim_vector=curr_token_ids_tensor.shape[1],
                                                                   initial_capacity=initial_capacity,
                                                                   increment_capacity=increment_capacity)

        logger.info('saving checkpoint of candidate tensors to the following temporary directories: %s  %s  %s' %
                    (curr_output_embeddings_path, curr_output_token_ids_path,
                     curr_output_label_mappings_path))
        logger.info('saving tensor of shape %s to %s' %
                    (str(curr_accumulated_candidate_embeddings.shape), curr_output_embeddings_path))
        logger.info('saving token_ids of shape %s  to %s' %
                    (str(curr_accumulated_candidate_token_ids.shape), curr_output_token_ids_path))
        start = time.time()

        logger.info('%s asserting len(wikidata_qid_to_label_id) of %s'
                    'to be equal to curr_accumulated_candidate_embeddings.shape[0] of %s' %
                    (curr_date_cut, len(wikidata_qid_to_label_id[curr_date_cut]),
                     curr_accumulated_candidate_embeddings.shape[0]))
        # this assert is needed, they HAVE TO MATCH, if not something is VERY wrong!!
        assert len(wikidata_qid_to_label_id[curr_date_cut]) == curr_accumulated_candidate_embeddings.shape[0]
        assert len(label_id_to_wikidata_qid[curr_date_cut]) == curr_accumulated_candidate_embeddings.shape[0]
        pickle.dump({'label_id_to_wikidata_qid': label_id_to_wikidata_qid[curr_date_cut],
                     'wikidata_qid_to_label_id': wikidata_qid_to_label_id[curr_date_cut],
                     'embeddings_tensor_shape': curr_accumulated_candidate_embeddings.shape},
                    open(curr_output_label_mappings_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump({'tensors_tok_ids': curr_accumulated_candidate_token_ids},
                    open(curr_output_token_ids_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump({'tensors': curr_accumulated_candidate_embeddings},
                    open(curr_output_embeddings_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        end = time.time()
        logger.info('pickle.dump to save tensors finished in %s minutes' % ((end - start) / 60))
        # now copies the tmp to permanent locations
        #
        # too expensive, lets not do this
        # logger.info('now copying to the permanent locations')
        # start = time.time()
        # shutil.copy(tmp_curr_output_embeddings_path, curr_output_embeddings_path)
        # shutil.copy(tmp_curr_output_token_ids_path, curr_output_token_ids_path)
        # shutil.copy(tmp_curr_output_label_mappings_path, curr_output_label_mappings_path)
        # end = time.time()
        # logger.info('copying from temp to permanent locations done in %s minutes' % ((end - start) / 60))

    end_copy_tensors = time.time()

    logger.info('time to make a checkpoint of all the tensors (copy_tensors) for nr_gathered %s'
                ': %s minutes. ' % (nr_gathered, ((end_copy_tensors - start_copy_tensors) / 60)))
    return candidate_embeddings_per_date_cut, candidate_token_ids_per_date_cut


class Gatherer:

    def __init__(self, dim_vector, initial_capacity=1000000, increment_capacity=100000000):
        self.capacity = initial_capacity
        self.data = np.zeros((self.capacity,), dtype=np.float32)
        self.increment_capacity = increment_capacity
        self.size = 0
        self.dim_vector = dim_vector

    def update(self, row):
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


def get_candidate_encodings_parallel(params, gpu_id, start_time, v_nr_entities, arq: ArticleReadingQueue,
                                     v_shutdown_encoders, already_processed_qids: Set,
                                     wikidata_qid_to_label_id: Dict):
    logger.info('-------------in get_candidate_encoding_parallel------------------')
    logger.info('get_candidate_encoding_parallel.params: %s' % str(params))
    logger.info('get_candidate_encoding_parallel.gpu_id: %s' % gpu_id)
    logger.info('get_candidate_encoding_parallel.start_time: %s' % start_time)
    logger.info('get_candidate_encoding_parallel.v_nr_entities: %s' % v_nr_entities)
    logger.info('get_candidate_encoding_parallel.arq: %s' % arq)
    logger.info('get_candidate_encoding_parallel.v_shutdown_encoders: %s' % v_shutdown_encoders)
    logger.info('get_candidate_encoding_parallel.len(already_processed_qids): %s' % len(already_processed_qids))
    logger.info('get_candidate_encoding_parallel.len(wikidata_qid_to_label_id): %s' % len(wikidata_qid_to_label_id))
    #
    tokenizer = BertTokenizer.from_pretrained(
        params['model']['bert_model'], do_lower_case=params['model']['lowercase'],
        cache_dir=params['model']['bert_cache_dir']
    )

    biencoder = BiEncoder(params['model'])
    model_path = params['model'].get('path_to_model', None)
    is_loaded = False
    while not is_loaded:
        try:

            if model_path is not None:
                load_model(biencoder, model_path, gpu_id=gpu_id, cpu=params['model']['no_cuda'])
            is_loaded = True
        except:
            logger.error('Error in load_model %s' % traceback.format_exc())
            time.sleep(random.randint(5, 100))
            continue

    if not params['model']['no_cuda']:
        device = torch.device('cuda:{}'.format(gpu_id))
        biencoder.to(device)

    max_seq_length = params['model']['max_cand_length']
    logger.info('!!inside get_candidate_encodings_parallel!!')
    logger.info('arq.process_files_queue.empty(): %s' % arq.process_files_queue.empty())
    logger.info('v_shutdown_encoders.value: %s' % v_shutdown_encoders.value)
    curr_file_path = ''
    while not (arq.process_files_queue.empty() and v_shutdown_encoders.value == 1):
        try:
            if v_nr_entities.value % 1000 == 0 and v_nr_entities.value > 0:
                logger.info('nr of loaded entities: %s' % v_nr_entities.value)
                time_lapse_mins = (time.time() - start_time) / 60
                logger.info('Avg. nr of entities per minute: %s' % ((v_nr_entities.value / time_lapse_mins)))

            try:
                json_wikipedia_str = arq.process_files_queue.get(block=True, timeout=30)

            except Exception as e:
                logger.error('Error getting curr file path from process_files_queue: %s' % traceback.format_exc())
                continue

            json_wikipedia = json.loads(json_wikipedia_str)
            # if the wikidata_qid was already processed in the previous run, then ignores
            wikidata_qid = json_wikipedia['wikidata_qid']
            wikidata_qid_already_tensorized = True

            for curr_date_cut in sorted(json_wikipedia['evolution'].keys()):
                b_curr_version_text = json_wikipedia['evolution'][curr_date_cut]['bert_tokenized']
                if len(b_curr_version_text) == 0:
                    continue
                if wikidata_qid not in wikidata_qid_to_label_id[curr_date_cut]:
                    wikidata_qid_already_tensorized = False

            # if wikidata_qid in already_processed_qids and wikidata_qid_already_tensorized:
            if wikidata_qid in already_processed_qids and wikidata_qid_already_tensorized:
                v_nr_entities.value += 1
                logger.info('%s SKIPPING ALLTOGETHER %s' % (v_nr_entities.value, wikidata_qid))
                continue

            cand_pool = list()
            for curr_date_cut in sorted(json_wikipedia['evolution'].keys()):
                b_curr_version_text = json_wikipedia['evolution'][curr_date_cut]['bert_tokenized']
                len_title = 0
                b_curr_version_title = []
                if params['model']['use_candidate_title']:
                    if 'bert_wikipedia_title' not in json_wikipedia['evolution'][curr_date_cut]:
                        curr_version_title = json_wikipedia['evolution'][curr_date_cut]['wikipedia_title']
                        b_curr_version_title = tokenizer.tokenize(curr_version_title)
                    else:
                        b_curr_version_title = json_wikipedia['evolution'][curr_date_cut][
                            'bert_wikipedia_title']
                    len_title = len(b_curr_version_title)

                # 11/05/2022 - commented this, because can lead to all sort of errors later during prediction
                #                 in case there is a mention that actually points to this entity, the encoding won't be
                #                 found, causing error, this produces the "WARNING CAN NOT FIND THE target_wikidata_qid!!!"
                #                   in wiki.wikipedia.bi_encoder.data_process.get_dataset_eval
                # if len(b_curr_version_text) == 0:
                #     print('WARNING - continuing because of lack of text in ', wikidata_qid, ' in cut ', curr_date_cut)
                #     continue
                if len(b_curr_version_text) == 0:
                    # just puts something here
                    b_curr_version_text = ['none']
                # 11/05/2022 - commented this
                text_passages = get_passages_bert_tokens_input(b_curr_version_text,
                                                               max_seq_length - 2 - len_title)
                rep = get_candidate_passage_representation(text_passages[0], tokenizer, max_seq_length,
                                                           b_curr_version_title)
                if not params['model']['no_cuda']:
                    tensor_rep = torch.tensor(rep['ids'], dtype=torch.int64,
                                              device=torch.device('cuda:{}'.format(gpu_id)))
                else:
                    tensor_rep = torch.LongTensor(rep['ids'])
                cand_pool.append((curr_date_cut, tensor_rep))

            if len(cand_pool) == 0:
                logger.warning('WARNING, len(cand_pool) is 0 for the following wikidata_qid: %s' % wikidata_qid)
                v_nr_entities.value += 1
                continue
            sampler = SequentialSampler(cand_pool)
            batch_size = params['model']['encode_batch_size']
            if batch_size > len(cand_pool):
                batch_size = len(cand_pool)
            # data_loader = DataLoader(cand_pool, sampler=sampler, batch_size=len(cand_pool))
            data_loader = DataLoader(cand_pool, sampler=sampler, batch_size=batch_size)

            iter_ = data_loader

            queue2_content = dict()
            for step, batch in enumerate(iter_):
                date_cuts_in_batch = batch[0]
                token_ids_in_batch = batch[1]
                token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(token_ids_in_batch, NULL_IDX)
                _, embedding_cands = biencoder(
                    None, None, None, token_idx_cands, segment_idx_cands, mask_cands
                )
                # print('token_idx_cands.shape --> ', token_idx_cands.shape)
                # print('token_ids_in_batch.shape --> ', token_ids_in_batch.shape)
                cand_encode = embedding_cands.detach().cpu()
                token_idx_cands = token_idx_cands.detach().cpu()
                for idx_time_cut, time_cut in enumerate(date_cuts_in_batch):
                    json_wikipedia['evolution'][time_cut]['encoded'] = cand_encode[idx_time_cut].tolist()
                    queue2_content[time_cut] = dict()
                    queue2_content[time_cut]['candidate_encoding'] = cand_encode[idx_time_cut].numpy()
                    queue2_content[time_cut]['token_idx_cands'] = token_idx_cands[idx_time_cut].numpy()

            if params['produce_encoded_jsonl'] and wikidata_qid not in already_processed_qids:
                arq.gatherer_encodings_queue.put(json_wikipedia)
            else:
                pass

            if not wikidata_qid_already_tensorized:
                arq.gatherer_encodings_queue2.put((queue2_content, wikidata_qid, max_seq_length))
            else:
                pass

            v_nr_entities.value += 1

        except Exception as e:
            logger.error('error when doing json.load in get_candidate_pool: %s  when processing %s, error stack: %s' %
                         (str(e), curr_file_path, traceback.format_exc))


def get_candidate_encodings_parallel_tn(params, gpu_id, start_time, v_nr_entities, arq: ArticleReadingQueue,
                                        v_shutdown_encoders, already_processed_qids: Set,
                                        wikidata_qid_to_label_id: Dict):
    logger.info('length of already_processed_qids: %s' % len(already_processed_qids))
    tokenizer = BertTokenizer.from_pretrained(
        params['model']['bert_model'], do_lower_case=params['model']['lowercase'],
        cache_dir=params['model']['bert_cache_dir']
        # params['bert_model'], do_lower_case=params['lowercase'], cache_dir='cache_bert/'
    )

    biencoder = BiEncoder(params['model'])
    model_path = params['model'].get('path_to_model', None)
    is_loaded = False
    while not is_loaded:
        try:

            if model_path is not None:
                load_model(biencoder, model_path, gpu_id=gpu_id, cpu=params['model']['no_cuda'])
            is_loaded = True
        except:
            logger.error('Error in load_model %s' % traceback.format_exc())
            time.sleep(random.randint(5, 10))
            continue

    if not params['model']['no_cuda']:
        device = torch.device('cuda:{}'.format(gpu_id))
        biencoder.to(device)

    # doc_id_to_passages = dict()
    # doc_id_to_coref_annotations = dict()
    # map_labels_to_doc_ids = dict()

    max_seq_length = params['model']['max_cand_length']
    logger.info('!!inside get_candidate_encodings_parallel!!')
    logger.info('arq.process_files_queue.empty(): %s' % str(arq.process_files_queue.empty()))
    logger.info('v_shutdown_encoders.value: %s' % v_shutdown_encoders.value)
    curr_file_path = ''
    while not (arq.process_files_queue.empty() and v_shutdown_encoders.value == 1):
        try:
            if v_nr_entities.value % 1000 == 0 and v_nr_entities.value > 0:
                logger.info('nr of loaded entities: %s' % v_nr_entities.value)
                time_lapse_mins = (time.time() - start_time) / 60
                logger.info('Avg. nr of entities per minute: %s' % (v_nr_entities.value / time_lapse_mins))

            try:
                json_wikipedia_str = arq.process_files_queue.get(block=True, timeout=30)

            except Exception as e:
                logger.error('Error getting curr file path from process_files_queue: %s' % traceback.format_exc())
                continue

            json_wikipedia = json.loads(json_wikipedia_str)
            # if the wikidata_qid was already processed in the previous run, then ignores
            wikidata_qid = json_wikipedia['wikidata_qid']
            wikidata_qid_already_tensorized = True

            for curr_date_cut in sorted(json_wikipedia['evolution'].keys()):
                b_curr_version_text = json_wikipedia['evolution'][curr_date_cut]['bert_tokenized']
                if len(b_curr_version_text) == 0:
                    continue
                if wikidata_qid not in wikidata_qid_to_label_id[curr_date_cut]:
                    wikidata_qid_already_tensorized = False

            # if wikidata_qid in already_processed_qids and wikidata_qid_already_tensorized:
            if wikidata_qid in already_processed_qids and wikidata_qid_already_tensorized:
                v_nr_entities.value += 1
                logger.info('%s SKIPPING ALLTOGETHER %s' % (v_nr_entities.value, wikidata_qid))
                continue

            cand_pool = list()
            for curr_date_cut in sorted(json_wikipedia['evolution'].keys()):
                b_curr_version_text = json_wikipedia['evolution'][curr_date_cut]['bert_tokenized']
                len_title = 0
                b_curr_version_title = []
                if params['model']['use_candidate_title']:
                    if 'bert_wikipedia_title' not in json_wikipedia['evolution'][curr_date_cut]:
                        curr_version_title = json_wikipedia['evolution'][curr_date_cut]['wikipedia_title']
                        b_curr_version_title = tokenizer.tokenize(curr_version_title)
                    else:
                        b_curr_version_title = json_wikipedia['evolution'][curr_date_cut][
                            'bert_wikipedia_title']
                    len_title = len(b_curr_version_title)

                if len(b_curr_version_text) == 0:
                    logger.warning('WARNING - continuing because of lack of text in %s in cut %s' %
                                   (wikidata_qid, curr_date_cut))
                    continue
                text_passages = get_passages_bert_tokens_input(b_curr_version_text,
                                                               max_seq_length - 2 - len_title)
                rep = get_candidate_passage_representation(text_passages[0], tokenizer, max_seq_length,
                                                           b_curr_version_title)
                if not params['model']['no_cuda']:
                    tensor_rep = torch.tensor(rep['ids'], dtype=torch.int64,
                                              device=torch.device('cuda:{}'.format(gpu_id)))
                else:
                    tensor_rep = torch.LongTensor(rep['ids'])
                cand_pool.append((curr_date_cut, tensor_rep))

            if len(cand_pool) == 0:
                logger.warning('WARNING, len(cand_pool) is 0 for the following wikidata_qid: %s' % wikidata_qid)
                v_nr_entities.value += 1
                continue
            sampler = SequentialSampler(cand_pool)
            batch_size = params['model']['encode_batch_size']
            if batch_size > len(cand_pool):
                batch_size = len(cand_pool)
            # data_loader = DataLoader(cand_pool, sampler=sampler, batch_size=len(cand_pool))
            data_loader = DataLoader(cand_pool, sampler=sampler, batch_size=batch_size)

            iter_ = data_loader

            queue2_content = dict()
            for step, batch in enumerate(iter_):
                date_cuts_in_batch = batch[0]
                token_ids_in_batch = batch[1]
                token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(token_ids_in_batch, NULL_IDX)
                _, embedding_cands = biencoder(
                    None, None, None, token_idx_cands, segment_idx_cands, mask_cands
                )
                # print('token_idx_cands.shape --> ', token_idx_cands.shape)
                # print('token_ids_in_batch.shape --> ', token_ids_in_batch.shape)
                cand_encode = embedding_cands.detach().cpu()
                token_idx_cands = token_idx_cands.detach().cpu()
                for idx_time_cut, time_cut in enumerate(date_cuts_in_batch):
                    json_wikipedia['evolution'][time_cut]['encoded'] = cand_encode[idx_time_cut].tolist()
                    queue2_content[time_cut] = dict()
                    queue2_content[time_cut]['candidate_encoding'] = cand_encode[idx_time_cut].numpy()
                    queue2_content[time_cut]['token_idx_cands'] = token_idx_cands[idx_time_cut].numpy()

            if params['produce_encoded_jsonl'] and wikidata_qid not in already_processed_qids:
                arq.gatherer_encodings_queue.put(json_wikipedia)
            else:
                pass

            if not wikidata_qid_already_tensorized:
                arq.gatherer_encodings_queue2.put((queue2_content, wikidata_qid, max_seq_length))
            else:
                pass

            v_nr_entities.value += 1

        except Exception as e:
            logger.error('error when doing json.load in get_candidate_pool: %s'
                         ' when processing %s with a stack: %s' % (str(e), curr_file_path, traceback.format_exc()))


def gather_candidate_encodings(v_shutdown_gather, arq: ArticleReadingQueue, params, wikidata_qid_to_label_id: Dict,
                               label_id_to_wikidata_qid: Dict):
    """
    New version with (hopefully) more efficient concatenator (class Gatherer)
    :param v_shutdown_gather:
    :param arq:
    :param output_path:
    :param frequency_concatenate:
    :return:
    """
    initial_capacity = params['initial_capacity']
    increment_capacity = params['increment_capacity']
    base_experiment_path = params['base_experiment_path']
    cand_faiss_path = os.path.join(base_experiment_path, params['cand_faiss_path'])
    os.makedirs(cand_faiss_path, exist_ok=True)
    str_buffer_nr_files_in_gatherer = params['str_buffer_nr_files_in_gatherer']

    candidate_embeddings_per_date_cut = dict()
    candidate_token_ids_per_date_cut = dict()
    nr_gathered_per_date_cut = dict()

    already_loaded_dates = list(wikidata_qid_to_label_id.keys())
    for curr_loaded_date in already_loaded_dates:
        assert len(wikidata_qid_to_label_id[curr_loaded_date]) == len(label_id_to_wikidata_qid[curr_loaded_date])
        nr_gathered_per_date_cut[curr_loaded_date] = len(wikidata_qid_to_label_id[curr_loaded_date])

    nr_gathered = 0
    while not (v_shutdown_gather.value == 1 and arq.gatherer_encodings_queue.empty()):
        try:
            # candidate_embeddings_per_date_cut : curr_date_cut --> candidate tensor
            # wikidata_qid_to_label_id : curr_date_cut --> wikidata_qid --> label_id
            # label_id_to_wikidata_qid : curr_date_cut --> label_id --. wikidata_qid
            queue2_content, wikidata_qid, max_seq_length = arq.gatherer_encodings_queue2.get(block=True, timeout=30)
            nr_gathered += 1
            if nr_gathered % 1000 == 0 and nr_gathered > 0:
                logger.info('nr_gathered tensors is: %s str_buffer_nr_files_in_gatherer is: %s'
                            ' nr_gathered %% str_buffer_nr_files_in_gatherer is %s' %
                            (nr_gathered, str_buffer_nr_files_in_gatherer,
                             (nr_gathered % str_buffer_nr_files_in_gatherer)))
            for curr_date_cut, content_date_cut in queue2_content.items():

                if wikidata_qid in wikidata_qid_to_label_id[curr_date_cut]:
                    logger.warning('%s SHOULD NOT HAPPEN: SHOULD BE ALREADY IGNORED: '
                                   'skipping gatherer for %s - %s' %
                                   (nr_gathered, wikidata_qid, curr_date_cut))
                    continue  # already exists
                if curr_date_cut not in nr_gathered_per_date_cut:
                    nr_gathered_per_date_cut[curr_date_cut] = 0

                if curr_date_cut not in wikidata_qid_to_label_id:
                    wikidata_qid_to_label_id[curr_date_cut] = dict()

                if curr_date_cut not in label_id_to_wikidata_qid:
                    label_id_to_wikidata_qid[curr_date_cut] = dict()
                curr_numpy_tensor = content_date_cut['candidate_encoding']
                # print('curr_numpy_tensor.shape --> ', curr_numpy_tensor.shape)
                # print('curr_date_cut --> ', curr_date_cut)
                curr_bert_tokens = content_date_cut['token_idx_cands']
                if curr_date_cut not in candidate_embeddings_per_date_cut:
                    dim_vector = curr_numpy_tensor.shape[0]
                    logger.info('%s initializing gatherer with dim_vector of %s' % (curr_date_cut, str(dim_vector)))
                    candidate_embeddings_per_date_cut[curr_date_cut] = Gatherer(dim_vector=dim_vector,
                                                                                initial_capacity=initial_capacity,
                                                                                increment_capacity=increment_capacity)
                    #
                    candidate_token_ids_per_date_cut[curr_date_cut] = Gatherer(dim_vector=curr_bert_tokens.shape[0],
                                                                               initial_capacity=initial_capacity,
                                                                               increment_capacity=increment_capacity)

                # tmp_list_tensors_per_date_cut[curr_date_cut].append(numpy_tensor)
                assert wikidata_qid not in wikidata_qid_to_label_id[curr_date_cut]
                nr_gathered_per_date_cut[curr_date_cut] += 1

                curr_nr_gathered = nr_gathered_per_date_cut[curr_date_cut]
                wikidata_qid_to_label_id[curr_date_cut][wikidata_qid] = curr_nr_gathered - 1
                label_id_to_wikidata_qid[curr_date_cut][curr_nr_gathered - 1] = wikidata_qid

                # print('curr_numpy_tensor.dtype: ', curr_numpy_tensor.dtype)
                candidate_embeddings_per_date_cut[curr_date_cut].add(curr_numpy_tensor)
                candidate_token_ids_per_date_cut[curr_date_cut].add(curr_bert_tokens)
                # candidate_lengths_per_date_cut[curr_date_cut].add(curr_bert_tokens_length)

            if nr_gathered % str_buffer_nr_files_in_gatherer == 0 and nr_gathered > 0:
                candidate_embeddings_per_date_cut, candidate_token_ids_per_date_cut = \
                    copy_tensors(candidate_embeddings_per_date_cut, candidate_token_ids_per_date_cut,
                                 cand_faiss_path, label_id_to_wikidata_qid, wikidata_qid_to_label_id, nr_gathered,
                                 initial_capacity, increment_capacity, nr_gathered_per_date_cut)
        except Exception as e:
            logger.error('type error gather_candidate_encodings: %s with stack %s' % (str(e), traceback.format_exc()))

    copy_tensors(candidate_embeddings_per_date_cut, candidate_token_ids_per_date_cut,
                 cand_faiss_path, label_id_to_wikidata_qid, wikidata_qid_to_label_id, nr_gathered,
                 initial_capacity, increment_capacity, nr_gathered_per_date_cut)


def gather_and_write_encodings(params, arq: ArticleReadingQueue, v_shutdown_file_writers):
    # TODO , here write IN A SINGLE FILE with a cache of 100 MB or so, little bit more than
    #  currently in s01_wikipedia_clean_and_tokenize.py
    tot_len = 0
    tot_nr_files = 0
    str_buffer_size = params['str_buffer_size']
    str_buffer_nr_files = params['str_buffer_nr_files']
    encoding_file_path = params['output_file']
    base_experiment_path = params['base_experiment_path']
    encoding_file_path = os.path.join(base_experiment_path, encoding_file_path)
    with open(encoding_file_path, 'ta', encoding='utf8') as outfile:
        accum_json_str = ''
        while not (v_shutdown_file_writers.value == 1 and arq.gatherer_encodings_queue.empty()):
            try:
                to_save = arq.gatherer_encodings_queue.get(block=True, timeout=30)
            except Exception as e:
                continue

            json_str = json.dumps(to_save, ensure_ascii=False)
            json_str += '\n'
            len_json_str = len(json_str)
            tot_len += (len_json_str / 1024 / 1024)
            accum_json_str += json_str
            # json_temporal_evolution_file.flush()
            # json_temporal_evolution_file.close()
            tot_nr_files += 1
            if tot_len > str_buffer_size or tot_nr_files > str_buffer_nr_files:
                logger.info('APPENDING to the encodingsfile with tot_len of %s'
                            ' and tot_nr_files of %s (in other words a checkpoint)' % (tot_len, tot_nr_files))
                outfile.write(accum_json_str)
                outfile.flush()
                logger.info('DONE WITH APPENDIND for %s'
                            ' and tot_nr_files of %s' % (tot_len, tot_nr_files))
                accum_json_str = ''
                tot_len = 0
                tot_nr_files = 0
        if len(accum_json_str) > 0:
            outfile.write(accum_json_str)
            outfile.flush()

    logger.info('!!(in process_evolution_file_writer) closed the file from process_page_info_writer(), DONE')


def process_input_files():
    # todo - read the input files, tokenize using BERT if still not tokenized, should be done in 1 thread or so

    pass


def encode_input_entities():
    # todo - takes as input the bert-tokenized from process_input_files() and encodes them!
    #  see the best number of threads vs. availability of GPUs, probably if run on CPUs to compensate for lack of
    #  speed has to be launched on more threads
    pass


def display(processes_files: List, process_gather_and_write: List, v_nr_entities, arq: ArticleReadingQueue,
            initial_time):
    while True:
        try:
            logger.info('\tDISPLAY Size of queue process_files_queue: %s' % arq.process_files_queue.qsize())
            logger.info('\tDISPLAY Size of queue gatherer_encodings_queue: %s' % arq.gatherer_encodings_queue.qsize())
            is_alive = 0
            is_dead = 0
            for curr_process_file in processes_files:
                if curr_process_file.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1
            logger.info('\tDISPLAY NR file processing processes alive: %s and dead: %s' % (is_alive, is_dead))
            is_alive = 0
            is_dead = 0
            for curr_process_file in process_gather_and_write:
                if curr_process_file.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1

            logger.info('\tDISPLAY NR gather processes: %s and dead: %s' % (is_alive, is_dead))

            curr_time = time.time()
            logger.info('\tDISPLAY TOT processed entities: %s'
                        ' Avg. articles per minute: %s' %
                        (v_nr_entities.value, (v_nr_entities.value / ((curr_time - initial_time) / 60))))

            time.sleep(60 * 1)
        except Exception as e:
            logger.error('type error display: %s with stack: %s' % (str(e), traceback.format_exc()))
            continue
