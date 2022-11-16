import json
import logging
import os
from typing import List

import torch
from tqdm import tqdm

from models.biencoder.biencoder import BiEncoder
from models.utils.utils import chunk_document_in_sentences
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def load_model(model: BiEncoder, fname, gpu_id, cpu=False):
    logger.info('before biencoder loading model the cpu is in %s' % cpu)
    if cpu:
        state_dict = torch.load(fname, map_location='cpu')
    else:
        device = torch.device('cuda:{}'.format(gpu_id))
        # 24/03/2022 - made this change without testing well, may fail
        # state_dict = torch.load(fname, map_location=lambda storage, loc: storage.cuda(gpu_id))
        logger.info('loading fname %s  into device %s  cuda: %s' % (fname, device, gpu_id))
        # state_dict = torch.load(fname, map_location=lambda storage, loc: storage.cuda(device))
        state_dict = torch.load(fname, map_location=device)
    model.load_state_dict(state_dict)


def get_passages_bert_tokens_input(bert_tokens: List[str], max_cand_length: int):
    """

    :param max_cand_length:
    :param text:
    :return: Gets the passages from the text, each one of maximum size max_seq_length; The function will always try
    to split by sentences, and if not possible will truncate.
    """
    to_ret_passages = chunk_document_in_sentences(bert_tokens, max_cand_length)
    return to_ret_passages


def get_coref_with_title(json_doc: dict):
    title_concept_ids = set()
    concept_id_to_mentions = dict()
    for curr_mention in json_doc['mentions']:
        concept_id = curr_mention['concept_id']
        if concept_id not in concept_id_to_mentions:
            concept_id_to_mentions[concept_id] = list()
        concept_id_to_mentions[concept_id].append({
            'start_char': curr_mention['start_char'],
            'end_char': curr_mention['end_char']
        })
        if curr_mention['field_same_as_title']:
            title_concept_ids.add(concept_id)
    to_ret_mentions_coref_with_title = list()
    for curr_title_concept_id in title_concept_ids:
        concept_id_mentions = concept_id_to_mentions[curr_title_concept_id]
        to_ret_mentions_coref_with_title.extend(concept_id_mentions)

    to_ret_mentions_coref_with_title.sort(key=lambda x: (x['start_char'], x['end_char']))

    return to_ret_mentions_coref_with_title


def generate_and_save_json(nn_data, test_samples, params,
                           # label_doc_id_map, passage_id_to_label_id,
                           # data_original_path, output_path,
                           subset_name,
                           date_cut, date_cut_to_label_id_to_wikidata_qid):
    """

    :param nn_data:
    :param test_data:
    :param test_tensor_data:
    :param test_samples:
    :return:

    resulting json format:
    [
    {
        "context_doc_id": "D46C64DD4AFBC7B6",
        "context_mention_pos": [500, 505], # pos (in chars?)
        "context_mention_str": "JoeDoe",
        "bert_sees_span": [250, 750], # what bi-encoder can "see" (in chars?)
        "gold_label_doc_id": "D46C64DD4AFBC7ZZ",
        "category": "LOW_OVERLAP",
        "candidates": # scores with selected candidates
        [{
            "candidate_doc_id": "D46C64DD4AFBC7ZY",
            "candidate_score": 0.98,
            "candidate_correct": False
        },
        ...
        ]
    }
    ]
    """

    data_path = os.path.join(params['base_experiment_path'], params['data_path'])

    output_path = os.path.join(params['base_experiment_path'], params['output_dir_predictions'],
                               'json', subset_name)

    test_samples_original = list()
    path_test_samples_original = os.path.join(data_path, '{}{}_{}_all.jsonl'
                                              .format(params['dataset_filename_prefix'], date_cut['mentions_cut'],
                                                      subset_name))
    with open(path_test_samples_original, 'rt') as orig_set:
        for idx, line in enumerate(orig_set):
            test_samples_original.append(json.loads(line))
            if params['debug'] and idx + 1 >= params['debug_size']:
                break
    logger.info('-------begin log generate_and_save_json for %s' % date_cut)
    logger.info('len(test_samples_original) from %s: %s' % (path_test_samples_original, len(test_samples_original)))
    logger.info('len(test_samples): %s' % len(test_samples))
    logger.info('-------end log generate_and_save_json for %s' % str(date_cut))
    assert len(test_samples_original) == len(test_samples)

    os.makedirs(output_path, exist_ok=True)

    out_json_path = os.path.join(output_path, 'pred_{}_m{}_e{}_all.jsonl'.format(subset_name,
                                                                                 date_cut['mentions_cut'],
                                                                                 date_cut['entities_cut']))

    directory = os.path.dirname(out_json_path)
    os.makedirs(directory, exist_ok=True)
    logger.info('saving json to %s' % out_json_path)
    with open(out_json_path, 'wt') as outfile:
        for idx_datapoint, (curr_sample, curr_sample_orig) in tqdm(enumerate(zip(test_samples, test_samples_original))):
            assert curr_sample['category'] == curr_sample_orig['category']
            assert curr_sample['anchor_wikidata_qid'] == curr_sample_orig['anchor_wikidata_qid']
            assert curr_sample['target_wikidata_qid'] == curr_sample_orig['target_wikidata_qid']

            curr_sample_json = dict()

            curr_sample_json['anchor_wikidata_qid'] = curr_sample_orig['anchor_wikidata_qid']
            curr_sample_json['mention_bert_tokenized'] = curr_sample_orig['mention_bert_tokenized']
            curr_sample_json['gold_label_doc_id'] = curr_sample_orig['target_wikidata_qid']

            curr_sample_json['category'] = curr_sample_orig['category']

            pred_passage_ids = nn_data['pred_passage_ids'][idx_datapoint].tolist()
            pred_passage_scores = nn_data['pred_passage_scores'][idx_datapoint].tolist()
            goldd_passage_ids = nn_data['gold_passage_ids'][idx_datapoint].tolist()
            goldd_passage_ids = [cid for cid in goldd_passage_ids if cid != -1]

            candidates_lst = list()
            goldd_passage_ids = set(goldd_passage_ids)
            for idx_candidate, (curr_pred_passage_id, curr_pred_passage_score) in enumerate(
                    zip(pred_passage_ids, pred_passage_scores)):
                if not params['debug']:
                    candidate_doc_id = \
                        date_cut_to_label_id_to_wikidata_qid[date_cut['entities_cut']][curr_pred_passage_id]
                else:
                    if curr_pred_passage_id in date_cut_to_label_id_to_wikidata_qid[date_cut['entities_cut']]:
                        candidate_doc_id = \
                            date_cut_to_label_id_to_wikidata_qid[date_cut['entities_cut']][curr_pred_passage_id]
                    else:
                        logger.info('%s not in date_cut_to_label_id_to_wikidata_qid for date cut %s'
                                    ' which has the length %s' %
                                    (curr_pred_passage_id, date_cut,
                                     len(date_cut_to_label_id_to_wikidata_qid[date_cut['entities_cut']])))
                        candidate_doc_id = -1
                candidate_score = curr_pred_passage_score
                candidate_correct = (curr_pred_passage_id in goldd_passage_ids)
                candidates_lst.append({'candidate_doc_id': candidate_doc_id,
                                       'candidate_score': candidate_score,
                                       'candidate_correct': candidate_correct})

            curr_sample_json['candidates'] = candidates_lst
            outfile.write(json.dumps(curr_sample_json) + '\n')

    logger.info('done writing json prediction of %s' % subset_name)


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Utility code for zeshel dataset
import json

DOC_PATH = 'data/zeshel/documents/'  # kzaporoj - original:  DOC_PATH = '/private/home/ledell/zeshel/data/documents/'

WORLDS = [
    'wikipedia'
    # 'american_football',
    # 'doctor_who',
    # 'fallout',
    # 'final_fantasy',
    # 'military',
    # 'pro_wrestling',
    # 'starwars',
    # 'world_of_warcraft',
    # 'coronation_street',
    # 'muppets',
    # 'ice_hockey',
    # 'elder_scrolls',
    # 'forgotten_realms',
    # 'lego',
    # 'star_trek',
    # 'yugioh'
]

world_to_id = {src: k for k, src in enumerate(WORLDS)}


# def load_entity_dict_zeshel(logger, params):
def load_entity_dict_zeshel(logger, params):
    # puts raise to see that no execution enters here
    # raise RuntimeError('checking if load_entity_dict_zeshel is necessary')

    entity_dict = {}
    # different worlds in train/valid/test
    if params['mode'] == 'train':
        start_idx = 0
        end_idx = 8
    elif params['mode'] == 'valid':
        start_idx = 8
        end_idx = 12
    else:
        start_idx = 12
        end_idx = 16
    # load data
    for i, src in enumerate(WORLDS[start_idx:end_idx]):
        fname = DOC_PATH + src + '.json'
        cur_dict = {}
        doc_list = []
        src_id = world_to_id[src]
        with open(fname, 'rt') as f:
            for line in f:
                line = line.rstrip()
                item = json.loads(line)
                text = item['text']
                doc_list.append(text[:256])

                if params['debug']:
                    # if len(doc_list) > 200:
                    if len(doc_list) >= params['debug_size']:
                        break

        logger.info('Load for world %s.' % src)
        entity_dict[src_id] = doc_list
    return entity_dict


class Stats():
    def __init__(self, top_k=1000, rank=[1, 4, 8, 16, 32, 64, 100, 128, 256, 512]):
        # raise RuntimeError('checking if Stats() is necessary')
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = rank
        self.LEN = len(self.rank)
        for i in range(self.LEN):
            self.hits.append(0)

    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = 'Total: %d examples.' % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            output_json += ' r@%d: %.4f' % (self.rank[i], self.hits[i] / float(self.cnt))
        return output_json

    def output_json(self):
        output_json = dict()
        output_json['nr_examples'] = self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            # output_json += ' r@%d: %.4f' % (self.rank[i], self.hits[i] / float(self.cnt))
            if self.cnt == 0:
                logger.warning('WARNING: count in zero in Stats.output_json()')
                output_json['acc@{}'.format(self.rank[i])] = 0.0
            else:
                output_json['acc@{}'.format(self.rank[i])] = self.hits[i] / float(self.cnt)
        return output_json
