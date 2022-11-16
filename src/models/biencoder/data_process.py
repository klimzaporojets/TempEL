import ast
import io
import itertools
import json
import logging
import os
import random
import time
from typing import Dict

import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm

from models.biencoder.biencoder import BiEncoder, to_bert_input, NULL_IDX
from models.biencoder.misc_utils import load_model, WORLDS
from models.utils.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from utils import tempel_logger

# WORLDS = [
#     'wikipedia'
# ]

world_to_id = {src: k for k, src in enumerate(WORLDS)}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
        sample,
        tokenizer,
        max_seq_length,
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,

):
    mention_tokens = []
    if sample['mention_bert_tokenized'] and len(sample['mention_bert_tokenized']) > 0:
        mention_tokens = sample['mention_bert_tokenized']
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample['context_left_bert_tokenized']
    context_right = sample['context_right_bert_tokenized']

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (context_left[-left_quota:] + mention_tokens + context_right[:right_quota])

    context_tokens = ['[CLS]'] + context_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        'tokens': context_tokens,
        'mention_tokens': mention_tokens,
        'ids': input_ids,
    }


def get_contexts_representation_train(
        sample,
        tokenizer,
        params,
        context_coref_annos,
        context_passages,
        context_passage_train_filtering,
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
):
    """
    This one takes into account that all contexts with coreferent entity mention can be asked
    (i.e., using the parameter context_passage_train_filtering).

    :param sample:
    :param tokenizer:
    :param max_seq_length:
    :param ent_start_token:
    :param ent_end_token:
    :return:
    """
    context_coref_annos = json.loads(context_coref_annos)
    context_text_passages = context_passages['text_passages']
    context_text_passages = ast.literal_eval(context_text_passages)
    mention_start_bert_token = sample['start_bert_token']
    mention_end_bert_token = sample['end_bert_token']
    bert_mention_pos_to_concept_id = dict()
    bert_mention_concept_id_to_mention_pos = dict()

    for curr_context_doc_mention in context_coref_annos['mentions']:
        curr_bert_start = curr_context_doc_mention['start_bert_token']
        curr_bert_end = curr_context_doc_mention['end_bert_token']
        curr_concept_id = curr_context_doc_mention['concept_id']
        bert_mention_pos_to_concept_id[(curr_bert_start, curr_bert_end)] = curr_concept_id
        if curr_concept_id not in bert_mention_concept_id_to_mention_pos:
            bert_mention_concept_id_to_mention_pos[curr_concept_id] = list()
        bert_mention_concept_id_to_mention_pos[curr_concept_id].append((curr_bert_start, curr_bert_end))

    if context_passage_train_filtering == 'all_coreferent_with_mention' and \
            ((mention_start_bert_token, mention_end_bert_token) in bert_mention_pos_to_concept_id):
        curr_concept_id = bert_mention_pos_to_concept_id[(mention_start_bert_token, mention_end_bert_token)]
        mentions_pos = bert_mention_concept_id_to_mention_pos[curr_concept_id]
        # sort the mention pos as well, since not always they are ordered
        mentions_pos = sorted(mentions_pos)
    else:
        mentions_pos = [(mention_start_bert_token, mention_end_bert_token)]

    # curr_mention_idx = 0
    contextual_mentions = list()
    curr_context_bert_tokenized = [item for sublist in context_text_passages for item in sublist]

    # the original mention to disambiguate
    original_mention = None
    for curr_mention_pos in mentions_pos:
        max_seq_length = params['max_context_length']
        mention_bert_tokens = curr_context_bert_tokenized[curr_mention_pos[0]:curr_mention_pos[1] + 1]
        mention_bert_tokens = [ent_start_token] + mention_bert_tokens + [ent_end_token]

        left_quota = (max_seq_length - len(mention_bert_tokens)) // 2 - 1
        right_quota = max_seq_length - len(mention_bert_tokens) - left_quota - 2

        context_left = curr_context_bert_tokenized[curr_mention_pos[0] - left_quota:curr_mention_pos[0]]
        context_right = curr_context_bert_tokenized[curr_mention_pos[1] + 1:curr_mention_pos[1] + right_quota + 1]
        # TODO WIP WE ARE HERE DEBUGGING
        context_tokens = (context_left + mention_bert_tokens + context_right)

        context_tokens = ['[CLS]'] + context_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length

        mention_to_add = {
            'tokens': context_tokens,
            'mention_tokens': mention_bert_tokens,
            'ids': input_ids,
            'start_mention_bert_token': sample['start_bert_token'],
            'end_mention_bert_token': sample['end_bert_token'],
            'start_context_bert_token': sample['start_bert_token'] - left_quota,
            'end_context_bert_token': sample['end_bert_token'] + right_quota
        }
        if (curr_mention_pos[0], curr_mention_pos[1]) == (mention_start_bert_token, mention_end_bert_token):
            original_mention = mention_to_add
        contextual_mentions.append(mention_to_add)
    return contextual_mentions, original_mention


def get_candidate_passage_representation(
        passage_text,
        tokenizer,
        max_seq_length,
        passage_title=None,
        title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = passage_text
    if passage_title is not None:
        title_tokens = passage_title
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    length = len(input_ids)
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        'tokens': cand_tokens,
        'ids': input_ids,
        'length': length
    }


def get_candidate_representation(
        candidate_desc,
        tokenizer,
        max_seq_length,
        candidate_title=None,
        title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    if isinstance(candidate_desc, str):
        cand_tokens = tokenizer.tokenize(candidate_desc)
    else:
        cand_tokens = candidate_desc
    if candidate_title is not None:
        if isinstance(candidate_title, str):
            title_tokens = tokenizer.tokenize(candidate_title)
        else:
            title_tokens = candidate_title
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        'tokens': cand_tokens,
        'ids': input_ids,
    }


def get_first_record(params, sample, tokenizer, src):
    label_text = sample['target_bert_tokenized']
    title = None
    if params['use_candidate_title']:
        title = sample.get('target_title_bert_tokenized', None)
    label_tokens = get_candidate_representation(
        label_text, tokenizer, params['max_cand_length'], title,
    )
    record = {'label': label_tokens,
              'src': src}
    return record


def get_all_records(params, all_passages, tokenizer, src, processed_record_ids):
    records = list()
    doc_title = all_passages['title']
    text_passages_parsed = ast.literal_eval(all_passages['text_passages'])
    for record_id, curr_passage in enumerate(text_passages_parsed):
        if record_id in processed_record_ids:
            continue
        title = None
        if params['use_candidate_title']:
            title = ast.literal_eval(doc_title)
        label_tokens = get_candidate_representation(
            curr_passage, tokenizer, params['max_cand_length'], title,
        )
        record = {'label': label_tokens, 'src': src}
        records.append(record)
        processed_record_ids.add(record_id)
    return processed_record_ids, records


def get_filtered_coref_title_records(params, all_passages, all_coref_annos, tokenizer, src,
                                     processed_record_ids, filter_properties):
    all_coref_annos = json.loads(all_coref_annos)
    doc_title = all_passages['title']
    text_passages = ast.literal_eval(all_passages['text_passages'])
    map_bert_to_token = ast.literal_eval(all_passages['map_bert_to_token'])

    concept_id_to_details = {en['concept_id']: en for en in all_coref_annos['concepts']}
    curr_mention_pos = 0
    bert_token_passage_begin = 0
    bert_token_passage_end = 0
    mentions_sorted_by_pos = sorted(all_coref_annos['mentions'], key=lambda x: (x['start_bert_token'],
                                                                                x['end_bert_token']))
    passage_idx_to_nr_title_mentions = dict()
    for curr_passage_idx, (curr_passage, curr_bert_to_tokens_map) in \
            enumerate(zip(*[text_passages, map_bert_to_token])):
        bert_token_passage_end += len(curr_passage)  # TODO
        if curr_passage_idx in processed_record_ids:
            bert_token_passage_begin += len(curr_passage)
            continue
        assert len(curr_passage) == len(curr_bert_to_tokens_map)
        len_mentions = len(mentions_sorted_by_pos)
        while curr_mention_pos < len_mentions and \
                mentions_sorted_by_pos[curr_mention_pos]['end_bert_token'] < bert_token_passage_end:
            curr_mention = mentions_sorted_by_pos[curr_mention_pos]
            curr_mention_pos += 1
            if curr_mention['start_bert_token'] < bert_token_passage_begin:
                continue
            mention_concept_id = curr_mention['concept_id']
            concept_details = concept_id_to_details[mention_concept_id]
            if concept_details['is_title']:
                if curr_passage_idx not in passage_idx_to_nr_title_mentions:
                    passage_idx_to_nr_title_mentions[curr_passage_idx] = 0
                passage_idx_to_nr_title_mentions[curr_passage_idx] += 1
        bert_token_passage_begin += len(curr_passage)
    #########################
    idx_passage_sorted = sorted(list(passage_idx_to_nr_title_mentions.items()), key=lambda x: x[1], reverse=True)

    idx_passage_sorted = [ip[0] for ip in idx_passage_sorted if ip[1] >= 1]

    idx_passage_sorted = idx_passage_sorted[:filter_properties['top_n']]

    records = list()

    title = None
    if params['use_candidate_title']:
        title = ast.literal_eval(doc_title)
    for curr_passage_idx in idx_passage_sorted:
        curr_passage = text_passages[curr_passage_idx]
        label_tokens = get_candidate_representation(
            curr_passage, tokenizer, params['max_cand_length'], title,
        )
        record = {'label': label_tokens,
                  'src': src}
        records.append(record)
        processed_record_ids.add(curr_passage_idx)
    return processed_record_ids, records


def get_filtered_coref_w_mention_in_context_records(params, all_label_passages, label_coref_annos, context_coref_annos,
                                                    tokenizer, context_tokens, src, processed_record_ids,
                                                    filter_properties):
    context_coref_annos = json.loads(context_coref_annos)
    mention_span_to_concept = dict()
    start_mention_bert_token = context_tokens['start_mention_bert_token']
    end_mention_bert_token = context_tokens['end_mention_bert_token']
    context_concept_id_to_mentions = dict()
    for curr_context_mention in context_coref_annos['mentions']:
        mention_span_to_concept[(curr_context_mention['start_bert_token'], curr_context_mention['end_bert_token'])] = \
            curr_context_mention['concept_id']
        if ('source' in curr_context_mention and curr_context_mention['source'] in {'NER', 'complemented-NER'}) or \
                sum([cw[0].isupper() for cw in curr_context_mention['text'].split(' ')]) >= 2:

            if curr_context_mention['concept_id'] not in context_concept_id_to_mentions:
                context_concept_id_to_mentions[curr_context_mention['concept_id']] = set()

            context_concept_id_to_mentions[curr_context_mention['concept_id']].add(curr_context_mention['text'])

    records = list()
    if (start_mention_bert_token, end_mention_bert_token) in mention_span_to_concept:
        mention_concept_id = mention_span_to_concept[(start_mention_bert_token, end_mention_bert_token)]
        if mention_concept_id in context_concept_id_to_mentions:
            mentions_to_match = context_concept_id_to_mentions[mention_concept_id]
            matched_label_doc_concepts = set()
            label_coref_annos = json.loads(label_coref_annos)

            for curr_label_mention in label_coref_annos['mentions']:
                if ('source' in curr_label_mention and curr_label_mention['source'] in {'NER', 'complemented-NER'}) or \
                        sum([cw[0].isupper() for cw in curr_label_mention['text'].split(' ')]) >= 2:
                    if curr_label_mention['text'] in mentions_to_match:
                        matched_label_doc_concepts.add(curr_label_mention['concept_id'])

            curr_mention_index = 0
            curr_passage_bert_start = 0
            curr_passage_bert_end = 0
            text_passages = ast.literal_eval(all_label_passages['text_passages'])
            len_mentions = len(label_coref_annos['mentions'])
            label_mentions = sorted(label_coref_annos['mentions'],
                                    key=lambda x: (x['start_bert_token'], x['end_bert_token']))
            label_passage_idx_to_stats = dict()
            for idx_passage, curr_label_passage in enumerate(text_passages):
                curr_passage_bert_end += len(curr_label_passage)
                if idx_passage in processed_record_ids:
                    curr_passage_bert_start += len(curr_label_passage)
                    continue
                passage_overlapped_mentions = set()
                passage_overlapped_concept_ids = set()
                while curr_mention_index < len_mentions and \
                        label_mentions[curr_mention_index]['end_bert_token'] <= curr_passage_bert_end:
                    curr_mention_label_passage = label_mentions[curr_mention_index]
                    curr_mention_index += 1
                    if curr_mention_label_passage['start_bert_token'] < curr_passage_bert_start:
                        continue
                    if curr_mention_label_passage['concept_id'] in matched_label_doc_concepts:
                        passage_overlapped_concept_ids.add(curr_mention_label_passage['concept_id'])
                        passage_overlapped_mentions.add((curr_mention_label_passage['start_bert_token'],
                                                         curr_mention_label_passage['end_bert_token'],))

                curr_passage_bert_start += len(curr_label_passage)
                label_passage_idx_to_stats[idx_passage] = {'nr_mentions': len(passage_overlapped_mentions),
                                                           'nr_concepts': len(passage_overlapped_concept_ids)}

            idx_passage_sorted = sorted(list(label_passage_idx_to_stats.items()),
                                        key=lambda x: (x[1]['nr_concepts'], x[1]['nr_mentions']), reverse=True)

            # here additional filters such as chose only top x sorted passages or the ones with more than x overlapped
            #  concepts, etc
            idx_passage_sorted = [ip[0] for ip in idx_passage_sorted if ip[1]['nr_concepts'] >= 1]

            idx_passage_sorted = idx_passage_sorted[:filter_properties['top_n']]

            records = list()  # '000077489F1CDA91' --> '000077489F1CDA91'

            doc_title = all_label_passages['title']

            title = None
            if params['use_candidate_title']:
                title = ast.literal_eval(doc_title)
            for curr_passage_idx in idx_passage_sorted:
                curr_passage = text_passages[curr_passage_idx]
                label_tokens = get_candidate_representation(
                    curr_passage, tokenizer, params['max_cand_length'], title,
                )
                record = {'label': label_tokens,
                          'src': src}
                records.append(record)
                processed_record_ids.add(curr_passage_idx)
    return processed_record_ids, records


def get_filtered_coref_context_entities_records(params, all_label_passages, label_coref_annos, context_coref_annos,
                                                tokenizer, context_tokens, src, processed_record_ids,
                                                filter_properties):
    # step 0 - information on contextual passage, as well as the concepts/mentions inside the contextual passage
    context_coref_annos = json.loads(context_coref_annos)
    mention_span_to_concept = dict()
    concept_id_to_NER_mentions = dict()
    # only for ners etc that are going to be used to match, should not be used for pronouns for example
    mention_text_to_concept_ids_context_passage = dict()
    for curr_context_mention in context_coref_annos['mentions']:
        mention_span_to_concept[(curr_context_mention['start_bert_token'], curr_context_mention['end_bert_token'])] = \
            curr_context_mention['concept_id']
        # only mentions that are of source NER/NER-complement or that have two or more words starting in upper
        if ('source' in curr_context_mention and curr_context_mention['source'] in {'NER', 'complemented-NER'}) or \
                sum([cw[0].isupper() for cw in curr_context_mention['text'].split(' ')]) >= 2:
            if curr_context_mention['concept_id'] not in concept_id_to_NER_mentions:
                concept_id_to_NER_mentions[curr_context_mention['concept_id']] = set()
            concept_id_to_NER_mentions[curr_context_mention['concept_id']].add(curr_context_mention['text'])

    start_context_bert_token = context_tokens['start_context_bert_token']
    end_context_bert_token = context_tokens['end_context_bert_token']
    for curr_context_mention in context_coref_annos['mentions']:
        if curr_context_mention['concept_id'] in concept_id_to_NER_mentions and \
                curr_context_mention['start_bert_token'] >= start_context_bert_token and \
                curr_context_mention['end_bert_token'] <= end_context_bert_token:
            for mention_text in concept_id_to_NER_mentions[curr_context_mention['concept_id']]:
                if mention_text not in mention_text_to_concept_ids_context_passage:
                    mention_text_to_concept_ids_context_passage[mention_text] = set()
                mention_text_to_concept_ids_context_passage[mention_text].add(curr_context_mention['concept_id'])

    concepts_label_overlap_all_doc = set()
    label_coref_annos = json.loads(label_coref_annos)

    for curr_label_mention in label_coref_annos['mentions']:
        if ('source' in curr_label_mention and curr_label_mention['source'] in {'NER', 'complemented-NER'}) or \
                sum([cw[0].isupper() for cw in curr_label_mention['text'].split(' ')]) >= 2:
            if curr_label_mention['text'] in mention_text_to_concept_ids_context_passage:
                concepts_label_overlap_all_doc.add(curr_label_mention['concept_id'])

    doc_title = all_label_passages['title']

    curr_mention_index = 0
    curr_passage_bert_start = 0
    curr_passage_bert_end = 0
    text_passages = ast.literal_eval(all_label_passages['text_passages'])
    len_mentions = len(label_coref_annos['mentions'])
    label_mentions = sorted(label_coref_annos['mentions'], key=lambda x: (x['start_bert_token'], x['end_bert_token']))
    label_passage_idx_to_stats = dict()
    for idx_passage, curr_label_passage in enumerate(text_passages):
        curr_passage_bert_end += len(curr_label_passage)
        if idx_passage in processed_record_ids:
            curr_passage_bert_start += len(curr_label_passage)
            continue
        passage_overlapped_mentions = set()
        passage_overlapped_concept_ids = set()
        while curr_mention_index < len_mentions and \
                label_mentions[curr_mention_index]['end_bert_token'] <= curr_passage_bert_end:
            curr_mention_label_passage = label_mentions[curr_mention_index]
            curr_mention_index += 1
            if curr_mention_label_passage['start_bert_token'] < curr_passage_bert_start:
                continue
            if curr_mention_label_passage['concept_id'] in concepts_label_overlap_all_doc:
                passage_overlapped_concept_ids.add(curr_mention_label_passage['concept_id'])
                passage_overlapped_mentions.add((curr_mention_label_passage['start_bert_token'],
                                                 curr_mention_label_passage['end_bert_token'],))

        curr_passage_bert_start += len(curr_label_passage)
        label_passage_idx_to_stats[idx_passage] = {'nr_mentions': len(passage_overlapped_mentions),
                                                   'nr_concepts': len(passage_overlapped_concept_ids)}

    idx_passage_sorted = sorted(list(label_passage_idx_to_stats.items()),
                                key=lambda x: (x[1]['nr_concepts'], x[1]['nr_mentions']), reverse=True)

    idx_passage_sorted = [ip[0] for ip in idx_passage_sorted if ip[1]['nr_concepts'] >= 1]
    idx_passage_sorted = idx_passage_sorted[:filter_properties['top_n']]

    records = list()  # '000077489F1CDA91' --> '000077489F1CDA91'

    title = None
    if params['use_candidate_title']:
        title = ast.literal_eval(doc_title)
    for curr_passage_idx in idx_passage_sorted:
        curr_passage = text_passages[curr_passage_idx]
        label_tokens = get_candidate_representation(
            curr_passage, tokenizer, params['max_cand_length'], title,
        )
        record = {'label': label_tokens,
                  'src': src}
        records.append(record)
        processed_record_ids.add(curr_passage_idx)
    return processed_record_ids, records


def get_mentions_from_dataset_wiki_cuts(subset_name: str, tokenizer: BertTokenizer, params: Dict, date_cut: Dict,
                                        # biencoder,
                                        label_id_to_wikidata_qid: Dict, wikidata_qid_to_label_id: Dict,
                                        is_training=True, gpu_id=0):
    """

    :param subset_name:
    :param tokenizer:
    :param params:
    :param doc_id_to_passages:
            doc_id_to_passages --> world_id (key)
                                -> doc_id (key)
                                -> 'text_passages': ast-string of list of lists of bert tokens
                                -> 'title': ast-string of list of bert tokens
                                -> 'map_bert_to_token': ast-string of list of lists of integers used
                                   to map from bert to token pos. TODO: not sure if this is needed

    :param doc_id_to_coref_annotations:
    :param is_training:
    :return:
    """
    base_experiment_path = params['base_experiment_path']
    mention_encode_path = os.path.join(base_experiment_path, params['encoded_mentions'])
    os.makedirs(mention_encode_path, exist_ok=True)

    subset_tensor_data = None
    subset_samples = None

    if mention_encode_path is not None:

        mention_encode_path = os.path.join(mention_encode_path,
                                           'm{}_e{}_{}.pt'.format(
                                               date_cut['mentions_cut'],
                                               date_cut['entities_cut'],
                                               subset_name
                                           ))
        try:
            (subset_samples, subset_tensor_data) = torch.load(mention_encode_path)
        except:
            logger.error('couldn\'t load mention from dataset %s' % subset_name)

    if subset_tensor_data is None:
        logger.info('about to get_dataset_to_predict (to get mentions with context, since not found in the path) %s' %
                    mention_encode_path)
        if is_training:
            pass
        else:
            begin_model = time.time()
            biencoder = BiEncoder(params)
            model_path = params.get('path_to_model', None)
            base_experiment_path = params.get('base_experiment_path', None)
            if base_experiment_path is not None:
                model_path = os.path.join(base_experiment_path, model_path)

            if model_path is not None and not params['debug']:
                load_model(biencoder, model_path, gpu_id, cpu=params['no_cuda'])

            if params['debug']:
                pass

            end_model = time.time()
            logger.info('loaded the model in ... %s minutes. ' % ((end_model - begin_model) / 60))
            if not params['no_cuda']:
                device = torch.device('cuda:{}'.format(gpu_id))
                biencoder.to(device)

            logger.info('BEGIN getting mentions for (get_dataset_eval) %s' % date_cut)
            subset_samples, subset_tensor_data = \
                get_dataset_eval(tokenizer, biencoder, params,
                                 # doc_id_to_passages, doc_id_to_coref_annotations,
                                 date_cut,
                                 label_id_to_wikidata_qid,
                                 wikidata_qid_to_label_id,
                                 subset_name)
            logger.info('END getting mentions for (get_dataset_eval) %s' % date_cut)
            del biencoder
            torch.cuda.empty_cache()

        if mention_encode_path is not None:
            torch.save((subset_samples, subset_tensor_data), mention_encode_path)
    #
    # subset_samples: None
    #
    #
    #
    # subset_tensor_data:
    #  if is_training in True:
    #    TensorDataset (from torch.utils.data.dataset), each row with following tuple:
    #      - pos 0: torch.tensor of bert token ids of mention context
    #      - pos 1: torch.tensor of bert token ids of candidate passage (ground truth)
    #    these tuples are paired according to the hyperparameters label_passage_train_filtering and
    #    context_passage_train_filtering.
    #
    return subset_samples, subset_tensor_data


def get_dataset(tokenizer,
                params,
                subset_name,
                doc_id_to_passages,
                doc_id_to_coref_annotations,
                ent_start_token=ENT_START_TAG,
                ent_end_token=ENT_END_TAG,
                is_training=True
                ):
    dataset_path = params['data_path']
    silent = params['silent']
    debug = params['debug']
    debug_size = params['debug_size']

    return_text = params.get('json_output', False)

    processed_labels_samples = []
    contexts_tokens = []

    file_name = '{}.jsonl'.format(subset_name)
    txt_file_path = os.path.join(dataset_path, file_name)

    if return_text:
        samples = []
    else:
        samples = None
    logger.info('getting dataset_to_predict from %s' % txt_file_path)

    context_passage_filtering = params['context_passage_filtering']

    with io.open(txt_file_path, mode='r', encoding='utf-8') as file:
        if silent:
            iter_ = file
        else:
            iter_ = tqdm(file)
        for idx, line in enumerate(iter_):
            if debug and idx >= debug_size:
                break

            sample = json.loads(line.strip())
            if return_text:
                samples.append(sample)

            src = sample['world']
            src = world_to_id[src]
            context_doc_id = sample['orig_context_doc_id']
            context_coref_annos = doc_id_to_coref_annotations[src][context_doc_id]
            context_passages = doc_id_to_passages[src][context_doc_id]
            contexts_tokens_in_doc, orig_mention_context_tokens = get_contexts_representation_train(
                sample,
                tokenizer,
                params,
                context_coref_annos,
                context_passages,
                context_passage_filtering,
                ent_start_token,
                ent_end_token,
            )
            processed_labels_samples_in_doc = []

            processed_record_ids = set()
            filter_name_to_properties = {fil['filter_name']: fil for fil in params['label_passage_filtering']}
            if 'ONLY_FIRST' in filter_name_to_properties:
                processed_record_ids.add(0)
                record = get_first_record(params, sample, tokenizer, src)
                processed_labels_samples_in_doc.append(record)

            if 'ALL' in filter_name_to_properties:
                label_doc_id = sample['orig_label_doc_id']
                if not params['debug']:
                    assert label_doc_id in doc_id_to_passages[src]
                if label_doc_id in doc_id_to_passages[src]:
                    all_label_passages = doc_id_to_passages[src][label_doc_id]
                else:
                    all_label_passages = list(doc_id_to_passages[src].items())[0][1]
                processed_record_ids, records = \
                    get_all_records(params, all_label_passages, tokenizer, src, processed_record_ids)
                processed_labels_samples_in_doc.extend(records)

            if 'COREF_W_TITLE_IN_CANDIDATE_DOC' in filter_name_to_properties:
                label_doc_id = sample['orig_label_doc_id']
                if not params['debug']:
                    assert label_doc_id in doc_id_to_passages[src]
                if label_doc_id in doc_id_to_passages[src]:
                    all_label_passages = doc_id_to_passages[src][label_doc_id]
                    label_coref_annos = doc_id_to_coref_annotations[src][label_doc_id]
                else:
                    # if not just adds the first one (SHOULD ONLY HAPPEN WHEN DEBUG IN TRUE)
                    all_label_passages = list(doc_id_to_passages[src].items())[0][1]
                    label_coref_annos = list(doc_id_to_coref_annotations[src].items())[0][1]

                processed_record_ids, records = get_filtered_coref_title_records(params, all_label_passages,
                                                                                 label_coref_annos, tokenizer,
                                                                                 src, processed_record_ids,
                                                                                 filter_name_to_properties[
                                                                                     'COREF_W_TITLE_IN_CANDIDATE_DOC'])
                processed_labels_samples_in_doc.extend(records)

            if 'COREF_W_ENTITIES_IN_CONTEXT' in filter_name_to_properties:
                label_doc_id = sample['orig_label_doc_id']
                if not params['debug']:
                    assert label_doc_id in doc_id_to_passages[src]
                    assert context_doc_id in doc_id_to_passages[src]
                if label_doc_id in doc_id_to_passages[src]:
                    all_label_passages = doc_id_to_passages[src][label_doc_id]
                    label_coref_annos = doc_id_to_coref_annotations[src][label_doc_id]
                    context_coref_annos = doc_id_to_coref_annotations[src][context_doc_id]
                else:
                    # if not just adds the first one (SHOULD ONLY HAPPEN WHEN DEBUG IN TRUE)
                    all_label_passages = list(doc_id_to_passages[src].items())[0][1]
                    label_coref_annos = list(doc_id_to_coref_annotations[src].items())[0][1]
                    context_coref_annos = list(doc_id_to_coref_annotations[src].items())[0][1]

                processed_record_ids, records = get_filtered_coref_context_entities_records(
                    params,
                    all_label_passages,
                    label_coref_annos,
                    context_coref_annos,
                    tokenizer,
                    orig_mention_context_tokens,
                    src, processed_record_ids, filter_name_to_properties['COREF_W_ENTITIES_IN_CONTEXT'])
                processed_labels_samples_in_doc.extend(records)

            if 'COREF_W_MENTION_IN_CONTEXT' in filter_name_to_properties:
                label_doc_id = sample['orig_label_doc_id']
                context_doc_id = sample['orig_context_doc_id']
                if not params['debug']:
                    assert label_doc_id in doc_id_to_passages[src]
                    assert context_doc_id in doc_id_to_passages[src]
                if label_doc_id in doc_id_to_passages[src]:
                    all_label_passages = doc_id_to_passages[src][label_doc_id]
                    label_coref_annos = doc_id_to_coref_annotations[src][label_doc_id]
                    context_coref_annos = doc_id_to_coref_annotations[src][context_doc_id]
                else:
                    # if not just adds the first one (SHOULD ONLY HAPPEN WHEN DEBUG IN TRUE)
                    all_label_passages = list(doc_id_to_passages[src].items())[0][1]
                    label_coref_annos = list(doc_id_to_coref_annotations[src].items())[0][1]
                    context_coref_annos = list(doc_id_to_coref_annotations[src].items())[0][1]

                processed_record_ids, records = get_filtered_coref_w_mention_in_context_records(
                    params,
                    all_label_passages,
                    label_coref_annos,
                    context_coref_annos,
                    tokenizer,
                    orig_mention_context_tokens,
                    src, processed_record_ids, filter_name_to_properties['COREF_W_MENTION_IN_CONTEXT'])
                processed_labels_samples_in_doc.extend(records)

            if len(processed_labels_samples_in_doc) == 0:
                # if no candidates selected, just takes the first paragraph
                processed_record_ids.add(0)
                record = get_first_record(params, sample, tokenizer, src)
                processed_labels_samples_in_doc.append(record)

            lists = [contexts_tokens_in_doc, processed_labels_samples_in_doc]
            for element in itertools.product(*lists):
                contexts_tokens.append(element[0])
                processed_labels_samples.append(element[1])

    context_vecs = torch.tensor(
        select_field(contexts_tokens, 'ids'), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_labels_samples, 'label', 'ids'), dtype=torch.long,
    )
    tensor_data = TensorDataset(context_vecs, cand_vecs)

    return samples, tensor_data


def get_candidates_for_eval():
    """During evaluation phase only, gets the candidates. In training phase, this is done in the same function as
    for the context (get_dataset)"""
    pass


def get_context_representation_orig_blink(
        sample,
        tokenizer,
        max_seq_length,
        mention_key="mention",
        context_key="context",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
            context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def process_mention_data(
        samples,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        mention_key="mention",
        context_key="context",
        label_key="label",
        title_key='label_title',
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
        title_token=ENT_TITLE_TAG,
        debug=False,
):
    processed_samples = []

    if debug:
        samples = samples[:5]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            ent_start_token,
            ent_end_token,
        )

        label = sample['target_bert_tokenized']
        title = sample['target_title_bert_tokenized']
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )

        record = {
            'context': context_tokens,
            'label': label_tokens
        }

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            # logger.info("Src : %d" % sample["src"][0])
            # logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs
        # "label_idx": label_idx,
    }

    tensor_data = TensorDataset(context_vecs, cand_vecs)
    return data, tensor_data


def get_dataset_eval(
        tokenizer,
        biencoder: BiEncoder,
        params,
        date_cut: Dict,
        label_id_to_wikidata_qid,
        wikidata_qid_to_label_id,
        subset_name,
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG
):
    """

    This is the version that also accounts for passage candidates during prediction

    """

    max_context_length = params['max_context_length']
    dataset_path = params['data_path']
    dataset_path = os.path.join(params['base_experiment_path'], dataset_path)
    silent = params['silent']
    debug = params['debug']
    debug_size = params['debug_size']

    return_text = params.get('json_output', False)

    processed_samples = []

    use_world = True
    # TODO!!! here make sure it is in the same file!!
    # file_name = '{}_{}_random.jsonl'.format(subset_name, date_cut)
    file_name = '{}{}_{}_all.jsonl'.format(params['dataset_filename_prefix'], date_cut['mentions_cut'], subset_name)
    txt_file_path = os.path.join(dataset_path, file_name)

    if return_text:
        samples = []
    else:
        samples = None
    logger.info('getting dataset_to_predict from eval from %s' % txt_file_path)

    with io.open(txt_file_path, mode='r', encoding='utf-8') as file:
        if silent:
            iter_ = file
        else:
            iter_ = tqdm(file)
        for idx, line in enumerate(iter_):
            processed_labels_samples_in_doc = list()
            if debug and idx >= debug_size:
                break

            sample = json.loads(line.strip())

            context_tokens = get_context_representation(
                sample,
                tokenizer,
                max_context_length,
                ent_start_token,
                ent_end_token,
            )

            if return_text:
                # TODO first removes text to make it smaller then returns
                del sample['context_right_bert_tokenized']
                del sample['context_left_bert_tokenized']
                del sample['mention_bert_tokenized']
                del sample['target_bert_tokenized']
                del sample['target_title_bert_tokenized']
                samples.append(sample)

            record = {
                'context': context_tokens
            }
            record['orig_index'] = idx

            if not debug:
                if sample['target_wikidata_qid'] in wikidata_qid_to_label_id[date_cut['entities_cut']]:
                    gold_label_id = wikidata_qid_to_label_id[date_cut['entities_cut']][sample['target_wikidata_qid']]
                else:
                    logger.warning('WARNING CAN NOT FIND THE target_wikidata_qid!!! for cut, '
                                   'CHOSING random!!! SHOULD NOT HAPPEN!!! %s '
                                   ' exact qid: %s' % (str(date_cut['entities_cut']), sample['target_wikidata_qid']))
                    continue
            else:
                # if we are debugging, not all labels will be present in dictionary, so we just put a random one
                # if the gold one doesn't exist
                if sample['target_wikidata_qid'] not in wikidata_qid_to_label_id[date_cut['entities_cut']]:
                    lst_all_label_ids = list(wikidata_qid_to_label_id[date_cut['entities_cut']].values())
                    gold_label_id = random.choice(lst_all_label_ids)
                else:
                    gold_label_id = wikidata_qid_to_label_id[date_cut['entities_cut']][sample['target_wikidata_qid']]

            record['gold_label_id'] = gold_label_id

            processed_samples.append(record)

    context_vecs = torch.tensor(select_field(processed_samples, 'context', 'ids'), dtype=torch.long, )
    token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(context_vecs, NULL_IDX)
    if not params['no_cuda']:
        token_idx_ctxt = token_idx_ctxt.cuda()
        segment_idx_ctxt = segment_idx_ctxt.cuda()
        mask_ctxt = mask_ctxt.cuda()

    lst_emb_ctxts = list()
    lst_emb_toks = list()
    #######################
    test_tensor_context = TensorDataset(token_idx_ctxt, segment_idx_ctxt, mask_ctxt)

    test_sampler = SequentialSampler(test_tensor_context)
    mentions_in_context_data_loader = DataLoader(
        test_tensor_context, sampler=test_sampler, batch_size=params['encode_batch_size']
    )
    if silent:
        iter_ = mentions_in_context_data_loader
    else:
        iter_ = tqdm(mentions_in_context_data_loader)

    for idx, (curr_token_idx_ctxt, curr_segment_idx_ctxt, curr_mask_ctxt) in enumerate(iter_):
        curr_embedding_ctxt, _ = biencoder(curr_token_idx_ctxt, curr_segment_idx_ctxt, curr_mask_ctxt, None, None,
                                           None)
        curr_embedding_ctxt = curr_embedding_ctxt.detach().cpu()
        lst_emb_ctxts.append(curr_embedding_ctxt)

    embedding_ctxt = torch.cat(lst_emb_ctxts)
    logger.info('(kzaporoj 10/03/2022) - embedding_ctx final shape: %s' % str(embedding_ctxt.shape))

    # (kzaporoj 10/03/2022) - this is the part I added:
    passage_ids_field = torch.tensor(
        select_field(processed_samples, 'gold_label_id'), dtype=torch.long,
    )

    orig_index = torch.tensor(
        select_field(processed_samples, 'orig_index'), dtype=torch.long,
    )

    tensor_data_context = TensorDataset(embedding_ctxt, passage_ids_field, orig_index, context_vecs)

    return samples, tensor_data_context
