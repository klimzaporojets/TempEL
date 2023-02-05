# one of the hypothesis is that the general decrease of performance is because the candidates
# become more similar between each other, more granular to predict.
# This script will produce output of three graphs:
#   1- (averaged) jackard similarity between consecutive pairs of candidates (ordered by bi-encoder
#   similarity score).
#
# THIS MODULE ONLY DEALS WITH JACCARD SIMILARITY only for now, dot product can potentially be added

import argparse
import json
import logging
import os
import pickle
import traceback

import psutil
from tqdm import tqdm

from src.tempel_creation.misc.article_queue import ArticleReadingQueue
from src.tempel_creation.misc.utils import from_bert_to_text
from src.utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def get_curr_mem_usage(process):
    return process.memory_info().rss / 1024 / 1024 / 1024


def get_content_entities(config, v_nr_entities, arq: ArticleReadingQueue):
    evolution_content_path = config['input_file_wikipedia_evolution_content']

    with open(evolution_content_path, 'rt', encoding='utf-8') as infile:
        logger.info('opening evolution file: %s' % evolution_content_path)
        for idx_line, curr_evolution_line in enumerate(tqdm(infile)):
            v_nr_entities.value += 1
            if idx_line % 100000 == 0:
                logger.info('===========================================================')
                mem_usage = get_curr_mem_usage(process)
                logger.info('%s used memory while reading evolution: %s Gb' % (idx_line, mem_usage))
                logger.info('===========================================================')
            arq.article_queue.put(curr_evolution_line)


def receive_content_entities(config, dict_qid_to_cuts, arq: ArticleReadingQueue, v_shutdown_receivers):
    context_len = config['context_len']
    while not (arq.article_queue.empty() and v_shutdown_receivers.value == 1):
        try:
            curr_evolution_line = arq.article_queue.get(block=True, timeout=1)
            parsed_line = json.loads(curr_evolution_line)
            wikidata_qid = parsed_line['wikidata_qid']
            if wikidata_qid in dict_qid_to_cuts:
                cuts_to_look_for = dict_qid_to_cuts[wikidata_qid]
                for curr_cut, curr_cut_content in parsed_line['evolution'].items():
                    if curr_cut in cuts_to_look_for:
                        tokenized_text = from_bert_to_text(curr_cut_content['bert_tokenized'][:context_len])
                        arq.evolution_content_queue.put((wikidata_qid, curr_cut, tokenized_text))

        except Exception as e:
            logger.error('Error getting curr file path from receive_content_entities, stack %s' %
                         traceback.format_exc())
            continue


def receive_final_dict(config, arq: ArticleReadingQueue, v_shutdown_writer):
    output_serialized_path = config['output_serialized_path']
    dir_where_serialize = os.path.dirname(output_serialized_path)
    os.makedirs(dir_where_serialize, exist_ok=True)
    dict_qid_to_cuts_to_content = dict()
    while not (arq.evolution_content_queue.empty() and v_shutdown_writer.value == 1):
        try:
            curr_evolution_content = arq.evolution_content_queue.get(block=True, timeout=1)
            wikidata_qid = curr_evolution_content[0]
            curr_cut = curr_evolution_content[1]
            tokenized_text = curr_evolution_content[2]
            if wikidata_qid not in dict_qid_to_cuts_to_content:
                dict_qid_to_cuts_to_content[wikidata_qid] = dict()
            dict_qid_to_cuts_to_content[wikidata_qid][curr_cut] = tokenized_text

        except Exception as e:
            logger.error('Error getting curr file path from receive_final_dict, stack: %s' % traceback.format_exc())
            continue
    pickle.dump(dict_qid_to_cuts_to_content, open(output_serialized_path, 'wb'))


if __name__ == '__main__':

    process = psutil.Process(os.getpid())

    # import pdb
    #
    # pdb.set_trace()

    parser = argparse.ArgumentParser(description='Evaluate biencoder\'s output')
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/dataset_creation/20220515/config/models/blink/'
                                'biencoder/evaluate/step07_analyze_similarity_of_predicted_candidates.json',
                        help='The config file that contains all the parameters')
    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))

    dict_qid_to_cuts = dict()

    evolution_content_path = config['input_file_wikipedia_evolution_content']

    dict_qid_to_cuts_to_content = dict()

    # max len context IN BERT TOKENS (so also use the function that passes from bert to normal tokens)
    output_serialized_path = config['output_serialized_path']
    if not os.path.exists(output_serialized_path):
        for curr_fine_tune_cut in config['fine_tune_cuts']:
            logger.info('===========================================================')
            mem_usage = get_curr_mem_usage(process)
            logger.info('used memory: %s Gb' % mem_usage)
            logger.info('===========================================================')
            logger.info('processing for %s' % curr_fine_tune_cut['input_dir_predictions'])
            for curr_subset_cut in config['subset_cuts']:
                curr_cut = curr_subset_cut['timestamp']
                curr_pred_file_path = os.path.join(curr_fine_tune_cut['input_dir_predictions'],
                                                   curr_subset_cut['pred_file_name'])
                logger.info('reading %s' % curr_pred_file_path)
                with open(curr_pred_file_path, 'rt') as infile:
                    for curr_line in infile:
                        parsed_line = json.loads(curr_line)
                        if parsed_line['category'] == 'continual' or parsed_line['category'] == 'continual':
                            candidates = parsed_line['candidates']
                            for curr_candidate in candidates:
                                # this candidate_doc_id is predicted target wikidata qid
                                candidate_doc_id = curr_candidate['candidate_doc_id']
                                if candidate_doc_id not in dict_qid_to_cuts:
                                    dict_qid_to_cuts[candidate_doc_id] = set()
                                dict_qid_to_cuts[candidate_doc_id].add(curr_cut)

        context_len = config['context_len']
        with open(evolution_content_path, 'rt', encoding='utf-8') as infile:
            logger.info('opening evolution file: %s' % evolution_content_path)
            for idx_line, curr_evolution_line in enumerate(tqdm(infile)):
                if idx_line % 100000 == 0:
                    logger.info('===========================================================')
                    mem_usage = get_curr_mem_usage(process)
                    logger.info('%s used memory while reading evolution: %s Gb' % (idx_line, mem_usage))
                    logger.info('===========================================================')

                parsed_line = json.loads(curr_evolution_line)
                wikidata_qid = parsed_line['wikidata_qid']
                if wikidata_qid in dict_qid_to_cuts:
                    dict_qid_to_cuts_to_content[wikidata_qid] = dict()
                    cuts_to_look_for = dict_qid_to_cuts[wikidata_qid]
                    for curr_cut, curr_cut_content in parsed_line['evolution'].items():
                        if curr_cut in cuts_to_look_for:
                            tokenized_text = from_bert_to_text(curr_cut_content['bert_tokenized'][:context_len])
                            dict_qid_to_cuts_to_content[wikidata_qid][curr_cut] = tokenized_text

        pickle.dump(dict_qid_to_cuts_to_content, open(output_serialized_path, 'wb'))
        logger.info('saved dictionary, now everything is fine!!')

    else:
        logger.info('ALREADY PICKLED, UNPICKLING')
        dict_qid_to_cuts_to_content = pickle.load(open(output_serialized_path, 'rb'))
        logger.info('DONE UNPICKLING')
    # step 03: now do the comparison (if everything fits in memory :P).
    dict_cut_to_avg_jaccard_diffs_micro = dict()  # between consecutive entities
    dict_cut_to_avg_jaccard_diffs_macro = dict()  # between consecutive entities

    dict_cut_to_avg_dotp_diffs_micro = dict()  # between consecutive entities
    dict_cut_to_avg_dotp_diffs_macro = dict()  # between consecutive entities

    for curr_fine_tune_cut in config['fine_tune_cuts']:
        curr_fine_tune_timestamp = curr_fine_tune_cut['timestamp']
        logger.info('===========================================================')
        mem_usage = get_curr_mem_usage(process)
        logger.info('step 03 - used memory: %s Gb' % mem_usage)
        logger.info('===========================================================')
        for curr_subset_cut in config['subset_cuts']:
            curr_cut = curr_subset_cut['timestamp']
            if curr_cut not in dict_cut_to_avg_jaccard_diffs_micro:
                dict_cut_to_avg_jaccard_diffs_micro[curr_cut] = list()
            if curr_cut not in dict_cut_to_avg_jaccard_diffs_macro:
                dict_cut_to_avg_jaccard_diffs_macro[curr_cut] = dict()
            if curr_fine_tune_timestamp not in dict_cut_to_avg_jaccard_diffs_macro[curr_cut]:
                dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_tune_timestamp] = list()

            curr_pred_file_path = os.path.join(curr_fine_tune_cut['input_dir_predictions'],
                                               curr_subset_cut['pred_file_name'])
            logger.info('reading %s' % curr_pred_file_path)
            with open(curr_pred_file_path, 'rt') as infile:
                for curr_line in tqdm(infile):
                    parsed_line = json.loads(curr_line)
                    if parsed_line['category'] == 'continual' or parsed_line['category'] == 'continual':
                        candidates = parsed_line['candidates']
                        min_cut_length = \
                            min([len(dict_qid_to_cuts_to_content[cc['candidate_doc_id']][curr_cut].split(' ')) \
                                 for cc in candidates])
                        prev_cand_text_tokens = None
                        jackard_dists = list()
                        for curr_candidate in candidates:
                            # this candidate_doc_id is predicted target wikidata qid
                            curr_cand_text_tokens = \
                                dict_qid_to_cuts_to_content[curr_candidate['candidate_doc_id']][curr_cut].strip(). \
                                    split(' ')
                            curr_cand_text_tokens = curr_cand_text_tokens[:min_cut_length]
                            curr_cand_text_tokens = set(curr_cand_text_tokens)
                            candidate_doc_id = curr_candidate['candidate_doc_id']
                            if prev_cand_text_tokens is not None:
                                intersec = curr_cand_text_tokens.intersection(prev_cand_text_tokens)
                                union = curr_cand_text_tokens.union(prev_cand_text_tokens)
                                jackard_dist_avg = len(intersec) / len(union)
                                jackard_dists.append(jackard_dist_avg)
                            prev_cand_text_tokens = curr_cand_text_tokens

                        avg_jackard_dist = sum(jackard_dists) / len(jackard_dists)
                        dict_cut_to_avg_jaccard_diffs_micro[curr_cut].append(avg_jackard_dist)
                        dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_tune_timestamp].append(avg_jackard_dist)
            logger.info('-----')
            logger.info('execution jackard stat so far (micro): ')
            for curr_cut in sorted(dict_cut_to_avg_jaccard_diffs_micro.keys()):
                curr_avg_jackard = sum(dict_cut_to_avg_jaccard_diffs_micro[curr_cut]) / len(
                    dict_cut_to_avg_jaccard_diffs_micro[curr_cut])
                logger.info('avg jackard micro for %s : %s' % (curr_cut, curr_avg_jackard))
            logger.info('execution jackard stat so far (macro): ')
            for curr_cut in sorted(dict_cut_to_avg_jaccard_diffs_macro.keys()):
                curr_cut_macros = list()
                for curr_fine_t_cut in dict_cut_to_avg_jaccard_diffs_macro[curr_cut].keys():
                    curr_avg_jackard = sum(dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_t_cut]) / len(
                        dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_t_cut])
                    curr_cut_macros.append(curr_avg_jackard)
                curr_avg_jackard_macro = sum(curr_cut_macros) / len(curr_cut_macros)
                logger.info('avg jackard macro for %s: %s' % (curr_cut, curr_avg_jackard_macro))
            logger.info('-----')

        logger.info('----------')
        logger.info('PARTIAL JACKARD DIFFERENCE MICRO: ')
        for curr_cut in sorted(dict_cut_to_avg_jaccard_diffs_micro.keys()):
            curr_avg_jackard = sum(dict_cut_to_avg_jaccard_diffs_micro[curr_cut]) / len(
                dict_cut_to_avg_jaccard_diffs_micro[curr_cut])
            logger.info('avg jackard micro for %s: %s' % (curr_cut, curr_avg_jackard))

        logger.info('PARTIAL JACKARD DIFFERENCE MACRO: ')
        for curr_cut in sorted(dict_cut_to_avg_jaccard_diffs_macro.keys()):
            curr_cut_macros = list()
            for curr_fine_t_cut in dict_cut_to_avg_jaccard_diffs_macro[curr_cut].keys():
                curr_avg_jackard = sum(dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_t_cut]) / len(
                    dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_t_cut])
                curr_cut_macros.append(curr_avg_jackard)
            curr_avg_jackard_macro = sum(curr_cut_macros) / len(curr_cut_macros)
            logger.info('avg jackard macro for %s: %s' % (curr_cut, curr_avg_jackard_macro))

        logger.info('-----------')

    logger.info('===================================================================')
    logger.info('FINAL JACKARD DIFFERENCE MICRO: ')
    for curr_cut in sorted(dict_cut_to_avg_jaccard_diffs_micro.keys()):
        curr_avg_jackard = sum(dict_cut_to_avg_jaccard_diffs_micro[curr_cut]) / len(
            dict_cut_to_avg_jaccard_diffs_micro[curr_cut])
        logger.info('avg jackard micro for %s: %s' % (curr_cut, curr_avg_jackard))

    logger.info('FINAL JACKARD DIFFERENCE MACRO: ')
    for curr_cut in sorted(dict_cut_to_avg_jaccard_diffs_macro.keys()):
        curr_cut_macros = list()
        for curr_fine_t_cut in dict_cut_to_avg_jaccard_diffs_macro[curr_cut].keys():
            curr_avg_jackard = sum(dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_t_cut]) / len(
                dict_cut_to_avg_jaccard_diffs_macro[curr_cut][curr_fine_t_cut])
            curr_cut_macros.append(curr_avg_jackard)
        curr_avg_jackard_macro = sum(curr_cut_macros) / len(curr_cut_macros)
        logger.info('avg jackard macro for %s: %s' % (curr_cut, curr_avg_jackard_macro))

    logger.info('===================================================================')
