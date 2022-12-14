# Calculates / displays the results of the jsons generated by ```main_eval_faiss_biencoder_blink.py```
import argparse
import csv
import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.biencoder.misc_utils import Stats
from stats.s07_analyze_results_specific_year_v2 import get_target_filter_type
from tempel_creation.misc.utils import from_bert_to_text
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def evaluate_predictions(predictions_path, config, stats, subset_name, curr_subset_timestamp,
                         curr_finetune_timestamp,
                         model_to_evaluate,
                         saved_tuples,
                         wikidata_qid_to_include: Set = None,
                         top_k_list=None,
                         csv_writer=None):
    categories = config['categories_to_evaluate']
    logger.info('evaluating for prediction_path: %s orig_category: %s subset_name: %s curr_timestamp_str: %s' %
                (predictions_path[subset_name][curr_subset_timestamp], categories, subset_name, curr_subset_timestamp))
    logger.info('reading %s' % predictions_path[subset_name][curr_subset_timestamp])

    debug = config['debug']
    debug_length = config['debug_length']
    # opens the ground truth tuples

    #
    with open(predictions_path[subset_name][curr_subset_timestamp], 'rt') as infile:
        for idx_mention_pred, curr_line in tqdm(enumerate(infile)):
            if debug:
                if idx_mention_pred >= debug_length:
                    logger.info('breaking because of debug after processing {} mentions'.format(idx_mention_pred + 1))
                    break
            parsed_line = json.loads(curr_line)
            parsed_curr_date = datetime.strptime(curr_subset_timestamp, '%Y-%m-%dT%H:%M:%SZ')
            curr_subset_year = str(parsed_curr_date.year)
            parsed_curr_date = datetime.strptime(curr_finetune_timestamp, '%Y-%m-%dT%H:%M:%SZ')
            curr_finetune_year = str(parsed_curr_date.year)

            # if this assert is correct, then we are fine to take the data from
            #   saved_tuples
            assert saved_tuples['{}_tuples_per_year'.format(subset_name)][curr_subset_year][idx_mention_pred] \
                       ['entity_qid'] == parsed_line['gold_label_doc_id']

            target_qid = saved_tuples['{}_tuples_per_year'.format(subset_name)] \
                [curr_subset_year][idx_mention_pred]['entity_qid']
            #
            anchor_length = saved_tuples['{}_tuples_per_year'.format(subset_name)] \
                [curr_subset_year][idx_mention_pred]['anchor_len']
            target_length = saved_tuples['{}_tuples_per_year'.format(subset_name)] \
                [curr_subset_year][idx_mention_pred]['target_len']
            target_orig_title = saved_tuples['{}_tuples_per_year'.format(subset_name)] \
                [curr_subset_year][idx_mention_pred]['target_orig_title']
            target_bert_tokenized = saved_tuples['{}_tuples_per_year'.format(subset_name)] \
                [curr_subset_year][idx_mention_pred]['target_bert_tokenized']

            if wikidata_qid_to_include is not None and \
                    (parsed_line['gold_label_doc_id'] not in wikidata_qid_to_include):
                continue

            if 'all' not in categories and parsed_line['category'] not in categories:
                continue

            pointer = -1

            # first sorts the candidates depending on the type of the model
            if model_to_evaluate == 'biencoder':
                parsed_line['candidates'] = sorted(parsed_line['candidates'], key=lambda x: x['candidate_score'],
                                                   reverse=True)
            elif model_to_evaluate == 'crossencoder':
                parsed_line['candidates'] = sorted(parsed_line['candidates'], key=lambda x: x['crossencoder_score'],
                                                   reverse=True)
            else:
                raise RuntimeError('model_to_evaluate type not recognized: ' + model_to_evaluate)
            for idx_rank, curr_candidate in enumerate(parsed_line['candidates']):
                if curr_candidate['candidate_correct']:
                    pointer = idx_rank
                    break
            if 'all' in categories:
                stats[subset_name][curr_subset_timestamp][model_to_evaluate]['all'].add(pointer)

            stats[subset_name][curr_subset_timestamp][model_to_evaluate][parsed_line['category']].add(pointer)

            ##### BEGIN: length statistics calculation
            #
            row_csv = []
            row_csv.append(target_qid)
            row_csv.append(target_orig_title)
            row_csv.append(anchor_length)
            row_csv.append(target_length)
            row_csv.append(parsed_line['category'])
            row_csv.append(subset_name)
            row_csv.append(model_to_evaluate)
            row_csv.append(curr_finetune_year)
            row_csv.append(curr_subset_year)

            text_entity_tokenized = from_bert_to_text(target_bert_tokenized).strip().lower().split(' ')
            target_orig_title = target_orig_title.lower()
            type = get_target_filter_type(curr_test_year=curr_subset_year, target_orig_title=target_orig_title,
                                          config=config, text_entity_tokenized=text_entity_tokenized)

            filter_result = type
            row_csv.append(filter_result)

            for curr_k in top_k_list:
                if pointer != -1:
                    row_csv.append(1 if ((pointer + 1) <= curr_k) else 0)
                else:
                    row_csv.append(0)

            csv_writer.writerow(row_csv)
            ##### END: length statistics calculation

    return stats


def print_table(category, results, list_cuts, subset='test', model='biencoder'):
    lst_to_add_to_pandas = list()
    to_ret_str = '\\begin{table*}[h]\n'
    to_ret_str += '\\centering\n'
    to_ret_str += '\\resizebox{1.0\\textwidth}{!}{\\begin{tabular}{lccccccccc}\n'
    to_ret_str += '\\toprule\n'
    to_ret_str += '& \\multicolumn{{7}}{{c}}{{{} ({})}} \\\\ \n'.format(model, category)
    to_ret_str += '\\cmidrule(lr){2-8} \\\\ \n'
    to_ret_str += '%s & \\multicolumn{1}{c}{acc@1} & \\multicolumn{1}{c}{acc@2} & \\multicolumn{1}{c}{acc@4} & ' \
                  '\\multicolumn{1}{c}{acc@8} & \\multicolumn{1}{c}{acc@16} & \\multicolumn{1}{c}{acc@32} & ' \
                  '\\multicolumn{1}{c}{acc@64} \\\\ \n' % model
    to_ret_str += '\\toprule \n'

    for curr_list_cut in list_cuts:
        curr_year = curr_list_cut[:curr_list_cut.index('-')]
        output_json_result = results[subset][curr_list_cut][model][category].output_json()
        to_ret_str += '{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \n'. \
            format(
            curr_year,
            output_json_result['acc@1'],
            output_json_result['acc@2'],
            output_json_result['acc@4'],
            output_json_result['acc@8'],
            output_json_result['acc@16'],
            output_json_result['acc@32'],
            output_json_result['acc@64'])
        lst_to_add_to_pandas.append({'year': curr_year,
                                     'acc@1': output_json_result['acc@1'],
                                     'acc@2': output_json_result['acc@2'],
                                     'acc@4': output_json_result['acc@4'],
                                     'acc@8': output_json_result['acc@8'],
                                     'acc@16': output_json_result['acc@16'],
                                     'acc@32': output_json_result['acc@32'],
                                     'acc@64': output_json_result['acc@64']
                                     })
    to_ret_str += '\\bottomrule \n'
    to_ret_str += '\\end{tabular}} \n'
    to_ret_str += '\\caption{TODO} \n'
    to_ret_str += '\\label{tab:first_results} \n'
    to_ret_str += '\\end{table*}'

    logger.info('to_ret_str is now: %s' % to_ret_str)


def print_general_table(results, config, categories_to_evaluate=None, subset_name=None, model_name=None, all_ks=True):
    """



    :param results:
    :param config:
    :param categories_to_evaluate:
    :param subset_name:
    :param model_name:
    :param all_ks:
    :return:
    """

    #
    #     \begin{table}[t]
    #     \caption{Accuracy@64 for \emph{continual} (top) and \emph{new} (bottom) entities. The intensity of colors is set on a row-by-row basis and indicates whether performance is \textcolor{darkspringgreen!100}{\textbf{better}} or \textcolor{deepcarmine!100}{\textbf{worse}} compared to the year the model was finetuned on (\ie the values that form the white diagonal).}
    #             \label{tab:general_table_shared}
    #     \centering
    #     \resizebox{\columnwidth}{!} {
    #     \begin{tabular}{c cccccccccc}
    #              \toprule
    #              %%%% we are here copying
    #              \multicolumn{11}{c}{Continual Entities} \\
    #              \midrule
    #              \backslashbox{\textbf{Train}}{\textbf{Test}}& 2013 & 2014 & 2015 & 2016 & 2017 & 2018 & 2019 & 2020 & 2021 & 2022 \\
    #             \midrule
    #     % 2013  & 0.726 & \cellcolor{deepcarmine!17}0.721 & \cellcolor{deepcarmine!18}0.720 & \cellcolor{deepcarmine!31}0.712 & \cellcolor{deepcarmine!42}0.705 & \cellcolor{deepcarmine!46}0.702 & \cellcolor{deepcarmine!52}0.699 & \cellcolor{deepcarmine!61}0.692 & \cellcolor{deepcarmine!63}0.691 & \cellcolor{deepcarmine!75}0.684 \\
    #     2014  & \cellcolor{darkspringgreen!17}0.692 & 0.687 & \cellcolor{deepcarmine!14}0.684 & \cellcolor{deepcarmine!27}0.676 & \cellcolor{deepcarmine!41}0.666 & \cellcolor{deepcarmine!39}0.668 & \cellcolor{deepcarmine!50}0.661 & \cellcolor{deepcarmine!62}0.653 & \cellcolor{deepcarmine!63}0.652 & \cellcolor{deepcarmine!75}0.645 \\
    #     2015  & \cellcolor{darkspringgreen!20}0.683 & \cellcolor{darkspringgreen!12}0.678 & 0.677 & \cellcolor{deepcarmine!24}0.668 & \cellcolor{deepcarmine!40}0.658 & \cellcolor{deepcarmine!36}0.660 & \cellcolor{deepcarmine!49}0.652 & \cellcolor{deepcarmine!60}0.645 & \cellcolor{deepcarmine!62}0.644 & \cellcolor{deepcarmine!75}0.636 \\
    #     2016  & \cellcolor{darkspringgreen!44}\underlinepers{0.706} & \cellcolor{darkspringgreen!31}\underlinepers{0.700} & \cellcolor{darkspringgreen!26}\underlinepers{0.697} & \underlinepers{0.689} & \cellcolor{deepcarmine!28}\underlinepers{0.680} & \cellcolor{deepcarmine!27}\underlinepers{0.681} & \cellcolor{deepcarmine!37}\underlinepers{0.676} & \cellcolor{deepcarmine!55}\underlinepers{0.667} & \cellcolor{deepcarmine!61}\underlinepers{0.665} & \cellcolor{deepcarmine!75}\underlinepers{0.658} \\
    #     2017  & \cellcolor{darkspringgreen!73}0.688 & \cellcolor{darkspringgreen!58}0.683 & \cellcolor{darkspringgreen!51}0.680 & \cellcolor{darkspringgreen!32}0.673 & 0.665 & \cellcolor{deepcarmine!12}0.664 & \cellcolor{deepcarmine!33}0.657 & \cellcolor{deepcarmine!55}0.649 & \cellcolor{deepcarmine!59}0.647 & \cellcolor{deepcarmine!75}0.642 \\
    #     2018  & \cellcolor{darkspringgreen!75}0.692 & \cellcolor{darkspringgreen!58}0.686 & \cellcolor{darkspringgreen!50}0.683 & \cellcolor{darkspringgreen!30}0.676 & \cellcolor{deepcarmine!15}0.667 & 0.669 & \cellcolor{deepcarmine!29}0.662 & \cellcolor{deepcarmine!53}0.653 & \cellcolor{deepcarmine!59}0.651 & \cellcolor{deepcarmine!74}0.646 \\
    #     2019  & \cellcolor{darkspringgreen!75}0.693 & \cellcolor{darkspringgreen!63}0.688 & \cellcolor{darkspringgreen!61}0.687 & \cellcolor{darkspringgreen!42}0.678 & \cellcolor{darkspringgreen!23}0.670 & \cellcolor{darkspringgreen!23}0.669 & 0.663 & \cellcolor{deepcarmine!31}0.653 & \cellcolor{deepcarmine!32}0.653 & \cellcolor{deepcarmine!47}0.646 \\
    #     2020  & \cellcolor{darkspringgreen!75}0.701 & \cellcolor{darkspringgreen!65}0.695 & \cellcolor{darkspringgreen!62}0.693 & \cellcolor{darkspringgreen!48}0.685 & \cellcolor{darkspringgreen!32}0.676 & \cellcolor{darkspringgreen!35}0.678 & \cellcolor{darkspringgreen!21}0.670 & 0.663 & \cellcolor{deepcarmine!13}0.661 & \cellcolor{deepcarmine!26}0.654 \\
    #     2021  & \cellcolor{darkspringgreen!75}0.696 & \cellcolor{darkspringgreen!66}0.690 & \cellcolor{darkspringgreen!62}0.688 & \cellcolor{darkspringgreen!51}0.681 & \cellcolor{darkspringgreen!35}0.672 & \cellcolor{darkspringgreen!37}0.673 & \cellcolor{darkspringgreen!29}0.668 & \cellcolor{darkspringgreen!15}0.659 & 0.656 & \cellcolor{deepcarmine!19}0.650 \\
    #     2022  & \cellcolor{darkspringgreen!75}0.703 & \cellcolor{darkspringgreen!66}0.697 & \cellcolor{darkspringgreen!63}0.695 & \cellcolor{darkspringgreen!53}0.688 & \cellcolor{darkspringgreen!40}0.678 & \cellcolor{darkspringgreen!41}0.678 & \cellcolor{darkspringgreen!33}0.673 & \cellcolor{darkspringgreen!21}0.664 & \cellcolor{darkspringgreen!18}0.662 & 0.656 \\
    #             \midrule
    #             \multicolumn{11}{c}{New Entities} \\
    #              \midrule
    #              \backslashbox{\textbf{Train}}{\textbf{Test}}& 2013 & 2014 & 2015 & 2016 & 2017 & 2018 & 2019 & 2020 & 2021 & 2022 \\
    #             \midrule
    #     % 2013  & 0.748 & \cellcolor{darkspringgreen!14}0.760 & \cellcolor{deepcarmine!19}0.726 & \cellcolor{deepcarmine!19}0.726 & \cellcolor{deepcarmine!38}0.680 & \cellcolor{deepcarmine!38}0.679 & \cellcolor{deepcarmine!45}0.662 & \cellcolor{deepcarmine!35}0.687 & \cellcolor{deepcarmine!75}0.590 & \cellcolor{deepcarmine!49}0.653 \\
    #     2014  & \cellcolor{darkspringgreen!13}0.700 & 0.692 & \cellcolor{deepcarmine!21}0.663 & \cellcolor{deepcarmine!48}0.597 & \cellcolor{deepcarmine!45}0.602 & \cellcolor{deepcarmine!44}0.607 & \cellcolor{deepcarmine!46}0.601 & \cellcolor{deepcarmine!50}0.589 & \cellcolor{deepcarmine!75}0.529 & \cellcolor{deepcarmine!39}0.618 \\
    #     2015  & \cellcolor{darkspringgreen!37}0.690 & \cellcolor{darkspringgreen!25}0.672 & 0.648 & \cellcolor{deepcarmine!48}0.589 & \cellcolor{deepcarmine!42}0.597 & \cellcolor{deepcarmine!48}0.589 & \cellcolor{deepcarmine!44}0.595 & \cellcolor{deepcarmine!38}0.604 & \cellcolor{deepcarmine!75}0.548 & \cellcolor{deepcarmine!35}0.609 \\
    #     2016  & \cellcolor{darkspringgreen!60}\underlinepers{0.717} & \cellcolor{darkspringgreen!67}\underlinepers{0.725} & \cellcolor{darkspringgreen!29}0.677 & \underlinepers{0.652} & \cellcolor{deepcarmine!12}\underlinepers{0.649} & \cellcolor{deepcarmine!27}0.630 & \cellcolor{deepcarmine!38}0.616 & \cellcolor{deepcarmine!21}0.637 & \cellcolor{deepcarmine!75}0.569 & \cellcolor{deepcarmine!24}0.634 \\
    #     2017  & \cellcolor{darkspringgreen!75}0.705 & \cellcolor{darkspringgreen!73}0.703 & \cellcolor{darkspringgreen!49}0.670 & \cellcolor{darkspringgreen!13}0.621 & 0.616 & \cellcolor{deepcarmine!17}0.606 & \cellcolor{deepcarmine!26}0.594 & \cellcolor{deepcarmine!17}0.606 & \cellcolor{deepcarmine!66}0.539 & \cellcolor{deepcarmine!12}0.613 \\
    #     2018  & \cellcolor{darkspringgreen!75}0.693 & \cellcolor{darkspringgreen!68}0.685 & \cellcolor{darkspringgreen!48}0.658 & \cellcolor{deepcarmine!12}0.604 & \cellcolor{deepcarmine!11}0.605 & 0.607 & \cellcolor{deepcarmine!21}0.592 & \cellcolor{deepcarmine!19}0.595 & \cellcolor{deepcarmine!52}0.551 & \cellcolor{darkspringgreen!17}0.618 \\
    #     2019  & \cellcolor{darkspringgreen!74}0.701 & \cellcolor{darkspringgreen!75}0.702 & \cellcolor{darkspringgreen!54}0.670 & \cellcolor{deepcarmine!10}0.603 & \cellcolor{darkspringgreen!19}0.618 & \cellcolor{darkspringgreen!12}0.607 & 0.603 & \cellcolor{deepcarmine!14}0.596 & \cellcolor{deepcarmine!59}0.527 & \cellcolor{darkspringgreen!13}0.608 \\
    #     2020  & \cellcolor{darkspringgreen!70}0.716 & \cellcolor{darkspringgreen!75}0.722 & \cellcolor{darkspringgreen!54}\underlinepers{0.695} & \cellcolor{darkspringgreen!10}0.635 & \cellcolor{darkspringgreen!10}0.635 & \cellcolor{deepcarmine!11}\underlinepers{0.633} & \cellcolor{deepcarmine!18}\underlinepers{0.624} & 0.635 & \cellcolor{deepcarmine!62}0.565 & \cellcolor{deepcarmine!16}0.627 \\
    #     2021  & \cellcolor{darkspringgreen!75}0.705 & \cellcolor{darkspringgreen!72}0.698 & \cellcolor{darkspringgreen!57}0.664 & \cellcolor{darkspringgreen!34}0.609 & \cellcolor{darkspringgreen!39}0.622 & \cellcolor{darkspringgreen!34}0.609 & \cellcolor{darkspringgreen!30}0.600 & \cellcolor{darkspringgreen!41}0.625 & 0.552 & \cellcolor{darkspringgreen!49}\underlinepers{0.645} \\
    #     2022  & \cellcolor{darkspringgreen!75}0.716 & \cellcolor{darkspringgreen!71}0.712 & \cellcolor{darkspringgreen!47}0.685 & \cellcolor{deepcarmine!12}0.639 & \cellcolor{deepcarmine!12}0.639 & \cellcolor{deepcarmine!17}\underlinepers{0.633} & \cellcolor{deepcarmine!27}0.622 & \cellcolor{deepcarmine!12}\underlinepers{0.639} & \cellcolor{deepcarmine!72}\underlinepers{0.570} & 0.642 \\
    #             \bottomrule
    #             \end{tabular}
    #     \end{table}
    min_intensity = 10
    max_intensity = 75
    range_intensity = max_intensity - min_intensity

    cuts_to_show = config['cuts_to_show']

    fine_tune_cuts = config['fine_tune_cuts']
    top_k_list = config['top_k_list']

    ks_to_iterate = top_k_list
    if not all_ks:
        ks_to_iterate = [max(top_k_list)]

    # cuts_with_results --> in years
    for curr_k in ks_to_iterate:
        to_ret_str = ''

        cuts_with_results = set()
        for curr_fine_tune_cut in fine_tune_cuts:
            curr_timestamp = curr_fine_tune_cut['timestamp']
            cuts_with_results.add(curr_timestamp)

        max_acc_per_fine_tune_cut = dict()
        min_acc_per_fine_tune_cut = dict()
        central_accuracies_per_fine_tune_cut = dict()
        max_accuracy = 0.0
        min_accuracy = 100.0

        for idx_category, curr_category_to_evaluate in enumerate(categories_to_evaluate):
            central_accuracies_per_fine_tune_cut[curr_category_to_evaluate] = dict()
            max_acc_per_fine_tune_cut[curr_category_to_evaluate] = dict()
            min_acc_per_fine_tune_cut[curr_category_to_evaluate] = dict()
            for curr_fine_tune_cut in cuts_to_show:
                max_acc_per_fine_tune_cut[curr_category_to_evaluate][curr_fine_tune_cut] = 0.0
                min_acc_per_fine_tune_cut[curr_category_to_evaluate][curr_fine_tune_cut] = 100.0
                for curr_predicted_cut in cuts_to_show:
                    #
                    if subset_name in results[curr_fine_tune_cut]:
                        curr_res = results[curr_fine_tune_cut][subset_name][curr_predicted_cut][model_name][
                            curr_category_to_evaluate]
                        curr_res = curr_res.output_json()
                        curr_acc = curr_res['acc@{}'.format(curr_k)]
                    else:
                        curr_acc = 0.0
                    max_acc_per_fine_tune_cut[curr_category_to_evaluate][curr_fine_tune_cut] = max(curr_acc,
                                                                                                   max_acc_per_fine_tune_cut[
                                                                                                       curr_category_to_evaluate][
                                                                                                       curr_fine_tune_cut])
                    min_acc_per_fine_tune_cut[curr_category_to_evaluate][curr_fine_tune_cut] = min(curr_acc,
                                                                                                   min_acc_per_fine_tune_cut[
                                                                                                       curr_category_to_evaluate][
                                                                                                       curr_fine_tune_cut])
                    min_accuracy = min(curr_acc, min_accuracy)
                    max_accuracy = max(curr_acc, max_accuracy)
                    if curr_fine_tune_cut == curr_predicted_cut:
                        central_accuracies_per_fine_tune_cut[curr_category_to_evaluate][curr_fine_tune_cut] = curr_acc

        to_ret_str += '\\begin{table}[t] \n'
        to_ret_str += '\\caption{Accuracy@%s for \\emph{continual} (top) and \\emph{new} (bottom) entities. ' \
                      'The intensity of colors is set on a row-by-row basis and indicates whether performance is ' \
                      '\\textcolor{darkspringgreen!100}{\\textbf{better}} or ' \
                      '\\textcolor{deepcarmine!100}{\\textbf{worse}} compared to the year the model was finetuned on ' \
                      '(\\ie the values that form the white diagonal).} \n' % curr_k
        to_ret_str += '\\label{tab:general_table_shared_k_in_%s} \n' % curr_k
        to_ret_str += '\\centering \n'
        to_ret_str += '\\resizebox{\columnwidth}{!} \n'
        to_ret_str += '{\\begin{tabular}{c ' + 'c' * len(cuts_to_show) + '} \n'
        to_ret_str += '\t \\toprule \n'

        for idx_category, curr_category_to_evaluate in enumerate(categories_to_evaluate):
            # to_ret_str += '\t'
            if idx_category > 0:
                to_ret_str += '\t\\midrule \n'
            if curr_category_to_evaluate in {'random', 'shared', 'continual'}:
                to_ret_str += '\t \\multicolumn{11}{c}{Continual Entities} \\\\ \n'
                to_ret_str += '\t \\midrule \n'
            elif curr_category_to_evaluate in {'new'}:
                to_ret_str += '\t \\multicolumn{11}{c}{New Entities} \\\\ \n'
                to_ret_str += '\t \\midrule \n'

            to_ret_str += '\t \\backslashbox{\\textbf{Train}}{\\textbf{Test}}'
            for curr_cut in cuts_to_show:
                parsed_curr_date = datetime.strptime(curr_cut, '%Y-%m-%dT%H:%M:%SZ')
                curr_year = parsed_curr_date.year
                # '& 2013 & 2014 & 2015 & 2016 & 2017 & 2018 & 2019 & 2020  & 2021  & 2022 \\'
                to_ret_str += '& {} '.format(curr_year)
            to_ret_str += '\\\\ \n'
            to_ret_str += '\t\\midrule \n'
            for curr_fine_tune_cut in cuts_to_show:
                parsed_curr_date = datetime.strptime(curr_fine_tune_cut, '%Y-%m-%dT%H:%M:%SZ')
                curr_year = parsed_curr_date.year
                to_ret_str += '%s ' % curr_year
                if curr_fine_tune_cut in cuts_with_results:
                    max_acc_pf = max_acc_per_fine_tune_cut[curr_category_to_evaluate][curr_fine_tune_cut]
                    min_acc_pf = min_acc_per_fine_tune_cut[curr_category_to_evaluate][curr_fine_tune_cut]
                    for curr_predicted_cut in cuts_to_show:
                        curr_res = results[curr_fine_tune_cut][subset_name][curr_predicted_cut][model_name][
                            curr_category_to_evaluate]
                        curr_res = curr_res.output_json()
                        if curr_fine_tune_cut == curr_predicted_cut:
                            to_ret_str += (' & %.3f' % curr_res['acc@{}'.format(curr_k)])
                        else:
                            curr_acc = curr_res['acc@{}'.format(curr_k)]
                            max_delta = central_accuracies_per_fine_tune_cut[curr_category_to_evaluate][
                                            curr_fine_tune_cut] - min_acc_pf
                            max_delta = max(max_delta,
                                            max_acc_pf -
                                            central_accuracies_per_fine_tune_cut[curr_category_to_evaluate][
                                                curr_fine_tune_cut])

                            if curr_acc < central_accuracies_per_fine_tune_cut[curr_category_to_evaluate][
                                curr_fine_tune_cut]:
                                curr_delta = central_accuracies_per_fine_tune_cut[curr_category_to_evaluate][
                                                 curr_fine_tune_cut] - curr_acc
                                intensity_name = 'deepcarmine'
                            else:
                                curr_delta = curr_acc - central_accuracies_per_fine_tune_cut[curr_category_to_evaluate][
                                    curr_fine_tune_cut]
                                intensity_name = 'darkspringgreen'

                            ratio_max_delta = curr_delta / max_delta
                            curr_range_intensity = int(range_intensity * ratio_max_delta)
                            curr_intensity = min_intensity + curr_range_intensity
                            to_ret_str += (' & \cellcolor{%s!%s}%.3f' % (intensity_name, curr_intensity, curr_acc))
                else:
                    for _ in cuts_to_show:
                        to_ret_str += ' & -'
                to_ret_str += ' \\\\ \n'
        to_ret_str += '\t\\bottomrule \n'
        to_ret_str += '\t\\end{tabular}} \n'
        to_ret_str += '\\end{table}\n'
        logger.info('++++++++++++++++++++++ K = {} ++++++++++++++++++++++'.format(curr_k))
        logger.info('generic table: categories %s subset_name %s model_name %s' %
                    (categories_to_evaluate, subset_name, model_name))
        logger.info(to_ret_str)


def serialize_results(stats, config):
    cuts_with_results = set()
    fine_tune_cuts = config['fine_tune_cuts']
    for curr_fine_tune_cut in fine_tune_cuts:
        curr_timestamp = curr_fine_tune_cut['timestamp']
        cuts_with_results.add(curr_timestamp)

    serialized_csv_results = config['serialized_csv_results']
    cuts_to_show = config['cuts_to_show']
    lst_pandas = list()
    for curr_model in config['models']:
        for curr_subset in config['subsets']:
            for curr_category_to_evaluate in config['categories_to_evaluate']:
                for curr_fine_tune_cut in cuts_to_show:
                    if curr_fine_tune_cut in cuts_with_results:
                        for curr_predicted_cut in cuts_to_show:
                            parsed_curr_date = datetime.strptime(curr_predicted_cut, '%Y-%m-%dT%H:%M:%SZ')
                            curr_predicted_cut_year = int(parsed_curr_date.year)

                            parsed_curr_date = datetime.strptime(curr_fine_tune_cut, '%Y-%m-%dT%H:%M:%SZ')
                            curr_finetune_cut_year = int(parsed_curr_date.year)

                            curr_json_results = \
                                stats[curr_fine_tune_cut][curr_subset['subset_name']][curr_predicted_cut][curr_model] \
                                    [curr_category_to_evaluate].output_json()
                            #
                            lst_pandas.append(
                                {'acc@1': curr_json_results['acc@1'],
                                 'acc@2': curr_json_results['acc@2'],
                                 'acc@4': curr_json_results['acc@4'],
                                 'acc@8': curr_json_results['acc@8'],
                                 'acc@16': curr_json_results['acc@16'],
                                 'acc@32': curr_json_results['acc@32'],
                                 'acc@64': curr_json_results['acc@64'],
                                 'year_finetune': curr_finetune_cut_year,
                                 'subset': curr_subset['subset_name'],
                                 'category': curr_category_to_evaluate,
                                 'year_predicted': curr_predicted_cut_year,
                                 'model': curr_model
                                 })
    results_df = pd.DataFrame(lst_pandas)
    dir_csv = os.path.dirname(serialized_csv_results)
    os.makedirs(dir_csv, exist_ok=True)
    results_df.to_csv(serialized_csv_results, sep='\t', index=False)


def print_result_tables(results, config):
    """

    :param results:
    :param config:
    :return:
    """

    for curr_model in config['models']:
        for curr_subset in config['subsets']:
            print_general_table(results=results,
                                config=config,
                                categories_to_evaluate=config['categories_to_evaluate'],
                                subset_name=curr_subset['subset_name'],
                                model_name=curr_model)


if __name__ == '__main__':
    # import pdb
    #
    # pdb.set_trace()

    parser = argparse.ArgumentParser(description='Evaluate biencoder\'s output')
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/data_export/wikipedia/dataset_creator/20220220/config/'
                                'calculate_results_and_metrics_local.json',
                        help='The config file that contains all the parameters')
    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))

    gold_context_mention_to_label = dict()
    # --
    ground_truth_path = dict()
    predictions_path = dict()

    subsets = config['subsets']
    serialized_gold_path = config['serialized_gold_path']
    os.makedirs(os.path.dirname(serialized_gold_path), exist_ok=True)
    serialized_filtering_path = config['serialized_filtering_path']
    os.makedirs(os.path.dirname(serialized_filtering_path), exist_ok=True)

    input_dir_dataset = config['input_dir_dataset']
    #
    stats = dict()
    top_k_list = config['top_k_list']
    top_k = max(top_k_list)
    list_cuts = list()
    dict_finetune_cuts = dict()
    list_show_cuts = config['cuts_to_show']
    for curr_list_show_cuts in list_show_cuts:
        stats[curr_list_show_cuts] = dict()

    logger.info('loading ground truth...')
    start_loading_gt = time.time()
    if not os.path.exists(serialized_gold_path):
        # loads the ground truth in subsets
        for curr_subset in subsets:
            subset_name = curr_subset['subset_name']
            gold_context_mention_to_label[subset_name] = dict()
            ground_truth_path[subset_name] = dict()

            for curr_subset_time_cut in curr_subset['subset_cuts']:
                curr_timestamp_str = curr_subset_time_cut['timestamp']
                curr_input_file_name = curr_subset_time_cut['input_file_name']
                ground_truth_path[subset_name][curr_timestamp_str] = os.path.join(input_dir_dataset,
                                                                                  curr_input_file_name)

                gold_context_mention_to_label[subset_name][curr_timestamp_str] = dict()
                idx_mention_ground_truth = 0
                logger.info('loading into gold_context_mention_to_label from: %s' %
                            ground_truth_path[subset_name][curr_timestamp_str])
                with open(ground_truth_path[subset_name][curr_timestamp_str], 'rt') as infile:
                    logger.info('reading %s' % ground_truth_path[subset_name][curr_timestamp_str])
                    for idx_mention_ground_truth, curr_line in tqdm(enumerate(infile)):
                        parsed_line = json.loads(curr_line)
                        ctxt_doc_id = parsed_line['anchor_wikidata_qid']
                        ctxt_mention_id = idx_mention_ground_truth
                        gold_label_doc_id = parsed_line['target_wikidata_qid']
                        gold_context_mention_to_label[subset_name][curr_timestamp_str] \
                            ['{}_{}'.format(ctxt_doc_id, ctxt_mention_id)] = \
                            {
                                'gold_doc_id': gold_label_doc_id,
                                'orig_category': parsed_line['category']
                            }
        pickle.dump(gold_context_mention_to_label, open(serialized_gold_path, 'wb'))
    else:
        logger.info('pickle detected, loading')
        gold_context_mention_to_label = pickle.load(open(serialized_gold_path, 'rb'))

    end_loading_gt = time.time()

    logger.info('%s mins to load ground truth...' % ((end_loading_gt - start_loading_gt) / 60))

    # if filter is activated, then gets some statistics from the training set of each of the cuts
    #  first , this can be serialized later on I guess
    dict_cut_to_target_qid_to_nr_mentions = dict()
    dict_cut_to_top_mentions_target_qids = dict()
    if config['filter_train']['top_most_targeted_entities'] > 0:
        logger.info('loading extra info for filtering')
        if not os.path.exists(serialized_filtering_path):
            logger.info('not serialized, loading dictionary for filtering')
            for curr_fine_tune_cut in config['fine_tune_cuts']:
                dict_cut_to_target_qid_to_nr_mentions[curr_fine_tune_cut['timestamp']] = dict()
                logger.info('loading %s' % curr_fine_tune_cut['input_file_train_set'])
                for curr_line in tqdm(open(curr_fine_tune_cut['input_file_train_set'], 'rt')):
                    parsed_line = json.loads(curr_line)
                    target_wikidata_qid = parsed_line['target_wikidata_qid']
                    if target_wikidata_qid not in dict_cut_to_target_qid_to_nr_mentions[
                        curr_fine_tune_cut['timestamp']]:
                        dict_cut_to_target_qid_to_nr_mentions[curr_fine_tune_cut['timestamp']][target_wikidata_qid] = 0
                    dict_cut_to_target_qid_to_nr_mentions[curr_fine_tune_cut['timestamp']][target_wikidata_qid] += 1
            pickle.dump(dict_cut_to_target_qid_to_nr_mentions, open(serialized_filtering_path, 'wb'))
        else:
            logger.info('filtering dictionary already serialized, unpickling')
            dict_cut_to_target_qid_to_nr_mentions = pickle.load(open(serialized_filtering_path, 'rb'))
            logger.info('DONE unpickling')

        top_most_targeted_entities = config['filter_train']['top_most_targeted_entities']
        for curr_cut, target_qid_to_nr_mentions in dict_cut_to_target_qid_to_nr_mentions.items():
            top_target_qids = [k for k, v in sorted(target_qid_to_nr_mentions.items(), key=lambda item: item[1],
                                                    reverse=True)]
            top_target_qids = top_target_qids[:top_most_targeted_entities]
            top_target_qids = set(top_target_qids)
            dict_cut_to_top_mentions_target_qids[curr_cut] = top_target_qids

    #
    out_file_csv_detail_stats = open(config['serialized_csv_detail_stats'], 'wt')

    csv_writer = csv.writer(out_file_csv_detail_stats, delimiter='\t')
    written_csv_header = False
    # calculates the actual results
    lst_to_add_to_pandas = list()
    cache_tuples_path = config['cache_tuples_path']
    #
    saved_tuples = pickle.load(open(cache_tuples_path, 'rb'))
    logger.info('tuples loaded')

    for curr_model in config['models']:
        for curr_fine_tune_cut in config['fine_tune_cuts']:
            input_dir_predictions = curr_fine_tune_cut['input_dir_predictions']
            logger.info('EVALUATING predictions in %s' % input_dir_predictions)
            if curr_fine_tune_cut['timestamp'] not in dict_finetune_cuts:
                dict_finetune_cuts[curr_fine_tune_cut['timestamp']] = None

            curr_finetune_timestamp = curr_fine_tune_cut['timestamp']
            for curr_subset in subsets:
                subset_name = curr_subset['subset_name']

                predictions_path[subset_name] = dict()
                stats[curr_finetune_timestamp][subset_name] = dict()

                for curr_subset_time_cut in curr_subset['subset_cuts']:
                    # YES, I KNOW IT IS O(N), but it is a very short list
                    if curr_subset_time_cut['timestamp'] not in list_cuts:
                        list_cuts.append(curr_subset_time_cut['timestamp'])
                    curr_timestamp_str = curr_subset_time_cut['timestamp']
                    curr_pred_file_name = curr_subset_time_cut['pred_file_name']
                    # --
                    predictions_path[subset_name][curr_timestamp_str] = os.path.join(input_dir_predictions,
                                                                                     curr_pred_file_name)
                    # --
                    stats[curr_finetune_timestamp][subset_name][curr_timestamp_str] = dict()
                    stats[curr_finetune_timestamp][subset_name][curr_timestamp_str][curr_model] = dict()

                    for curr_category_to_evaluate in config['categories_to_evaluate']:
                        stats[curr_finetune_timestamp][subset_name][curr_timestamp_str][curr_model] \
                            [curr_category_to_evaluate] = Stats(top_k=top_k, rank=top_k_list)

                    wikidata_qid_to_include = None
                    if curr_timestamp_str in dict_cut_to_top_mentions_target_qids:
                        wikidata_qid_to_include = dict_cut_to_top_mentions_target_qids[curr_timestamp_str]

                    if not written_csv_header:
                        row_header = ['target_qid', 'target_title', 'anchor_length', 'target_length',
                                      'target_entity_type', 'subset',
                                      'model', 'finetune_year', 'subset_year', 'target_entity_filter']
                        #
                        for k_acc in config['top_k_list']:
                            row_header.append('acc@{}'.format(k_acc))
                        #
                        csv_writer.writerow(row_header)
                        written_csv_header = True
                    evaluate_predictions(predictions_path, config,
                                         stats[curr_finetune_timestamp], subset_name,
                                         curr_timestamp_str,
                                         curr_finetune_timestamp,
                                         model_to_evaluate=curr_model,
                                         saved_tuples=saved_tuples,
                                         wikidata_qid_to_include=wikidata_qid_to_include,
                                         top_k_list=top_k_list,
                                         csv_writer=csv_writer)

                    logger.info('len of lst_to_add_to_pandas: %s' % len(lst_to_add_to_pandas))
            for curr_category in config['categories_to_evaluate']:
                print_table(category=curr_category, results=stats[curr_finetune_timestamp], list_cuts=list_cuts,
                            subset='test', model=curr_model)

    #
    out_file_csv_detail_stats.flush()
    out_file_csv_detail_stats.close()
    logger.info('reading saved csv from disc to pandas dataframe')
    df_to_subsample = pd.read_csv(open(config['serialized_csv_detail_stats'], 'rt'), sep='\t', header=[0])
    logger.info('finished reading saved csv from disc to pandas dataframe, now subsampling')
    df_to_subsample = df_to_subsample.loc[np.random.choice(df_to_subsample.index, 100000, replace=False)]
    logger.info('finished subsampling saved csv from disc to pandas dataframe, now saving back to disc')
    df_to_subsample.to_csv(config['serialized_csv_detail_stats_subsampled'], sep='\t', index=False)
    logger.info('finished saving subsampled dataframe to disc')
    #
    serialize_results(stats, config)
    print_result_tables(stats, config)
