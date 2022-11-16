# v2 different in that we want to separate three cases:
#  1- recurrent
#  2- covid-related
#  3- others

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Set

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from models.biencoder.misc_utils import Stats
from tempel_creation.misc.utils import from_bert_to_text
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def evaluate_predictions(predictions_path, config, stats, subset_name, curr_timestamp_str,
                         model_to_evaluate,
                         dict_filtered_to_cut_to_doc_ids,
                         wikidata_qid_to_include: Set = None,
                         stats_mean=None, stats_mean_category=None, top_k=None, top_k_list=None):
    logger.info('evaluating for prediction_path: %s subset_name: %s curr_timestamp_str: %s ' %
                (predictions_path[subset_name][curr_timestamp_str], subset_name, curr_timestamp_str))
    logger.info('reading %s ' % predictions_path[subset_name][curr_timestamp_str])
    nr_covid_entities = 0
    nr_recurrent_entities = 0
    nr_other_entities = 0
    already_counted = set()
    with open(predictions_path[subset_name][curr_timestamp_str], 'rt') as infile:
        for idx_mention_pred, curr_line in tqdm(enumerate(infile)):
            parsed_line = json.loads(curr_line)

            if wikidata_qid_to_include is not None and \
                    (parsed_line['gold_label_doc_id'] not in wikidata_qid_to_include):
                continue

            gold_doc_id = parsed_line['gold_label_doc_id']

            pointer = -1

            if model_to_evaluate == 'biencoder':
                parsed_line['candidates'] = sorted(parsed_line['candidates'], key=lambda x: x['candidate_score'],
                                                   reverse=True)
            elif model_to_evaluate == 'crossencoder':
                parsed_line['candidates'] = sorted(parsed_line['candidates'], key=lambda x: x['crossencoder_score'],
                                                   reverse=True)
            else:
                raise RuntimeError('model_to_evaluate type not recognized: ' + model_to_evaluate)
            for idx_rank, curr_candidate in enumerate(parsed_line['candidates']):
                if curr_candidate['candidate_doc_id'] == gold_doc_id:
                    pointer = idx_rank
                    break

            if parsed_line['category'] == config['category_new_entities']:
                if gold_doc_id in dict_filtered_to_cut_to_doc_ids[curr_timestamp_str]['filter-covid']:
                    stats[subset_name][curr_timestamp_str][model_to_evaluate]['filter-covid'].add(pointer)
                    stats_mean[subset_name][model_to_evaluate]['filter-covid'].add(pointer)
                    if gold_doc_id not in already_counted:
                        nr_covid_entities += 1
                        already_counted.add(gold_doc_id)
                elif gold_doc_id in dict_filtered_to_cut_to_doc_ids[curr_timestamp_str]['filter-recurrent']:
                    if not config['only_corona_vs_others']:
                        stats[subset_name][curr_timestamp_str][model_to_evaluate]['filter-recurrent'].add(pointer)
                        stats_mean[subset_name][model_to_evaluate]['filter-recurrent'].add(pointer)
                    else:
                        stats[subset_name][curr_timestamp_str][model_to_evaluate]['filter-other'].add(pointer)
                        stats_mean[subset_name][model_to_evaluate]['filter-other'].add(pointer)
                    if gold_doc_id not in already_counted:
                        already_counted.add(gold_doc_id)
                        nr_recurrent_entities += 1
                else:
                    stats[subset_name][curr_timestamp_str][model_to_evaluate]['filter-other'].add(pointer)
                    stats_mean[subset_name][model_to_evaluate]['filter-other'].add(pointer)
                    if gold_doc_id not in already_counted:
                        already_counted.add(gold_doc_id)
                        nr_other_entities += 1

            if parsed_line['category'] not in stats_mean_category[subset_name][model_to_evaluate]:
                stats_mean_category[subset_name][model_to_evaluate][parsed_line['category']] = Stats(top_k=top_k,
                                                                                                     rank=top_k_list)

            stats_mean_category[subset_name][model_to_evaluate][parsed_line['category']].add(pointer)
    logger.info('NR COVID NEW ENTITIES: %s  RECURRENT: %s  OTHER: %s' %
                (nr_covid_entities, nr_recurrent_entities, nr_other_entities))
    return stats, stats_mean


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

    logger.info('to_ret_str is now: \n%s' % to_ret_str)

    results_df = pd.DataFrame(lst_to_add_to_pandas)
    # and now plots it
    fig_plt, axs_plt = plt.subplots(1, 1)

    results_df.plot(ax=axs_plt, x='year')

    axs_plt.set_xlabel('Year')
    axs_plt.set_ylabel('Accuracy')
    axs_plt.set_title('{} first results'.format(model))

    logger.info('logging image')


def print_individual_table(model, category, results, subset):
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

    logger.info('to_ret_str is now: \n%s' % to_ret_str)

    results_df = pd.DataFrame(lst_to_add_to_pandas)
    # and now plots it
    fig_plt, axs_plt = plt.subplots(1, 1)

    results_df.plot(ax=axs_plt, x='year')

    axs_plt.set_xlabel('Year')
    axs_plt.set_ylabel('Accuracy')
    axs_plt.set_title('{} first results'.format(model))

    logger.info('logging image')


def print_general_table(results, config, category=None, subset_name=None, model_name=None):
    to_ret_str = ''

    # \begin{table}[t]
    # \centering
    # \resizebox{1.0\columnwidth}{!}
    # {\begin{tabular}{l cccccccccc}
    #         \toprule
    #         & 2013 & 2014 & 2015 & 2016 & 2017 & 2018 & 2019 & 2020  & 2021  & 2022 \\
    #         \midrule
    #                 2013 & 0.000 & \cellcolor{deepcarmine!10}0.000 & \cellcolor{deepcarmine!15}0.000 & \cellcolor{deepcarmine!20}0.000 & \cellcolor{deepcarmine!25}0.000 & \cellcolor{deepcarmine!30}0.000 & \cellcolor{deepcarmine!35}0.000 & \cellcolor{deepcarmine!40}0.000 & \cellcolor{deepcarmine!45}0.000 & \cellcolor{deepcarmine!70}0.000 \\
    #                 2014 & \cellcolor{deepcarmine!10}0.000 & 0.000 & \cellcolor{deepcarmine!10}0.000 & \cellcolor{deepcarmine!15}0.000 & \cellcolor{deepcarmine!20}0.000 & \cellcolor{deepcarmine!25}0.000 & \cellcolor{deepcarmine!30}0.000 & \cellcolor{deepcarmine!35}0.000 & \cellcolor{deepcarmine!40}0.000 & \cellcolor{deepcarmine!45}0.000 \\
    #                 2015 & \cellcolor{deepcarmine!15}0.000 & \cellcolor{deepcarmine!10}0.000 & 0.000 & \cellcolor{deepcarmine!10}0.000 & \cellcolor{deepcarmine!15}0.000 & \cellcolor{deepcarmine!20}0.000 & \cellcolor{deepcarmine!25}0.000 & \cellcolor{deepcarmine!30}0.000 & \cellcolor{deepcarmine!35}0.000 & \cellcolor{deepcarmine!40}0.000 \\
    #                 2016 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
    #                 2017 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
    #                 2018 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
    #                 2019 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
    #                 2020 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
    #                 2021 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
    #                 2022 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
    #         \bottomrule
    #         \end{tabular}}
    #         \caption{Absolute accuracy or delta.}
    #         \label{tab:dataset}
    # \end{table}
    #
    min_intensity = 10
    max_intensity = 75
    range_intensity = max_intensity - min_intensity

    cuts_to_show = config['cuts_to_show']

    fine_tune_cuts = config['fine_tune_cuts']
    top_k_list = config['top_k_list']
    max_top_k = max(top_k_list)
    # cuts_with_results --> in years
    cuts_with_results = set()
    for curr_fine_tune_cut in fine_tune_cuts:
        curr_timestamp = curr_fine_tune_cut['timestamp']
        cuts_with_results.add(curr_timestamp)

    max_acc_per_fine_tune_cut = dict()
    min_acc_per_fine_tune_cut = dict()
    central_accuracies_per_fine_tune_cut = dict()
    max_accuracy = 0.0
    min_accuracy = 100.0
    for curr_fine_tune_cut in cuts_to_show:
        max_acc_per_fine_tune_cut[curr_fine_tune_cut] = 0.0
        min_acc_per_fine_tune_cut[curr_fine_tune_cut] = 100.0
        for curr_predicted_cut in cuts_to_show:
            #
            if subset_name in results[curr_fine_tune_cut]:
                curr_res = results[curr_fine_tune_cut][subset_name][curr_predicted_cut][model_name][category]
                curr_res = curr_res.output_json()
                curr_acc = curr_res['acc@{}'.format(max_top_k)]
            else:
                curr_acc = 0.0
            max_acc_per_fine_tune_cut[curr_fine_tune_cut] = max(curr_acc, max_acc_per_fine_tune_cut[curr_fine_tune_cut])
            min_acc_per_fine_tune_cut[curr_fine_tune_cut] = min(curr_acc, min_acc_per_fine_tune_cut[curr_fine_tune_cut])
            min_accuracy = min(curr_acc, min_accuracy)
            max_accuracy = max(curr_acc, max_accuracy)
            if curr_fine_tune_cut == curr_predicted_cut:
                central_accuracies_per_fine_tune_cut[curr_fine_tune_cut] = curr_acc

    to_ret_str += '\\begin{table}[t] \n'
    to_ret_str += '\\centering \n'
    to_ret_str += '\\resizebox{1.0\columnwidth}{!} \n'
    to_ret_str += '{\\begin{tabular}{l ' + 'c' * len(cuts_to_show) + '} \n'
    to_ret_str += '\t \\toprule \n'
    to_ret_str += '\t'
    for curr_cut in cuts_to_show:
        parsed_curr_date = datetime.strptime(curr_cut, '%Y-%m-%dT%H:%M:%SZ')
        curr_year = parsed_curr_date.year
        to_ret_str += ' & {}'.format(curr_year)
    to_ret_str += ' \\\\ \n'
    to_ret_str += '\t\\midrule \n'
    for curr_fine_tune_cut in cuts_to_show:
        parsed_curr_date = datetime.strptime(curr_fine_tune_cut, '%Y-%m-%dT%H:%M:%SZ')
        curr_year = parsed_curr_date.year
        to_ret_str += '%s ' % curr_year
        if curr_fine_tune_cut in cuts_with_results:
            max_acc_pf = max_acc_per_fine_tune_cut[curr_fine_tune_cut]
            min_acc_pf = min_acc_per_fine_tune_cut[curr_fine_tune_cut]
            for curr_predicted_cut in cuts_to_show:
                curr_res = results[curr_fine_tune_cut][subset_name][curr_predicted_cut][model_name][category]
                curr_res = curr_res.output_json()
                if curr_fine_tune_cut == curr_predicted_cut:
                    to_ret_str += (' & %.3f' % curr_res['acc@{}'.format(max_top_k)])
                else:
                    curr_acc = curr_res['acc@{}'.format(max_top_k)]
                    max_delta = central_accuracies_per_fine_tune_cut[curr_fine_tune_cut] - min_acc_pf
                    max_delta = max(max_delta, max_acc_pf - central_accuracies_per_fine_tune_cut[curr_fine_tune_cut])

                    if curr_acc < central_accuracies_per_fine_tune_cut[curr_fine_tune_cut]:
                        curr_delta = central_accuracies_per_fine_tune_cut[curr_fine_tune_cut] - curr_acc
                        intensity_name = 'deepcarmine'
                    else:
                        curr_delta = curr_acc - central_accuracies_per_fine_tune_cut[curr_fine_tune_cut]
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
    to_ret_str += ('\t\\caption{Accuracy@%s for category %s} \n' % (max_top_k, category))
    to_ret_str += ('\t\\label{tab:general_table_%s}\n' % category)
    to_ret_str += '\\end{table}\n'
    logger.info('generic table: category %s subset_name %s model_name %s ' % (category, subset_name, model_name))
    print(to_ret_str)


def print_result_tables(stats, stats_mean, stats_mean_category, config):
    """

    :param results:
    :param config:
    :return:
    """

    output_csv_path = config['output_csv_path']
    # create the dir if doesn't exist
    output_csv_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_csv_dir, exist_ok=True)
    with open(output_csv_path, 'wt') as outfile:
        writer = csv.writer(outfile)
        if not config['only_corona_vs_others']:
            writer.writerow(['snapshot', 'k', 'filter-covid', 'filter-recurrent', 'filter-other'])
        else:
            writer.writerow(['snapshot', 'k', 'filter-covid', 'filter-other'])

        for curr_cut in config['cuts_to_show']:
            parsed_curr_date = datetime.strptime(curr_cut, '%Y-%m-%dT%H:%M:%SZ')
            curr_year = parsed_curr_date.year

            logger.info('--------STATS------')
            logger.info(
                '%s filter-covid cnt: %s' % (curr_cut, stats['test'][curr_cut]['biencoder']['filter-covid'].cnt))
            logger.info('%s filter-recurrent cnt: %s' % (
                curr_cut, stats['test'][curr_cut]['biencoder']['filter-recurrent'].cnt))
            logger.info('%s filter-other cnt: %s' % (curr_cut,
                                                     stats['test'][curr_cut]['biencoder']['filter-other'].cnt))
            logger.info('%s filter-covid result: %s' % (curr_cut,
                                                        stats['test'][curr_cut]['biencoder'][
                                                            'filter-covid'].output_json()))
            logger.info('%s filter-recurrent result: %s' % (curr_cut,
                                                            stats['test'][curr_cut]['biencoder'][
                                                                'filter-recurrent'].output_json()))
            logger.info('%s filter-other result: %s' % (curr_cut,
                                                        stats['test'][curr_cut]['biencoder'][
                                                            'filter-other'].output_json()))
            f_other_st = stats['test'][curr_cut]['biencoder']['filter-other']
            f_recurrent_st = stats['test'][curr_cut]['biencoder']['filter-recurrent']
            f_covid_st = stats['test'][curr_cut]['biencoder']['filter-covid']
            for i in range(f_other_st.LEN):
                if f_other_st.top_k < f_other_st.rank[i]:
                    break
                curr_k = f_other_st.rank[i]
                curr_acc_other = 0.0
                curr_acc_recurrent = 0.0
                curr_acc_covid = 0.0
                if f_other_st.cnt > 0:
                    curr_acc_other = f_other_st.hits[i] / float(f_other_st.cnt)

                if f_recurrent_st.cnt > 0:
                    curr_acc_recurrent = f_recurrent_st.hits[i] / float(f_recurrent_st.cnt)

                if f_covid_st.cnt > 0:
                    curr_acc_covid = f_covid_st.hits[i] / float(f_covid_st.cnt)
                if not config['only_corona_vs_others']:
                    writer.writerow([curr_year, curr_k, curr_acc_covid, curr_acc_recurrent, curr_acc_other])
                else:
                    writer.writerow([curr_year, curr_k, curr_acc_covid, curr_acc_other])

        logger.info('---------------STATS MEAN---------------')
        logger.info('MEAN filter-covid cnt: %s' % stats_mean['test']['biencoder']['filter-covid'].cnt)
        logger.info('MEAN filter-recurrent cnt: %s' % stats_mean['test']['biencoder']['filter-recurrent'].cnt)
        logger.info('MEAN filter-other cnt: %s' % stats_mean['test']['biencoder']['filter-other'].cnt)
        logger.info('MEAN filter-covid result: %s' % stats_mean['test']['biencoder']['filter-covid'].output_json())
        logger.info(
            'MEAN filter-recurrent result: %s' % stats_mean['test']['biencoder']['filter-recurrent'].output_json())
        logger.info('MEAN filter-other result: %s' % stats_mean['test']['biencoder']['filter-other'].output_json())

        logger.info('---------------STATS MEAN CATEGORY---------------')
        if 'continual' in stats_mean_category['test']['biencoder']:
            logger.info('MEAN random cnt: %s' % stats_mean_category['test']['biencoder']['continual'].cnt)
            logger.info('MEAN new_entities cnt: %s' % stats_mean_category['test']['biencoder']['new'].cnt)
            logger.info('MEAN random result: %s' % stats_mean_category['test']['biencoder']['continual'].output_json())
            logger.info('MEAN new_entities result: %s' % stats_mean_category['test']['biencoder']['new'].output_json())
        else:
            logger.info('MEAN random cnt: %s' % stats_mean_category['test']['biencoder']['shared'].cnt)
            logger.info('MEAN new_entities cnt: %s' % stats_mean_category['test']['biencoder']['new_entities'].cnt)
            logger.info('MEAN random result: %s' % stats_mean_category['test']['biencoder']['shared'].output_json())
            logger.info(
                'MEAN new_entities result: %s' % stats_mean_category['test']['biencoder']['new_entities'].output_json())


def get_target_filter_type(curr_test_year, target_orig_title, config, text_entity_tokenized):
    type = 'filter-other'
    # other types: filter-covid, filter-recurrent

    curr_test_year = int(curr_test_year)
    year_str = str(curr_test_year - 1)

    if target_orig_title.startswith('20') or (year_str in target_orig_title) or \
            (str(curr_test_year) in target_orig_title) or (str(curr_test_year + 1) in target_orig_title):
        curr_test_year_range = '{}–{}'.format(year_str[-2:], str(curr_test_year)[-2:])
        should_be_in = ['league', 'liga', 'cup', 'election', 'tournament', 'liiga', 'tour',
                        'open', 'championship', 'prix', 'copa', 'season', 'olympics',
                        'team', 'football', curr_test_year_range, '_games', 'uefa', 'euro_', 'qualifying',
                        'campeonato', 'finals', '_nba_', 'basketball', '_final', '_olympic',
                        'segunda_di']

        for curr_sh_in in should_be_in:
            if curr_sh_in in target_orig_title.lower():
                type = 'filter-recurrent'

    for curr_keyword in ['covid', 'covid-19', 'coronavirus']:
        if curr_keyword in target_orig_title:
            type = 'filter-covid'
        if config['look_in_entity_text']:
            if curr_keyword in text_entity_tokenized:
                type = 'filter-covid'

    return type


if __name__ == '__main__':
    # import pdb
    #
    # pdb.set_trace()

    parser = argparse.ArgumentParser(description='Evaluate biencoder\'s output')
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/stats/paper_stats/year_analyze_20220630/config/s07_analyze_year.json',
                        help='The config file that contains all the parameters')
    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))

    gold_context_mention_to_label = dict()
    ground_truth_path = dict()
    predictions_path = dict()

    subsets = config['subsets']

    input_dir_dataset = config['input_dir_dataset']
    stats = dict()
    # mean across years
    stats_mean = dict()
    stats_mean_category = dict()
    top_k_list = config['top_k_list']
    top_k = max(top_k_list)
    list_cuts = list()
    dict_finetune_cuts = dict()

    context_length = config['context_length']
    logger.info('loading ground truth...')
    start_loading_gt = time.time()
    # filtered according to "filter" attribute in config file
    dict_filtered_to_cut_to_doc_ids = dict()

    # loads the ground truth in subsets, TODO this can be cached
    for curr_subset in subsets:
        subset_name = curr_subset['subset_name']
        ground_truth_path[subset_name] = dict()

        for curr_subset_time_cut in curr_subset['subset_cuts']:
            curr_timestamp_str = curr_subset_time_cut['timestamp']
            year = datetime.strptime(curr_timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
            curr_year = year.year
            curr_input_file_name = curr_subset_time_cut['input_file_name']
            ground_truth_path[subset_name][curr_timestamp_str] = os.path.join(input_dir_dataset,
                                                                              curr_input_file_name)
            if curr_timestamp_str not in dict_filtered_to_cut_to_doc_ids:
                dict_filtered_to_cut_to_doc_ids[curr_timestamp_str] = dict()
                dict_filtered_to_cut_to_doc_ids[curr_timestamp_str]['filter-other'] = set()
                dict_filtered_to_cut_to_doc_ids[curr_timestamp_str]['filter-covid'] = set()
                dict_filtered_to_cut_to_doc_ids[curr_timestamp_str]['filter-recurrent'] = set()

            logger.info('loading into gold_context_mention_to_label from: %s' %
                        ground_truth_path[subset_name][curr_timestamp_str])
            with open(ground_truth_path[subset_name][curr_timestamp_str], 'rt') as infile:
                logger.info('reading %s' % ground_truth_path[subset_name][curr_timestamp_str])
                for idx_mention_ground_truth, curr_line in tqdm(enumerate(infile)):
                    parsed_line = json.loads(curr_line)
                    if parsed_line['category'] != config['category_new_entities']:
                        continue
                    bert_entity_tokenized = parsed_line['target_bert_tokenized'][:context_length]
                    target_orig_title = parsed_line['target_orig_title'].lower()
                    text_entity_tokenized = set(from_bert_to_text(bert_entity_tokenized).strip().lower().split(' '))
                    men_context_length = int(context_length / 2)
                    parsed_line['mention'] = from_bert_to_text(parsed_line['mention_bert_tokenized'])
                    bert_ctxt_men_tokenized = parsed_line['context_left_bert_tokenized'][-men_context_length:] + \
                                              [parsed_line['mention']] + \
                                              parsed_line['context_right_bert_tokenized'][:men_context_length]
                    text_ctxt_men_tokenized = set(from_bert_to_text(bert_ctxt_men_tokenized).strip().split(' '))

                    filtered_entity = False
                    filtered_mention = False
                    filtered_entity_title = False

                    type = get_target_filter_type(curr_test_year=curr_year, target_orig_title=target_orig_title,
                                                  config=config, text_entity_tokenized=text_entity_tokenized)

                    logger.debug('%s +++ %s ===== %s' % (curr_year, target_orig_title, type))
                    dict_filtered_to_cut_to_doc_ids[curr_timestamp_str][type].add(parsed_line['target_wikidata_qid'])

    end_loading_gt = time.time()

    logger.info('%s mins to load ground truth...' % ((end_loading_gt - start_loading_gt) / 60))

    # if filter is activated, then gets some statistics from the training set of each of the cuts
    #  first , this can be serialized later on I guess
    dict_cut_to_target_qid_to_nr_mentions = dict()
    dict_cut_to_top_mentions_target_qids = dict()

    # calculates the actual results
    for curr_model in config['models']:
        for curr_fine_tune_cut in config['fine_tune_cuts']:
            input_dir_predictions = curr_fine_tune_cut['input_dir_predictions']
            logger.info('EVALUATING predictions in %s' % input_dir_predictions)

            for curr_subset in subsets:
                subset_name = curr_subset['subset_name']

                predictions_path[subset_name] = dict()
                stats[subset_name] = dict()
                stats_mean[subset_name] = dict()
                stats_mean_category[subset_name] = dict()

                for curr_subset_time_cut in curr_subset['subset_cuts']:
                    # IT IS O(N), but it is a very short list
                    if curr_subset_time_cut['timestamp'] not in list_cuts:
                        list_cuts.append(curr_subset_time_cut['timestamp'])
                    curr_timestamp_str = curr_subset_time_cut['timestamp']
                    curr_pred_file_name = curr_subset_time_cut['pred_file_name']
                    # --
                    predictions_path[subset_name][curr_timestamp_str] = os.path.join(input_dir_predictions,
                                                                                     curr_pred_file_name)
                    # --
                    # both finetune cut and subset cut has to be accounted for
                    stats[subset_name][curr_timestamp_str] = dict()
                    stats[subset_name][curr_timestamp_str][curr_model] = dict()

                    if curr_model not in stats_mean[subset_name]:
                        stats_mean[subset_name][curr_model] = dict()
                        stats_mean_category[subset_name][curr_model] = dict()

                    for curr_filter_to_evaluate in ['filter-covid', 'filter-recurrent', 'filter-other']:
                        stats[subset_name][curr_timestamp_str][curr_model] \
                            [curr_filter_to_evaluate] = Stats(top_k=top_k, rank=top_k_list)
                        if curr_filter_to_evaluate not in stats_mean[subset_name][curr_model]:
                            stats_mean[subset_name][curr_model][curr_filter_to_evaluate] = Stats(top_k=top_k,
                                                                                                 rank=top_k_list)

                    wikidata_qid_to_include = None
                    if curr_timestamp_str in dict_cut_to_top_mentions_target_qids:
                        wikidata_qid_to_include = dict_cut_to_top_mentions_target_qids[curr_timestamp_str]

                    evaluate_predictions(predictions_path, config,
                                         stats, subset_name,
                                         curr_timestamp_str,
                                         model_to_evaluate=curr_model,
                                         wikidata_qid_to_include=wikidata_qid_to_include,
                                         dict_filtered_to_cut_to_doc_ids=dict_filtered_to_cut_to_doc_ids,
                                         stats_mean=stats_mean,
                                         stats_mean_category=stats_mean_category,
                                         top_k=top_k,
                                         top_k_list=top_k_list)

    print_result_tables(stats, stats_mean, stats_mean_category, config)
