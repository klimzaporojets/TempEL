# statistics of the final dataset
import json
import logging
import math
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from tempel_creation.misc import from_bert_to_text
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def get_subset_distribution_v2(subset_tuples):
    """
    This is the ones that accepts already extracted tuples (by process_dataset_tuple).
    :param subset_path:
    :param bucket_config:
        [{"bucket_name": "bucket1", "min_mentions_per_entity": 100, "max_mentions_per_entity": 1000}],
    :return:
    """

    nr_mentions = 0
    nr_mentions_new = 0
    nr_mentions_shared = 0
    target_wikidata_qids = set()
    target_wikidata_qids_new = set()
    target_wikidata_qids_shared = set()
    distinct_mentions = set()
    entity_to_nr_mentions = dict()
    for curr_tuple in subset_tuples:
        nr_mentions += 1
        target_wikidata_qid = curr_tuple['entity_qid']
        curr_mention = curr_tuple['mention']
        target_wikidata_qids.add(target_wikidata_qid)
        if curr_tuple['category'] in {'continual', 'shared'}:
            target_wikidata_qids_shared.add(target_wikidata_qid)
            nr_mentions_shared += 1
        elif curr_tuple['category'] in {'new', 'new_entities'}:
            target_wikidata_qids_new.add(target_wikidata_qid)
            nr_mentions_new += 1
        else:
            raise RuntimeError('parsed_line[\'category\'] not recognized: ' + curr_tuple['category'])
        distinct_mentions.add(curr_mention)
        if target_wikidata_qid not in entity_to_nr_mentions:
            entity_to_nr_mentions[target_wikidata_qid] = 0

        entity_to_nr_mentions[target_wikidata_qid] += 1

    nr_entities = len(target_wikidata_qids)
    nr_distinct_mentions = len(distinct_mentions)
    nr_entities_new = len(target_wikidata_qids_new)
    nr_entities_shared = len(target_wikidata_qids_shared)

    return {'nr_mentions': nr_mentions, 'nr_distinct_mentions': nr_distinct_mentions, 'nr_entities': nr_entities,
            'nr_entities_new': nr_entities_new, 'nr_entities_shared': nr_entities_shared,
            'nr_mentions_new': nr_mentions_new, 'nr_mentions_shared': nr_mentions_shared}


def get_subset_distribution(subset_path):
    """

    :param subset_path:
    :param bucket_config:
        [{"bucket_name": "bucket1", "min_mentions_per_entity": 100, "max_mentions_per_entity": 1000}],
    :return:
    """

    bucket_config = [{'bucket_name': 'bucket1', 'min_mentions_per_entity': 100, 'max_mentions_per_entity': 1000},
                     {'bucket_name': 'bucket2', 'min_mentions_per_entity': 50, 'max_mentions_per_entity': 99},
                     {'bucket_name': 'bucket3', 'min_mentions_per_entity': 10, 'max_mentions_per_entity': 49},
                     {'bucket_name': 'bucket4', 'min_mentions_per_entity': 1, 'max_mentions_per_entity': 9},
                     ]

    bucket_config_nr_entities_new = {'bucket1': 0,
                                     'bucket2': 0,
                                     'bucket3': 0,
                                     'bucket4': 0
                                     }
    bucket_config_nr_entities_shared = {'bucket1': 0,
                                        'bucket2': 0,
                                        'bucket3': 0,
                                        'bucket4': 0
                                        }

    nr_mentions = 0
    nr_mentions_new = 0
    nr_mentions_shared = 0

    target_wikidata_qids = set()
    target_wikidata_qids_new = set()
    target_wikidata_qids_shared = set()
    distinct_mentions = set()
    entity_to_nr_mentions = dict()
    with open(subset_path) as infile:
        for curr_line in infile:
            parsed_line = json.loads(curr_line)
            nr_mentions += 1
            target_wikidata_qid = parsed_line['target_wikidata_qid']
            curr_mention = parsed_line['mention']
            target_wikidata_qids.add(target_wikidata_qid)
            if parsed_line['category'] in {'continual', 'shared'}:
                target_wikidata_qids_shared.add(target_wikidata_qid)
                nr_mentions_shared += 1
            elif parsed_line['category'] in {'new', 'new_entities'}:
                target_wikidata_qids_new.add(target_wikidata_qid)
                nr_mentions_new += 1
            else:
                raise RuntimeError('parsed_line[\'category\'] not recognized: ' + parsed_line['category'])
            distinct_mentions.add(curr_mention)
            if target_wikidata_qid not in entity_to_nr_mentions:
                entity_to_nr_mentions[target_wikidata_qid] = 0

            entity_to_nr_mentions[target_wikidata_qid] += 1

    nr_entities = len(target_wikidata_qids)
    nr_distinct_mentions = len(distinct_mentions)
    nr_entities_new = len(target_wikidata_qids_new)
    nr_entities_shared = len(target_wikidata_qids_shared)

    for curr_entity, curr_nr_mentions in entity_to_nr_mentions.items():
        for idx_bucket, curr_bucket_config in enumerate(bucket_config):
            if curr_bucket_config['min_mentions_per_entity'] <= curr_nr_mentions \
                    <= curr_bucket_config['max_mentions_per_entity']:
                if curr_entity in target_wikidata_qids_new:
                    bucket_config_nr_entities_new[curr_bucket_config['bucket_name']] += 1
                else:
                    bucket_config_nr_entities_shared[curr_bucket_config['bucket_name']] += 1
                break
    logger.info('bucket_config_nr_entities_new: %s' % bucket_config_nr_entities_new)
    logger.info('bucket_config_nr_entities_shared: %s' % bucket_config_nr_entities_shared)
    return {'nr_mentions': nr_mentions, 'nr_distinct_mentions': nr_distinct_mentions, 'nr_entities': nr_entities,
            'nr_entities_new': nr_entities_new, 'nr_entities_shared': nr_entities_shared,
            'nr_mentions_new': nr_mentions_new, 'nr_mentions_shared': nr_mentions_shared}


def process_dataset_tuple(curr_tuple, subset_name):
    parsed_tuple = json.loads(curr_tuple)
    fld_mention = parsed_tuple['mention_bert_tokenized']
    fld_mention = from_bert_to_text(fld_mention)
    fld_subset = parsed_tuple['subset']
    fld_entity_qid = parsed_tuple['target_wikidata_qid']
    fld_cut = parsed_tuple['filtered_date']
    parsed_curr_date = datetime.strptime(fld_cut, '%Y-%m-%dT%H:%M:%SZ')
    fld_cut = parsed_curr_date.year

    fld_target_len = parsed_tuple['target_len']
    fld_anchor_len = parsed_tuple['anchor_len']

    fld_category = parsed_tuple['category']
    fld_target_orig_title = parsed_tuple['target_orig_title']
    # just the first 256 bert tokens
    fld_target_bert_tokenized = parsed_tuple['target_bert_tokenized'][:256]
    assert fld_subset == subset_name

    return {'mention': fld_mention, 'subset': fld_subset, 'entity_qid': fld_entity_qid, 'cut': fld_cut,
            'category': fld_category, 'target_len': fld_target_len, 'anchor_len': fld_anchor_len,
            'target_orig_title': fld_target_orig_title, 'target_bert_tokenized': fld_target_bert_tokenized}


def get_tuples_per_year(config):
    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)
    dataset_path = config['dataset_path']
    cache_tuples_path = config['cache_tuples_path']
    if not os.path.exists(cache_tuples_path):
        dirname = os.path.dirname(cache_tuples_path)
        os.makedirs(dirname, exist_ok=True)
        train_tuples_per_year = dict()
        validation_tuples_per_year = dict()
        test_tuples_per_year = dict()
        for curr_year in sorted_cuts:
            train_tuples_per_year[curr_year] = list()
            validation_tuples_per_year[curr_year] = list()
            test_tuples_per_year[curr_year] = list()
            year_config = config['time_cuts'][curr_year]
            train_path = os.path.join(dataset_path, year_config['train_path'])
            validation_path = os.path.join(dataset_path, year_config['validation_path'])
            test_path = os.path.join(dataset_path, year_config['test_path'])
            logger.info('stat_mentions_per_entity loading %s' % train_path)
            for curr_tuple in tqdm(open(train_path)):
                curr_tuple = process_dataset_tuple(curr_tuple, 'train')
                train_tuples_per_year[curr_year].append(curr_tuple)

            logger.info('stat_mentions_per_entity loading %s' % validation_path)
            for curr_tuple in tqdm(open(validation_path)):
                curr_tuple = process_dataset_tuple(curr_tuple, 'validation')
                validation_tuples_per_year[curr_year].append(curr_tuple)

            logger.info('stat_mentions_per_entity loading %s' % test_path)
            for curr_tuple in tqdm(open(test_path)):
                curr_tuple = process_dataset_tuple(curr_tuple, 'test')
                test_tuples_per_year[curr_year].append(curr_tuple)
        pickle.dump({
            'train_tuples_per_year': train_tuples_per_year,
            'validation_tuples_per_year': validation_tuples_per_year,
            'test_tuples_per_year': test_tuples_per_year
        }
            , open(cache_tuples_path, 'wb'))
    else:
        saved_tuples = pickle.load(open(cache_tuples_path, 'rb'))
        train_tuples_per_year = saved_tuples['train_tuples_per_year']
        validation_tuples_per_year = saved_tuples['validation_tuples_per_year']
        test_tuples_per_year = saved_tuples['test_tuples_per_year']
    return train_tuples_per_year, validation_tuples_per_year, test_tuples_per_year


def stat_target_anchor_lengths(config, train_tuples_per_year, validation_tuples_per_year, test_tuples_per_year):
    # subset, entity, mention, cut
    lst_entity_mentions_year = []

    for curr_year, tuples_year in train_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year
    for curr_year, tuples_year in validation_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year
    for curr_year, tuples_year in test_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year

    df_target_anchor_lengths = pd.DataFrame(lst_entity_mentions_year)

    df_target_anchor_lengths: pd.DataFrame = \
        df_target_anchor_lengths[['category', 'target_len', 'anchor_len', 'subset']]

    df_target_anchor_lengths = df_target_anchor_lengths[df_target_anchor_lengths['subset'] == 'test']
    df_target_anchor_lengths: pd.DataFrame = df_target_anchor_lengths[['category', 'target_len', 'anchor_len']] \
        .groupby(['category']).mean().reset_index()

    logger.info('content of df_target_anchor_lengths: %s' % df_target_anchor_lengths.to_string())


def stat_mentions_per_entity(config, train_tuples_per_year, validation_tuples_per_year, test_tuples_per_year):
    output_plot_path = config['output_plot_path']

    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)
    # subset, entity, mention, cut
    lst_entity_mentions_year = []

    for curr_year, tuples_year in train_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year
    for curr_year, tuples_year in validation_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year
    for curr_year, tuples_year in test_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year

    df_entity_mentions_stats = pd.DataFrame(lst_entity_mentions_year)
    df_entity_mentions_stats['cut'] = pd.to_datetime(df_entity_mentions_stats['cut'], format='%Y-%m-%dT%H:%M:%SZ')

    ##### BEGIN: CUMSUM PER SUBSET ACROSS CUTS
    df_nr_mentions_per_entity: pd.DataFrame = df_entity_mentions_stats[['mention', 'subset', 'entity_qid', 'cut']] \
        .groupby(['subset', 'entity_qid']).count().reset_index()
    df_nr_mentions_per_entity.rename(columns={'mention': 'nr_mentions'}, inplace=True)

    df_entities_per_nr_mentions = df_nr_mentions_per_entity[['subset', 'entity_qid', 'nr_mentions']] \
        .groupby(['subset', 'nr_mentions']).count().reset_index()
    df_entities_per_nr_mentions.rename(columns={'entity_qid': 'nr_entities_per_nr_mentions'}, inplace=True)

    # averages across cuts
    df_entities_per_nr_mentions['nr_entities_per_nr_mentions'] /= len(sorted_cuts)
    df_entities_per_nr_mentions['nr_entities_per_nr_mentions'] = \
        df_entities_per_nr_mentions['nr_entities_per_nr_mentions'].apply(np.ceil)

    df_entities_per_nr_mentions.nr_entities_per_nr_mentions = \
        df_entities_per_nr_mentions.nr_entities_per_nr_mentions.astype(int)
    df_entities_per_nr_mentions = df_entities_per_nr_mentions.sort_values(['subset', 'nr_mentions'],
                                                                          ascending=[True, True])
    df_entities_per_nr_mentions['cumsum'] = df_entities_per_nr_mentions.groupby(['subset']) \
        ['nr_entities_per_nr_mentions'].transform(pd.Series.cumsum)

    df_gr_max_cumsum = df_entities_per_nr_mentions[['subset', 'cumsum']].groupby(['subset']).max().reset_index()
    df_gr_max_cumsum.rename(columns={'cumsum': 'max_cumsum'}, inplace=True)
    df_entities_per_nr_mentions = pd.merge(df_gr_max_cumsum, df_entities_per_nr_mentions, on='subset', how='inner')
    df_entities_per_nr_mentions['frac_max_cumsum'] = df_entities_per_nr_mentions['cumsum'] / \
                                                     df_entities_per_nr_mentions['max_cumsum']

    df_to_plot = df_entities_per_nr_mentions[['subset', 'nr_mentions', 'cumsum']]
    fig_plt, axs_plt = plt.subplots(1, 1, figsize=(10, 10))

    df_pivotet = df_to_plot.pivot_table('cumsum', ['nr_mentions'], 'subset').reset_index()
    df_pivotet.plot(ax=axs_plt, x='nr_mentions')

    min_nr_mentions = 1
    max_nr_mentions = df_to_plot[['nr_mentions']].max().item()

    interval_size = 1
    x_ticks = list(range(min_nr_mentions, max_nr_mentions, interval_size))

    axs_plt.set_ylabel('Nr of linked entities')
    axs_plt.set_xlabel('Nr Mentions')
    axs_plt.set_title('Temporal link stats (all links)')
    curr_plot_path = os.path.join(output_plot_path, 'train_mentions_per_entity.pdf')
    fig_plt.savefig(curr_plot_path, format='pdf')
    plt.xticks(x_ticks, x_ticks)
    plt.show()


def stat_mentions_per_entity_across_cuts(config, train_tuples_per_year, validation_tuples_per_year,
                                         test_tuples_per_year):
    output_plot_path = config['output_plot_path']

    lst_entity_mentions_year = []

    for curr_year, tuples_year in train_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year
    for curr_year, tuples_year in validation_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year
    for curr_year, tuples_year in test_tuples_per_year.items():
        lst_entity_mentions_year += tuples_year

    df_entity_mentions_stats = pd.DataFrame(lst_entity_mentions_year)

    ##### BEGIN: CUMSUM PER SUBSET ACROSS CUTS
    df_nr_mentions_per_entity: pd.DataFrame = \
        df_entity_mentions_stats[['mention', 'subset', 'entity_qid', 'cut', 'category']] \
            .groupby(['subset', 'entity_qid', 'cut', 'category']).count().reset_index()
    df_nr_mentions_per_entity.rename(columns={'mention': 'nr_mentions'}, inplace=True)

    df_entities_per_nr_mentions = df_nr_mentions_per_entity[['subset', 'entity_qid', 'nr_mentions', 'cut', 'category']] \
        .groupby(['subset', 'nr_mentions', 'cut', 'category']).count().reset_index()
    df_entities_per_nr_mentions.rename(columns={'entity_qid': 'nr_entities_per_nr_mentions'}, inplace=True)

    df_entities_per_nr_mentions.nr_entities_per_nr_mentions = \
        df_entities_per_nr_mentions.nr_entities_per_nr_mentions.astype(int)
    df_entities_per_nr_mentions = df_entities_per_nr_mentions.sort_values(['subset', 'cut', 'category', 'nr_mentions'],
                                                                          ascending=[True, True, True, True])
    df_entities_per_nr_mentions['cumsum'] = df_entities_per_nr_mentions.groupby(['subset', 'cut', 'category']) \
        ['nr_entities_per_nr_mentions'].transform(pd.Series.cumsum)

    df_gr_max_cumsum = df_entities_per_nr_mentions[['subset', 'cut', 'category', 'cumsum']] \
        .groupby(['subset', 'cut', 'category']).max().reset_index()
    df_gr_max_cumsum.rename(columns={'cumsum': 'max_cumsum'}, inplace=True)
    df_entities_per_nr_mentions = pd.merge(df_gr_max_cumsum, df_entities_per_nr_mentions,
                                           on=['subset', 'cut', 'category'],
                                           how='inner')
    df_entities_per_nr_mentions['frac_max_cumsum'] = df_entities_per_nr_mentions['cumsum'] / \
                                                     df_entities_per_nr_mentions['max_cumsum']

    df_to_plot = df_entities_per_nr_mentions[['subset', 'nr_mentions', 'cumsum', 'cut', 'category']]

    dist_subsets = df_to_plot.subset.unique()
    dist_categories = df_to_plot.category.unique()
    output_men_per_ent_cuts_csv = config['output_men_per_ent_cuts_csv']
    os.makedirs(os.path.dirname(output_men_per_ent_cuts_csv), exist_ok=True)
    df_to_plot.to_csv(output_men_per_ent_cuts_csv, sep='\t', index=False)

    fig_plt, axs_plts = plt.subplots(2, 3, figsize=(15, 8))

    for idx_category, curr_category in enumerate(dist_categories):
        for idx_subset, curr_subset in enumerate(dist_subsets):
            df_to_plot_curr = df_to_plot.loc[((df_to_plot['category'] == curr_category) &
                                              (df_to_plot['subset'] == curr_subset)
                                              ), ['cumsum', 'nr_mentions', 'cut']]
            df_pivotet = df_to_plot_curr.pivot_table('cumsum', ['nr_mentions'], 'cut').reset_index()

            df_pivotet.fillna(method='ffill', inplace=True)

            df_pivotet.plot(ax=axs_plts[idx_category][idx_subset], x='nr_mentions')

            min_nr_mentions = 1
            max_nr_mentions = df_to_plot_curr[['nr_mentions']].max().item()
            interval_size = math.ceil(max_nr_mentions / 10)

            x_ticks = list(range(min_nr_mentions, max_nr_mentions, interval_size))

            axs_plts[idx_category][idx_subset].set_ylabel('Nr of linked entities')
            axs_plts[idx_category][idx_subset].set_xlabel('Nr Mentions')
            axs_plts[idx_category][idx_subset].set_title('Temporal link stats (all links)')

            axs_plts[idx_category][idx_subset].set_xticks(x_ticks)

            axs_plts[idx_category][idx_subset].yaxis.set_major_locator(MaxNLocator(integer=True))
            axs_plts[idx_category][idx_subset].xaxis.set_major_locator(MaxNLocator(integer=True))
            curr_title = 'Distribution ' + curr_category + '-' + curr_subset
            axs_plts[idx_category][idx_subset].set_title(curr_title)
            axs_plts[idx_category][idx_subset].get_legend().remove()

    handles, labels = axs_plts[1][2].get_legend_handles_labels()
    fig_plt.legend(handles, labels, loc='upper right')

    curr_plot_path = os.path.join(output_plot_path, 'train_mentions_per_entity.pdf')
    fig_plt.tight_layout()
    fig_plt.subplots_adjust(wspace=0.3)
    fig_plt.subplots_adjust(hspace=0.3)
    plt.show()
    fig_plt.savefig(curr_plot_path, format='pdf')


def get_subsets_distribution(train_path, validation_path, test_path, config):
    logger.info('processing %s' % train_path)
    train_distribution = get_subset_distribution(train_path)
    print('processing %s' % validation_path)
    validation_distribution = get_subset_distribution(validation_path)
    print('processing %s' % test_path)
    test_distribution = get_subset_distribution(test_path)

    # across the subsets the nr of entities has to be the same
    if not config['debug']:
        assert train_distribution['nr_entities'] == validation_distribution['nr_entities']
        assert validation_distribution['nr_entities'] == test_distribution['nr_entities']

    return {'train_distribution': train_distribution, 'validation_distribution': validation_distribution,
            'test_distribution': test_distribution}


def get_subsets_distribution_v2(train_tuples, validation_tuples, test_tuples, config):
    """
    The one working with tuples only
    :param train_path:
    :param validation_path:
    :param test_path:
    :param config:
    :return:
    """
    logger.info('processing train_tuples')
    train_distribution = get_subset_distribution_v2(train_tuples)
    logger.info('processing validation_tuples')
    validation_distribution = get_subset_distribution_v2(validation_tuples)
    logger.info('processing test_tuples')
    test_distribution = get_subset_distribution_v2(test_tuples)

    if train_distribution['nr_entities'] != validation_distribution['nr_entities']:
        logger.info('distinct number of entities between train and validation: ', train_distribution['nr_entities'],
                    ' vs ', validation_distribution['nr_entities'])
    if validation_distribution['nr_entities'] != test_distribution['nr_entities']:
        logger.info('distinct number of entities between validation and test: ', validation_distribution['nr_entities'],
                    ' vs ', test_distribution['nr_entities'])
    #

    return {'train_distribution': train_distribution, 'validation_distribution': validation_distribution,
            'test_distribution': test_distribution}


def print_some_examples_in_md(config):
    """
    Produces some examples of the dataset.
    :param config:
    :return:
    """
    base_input_path = config['base_input_path']
    output_examples_path = config['output_examples_path']
    dir_examples = os.path.dirname(output_examples_path)
    os.makedirs(dir_examples, exist_ok=True)
    max_examples_per_cut = 50

    with open(output_examples_path, 'wt') as outfile:
        for curr_cut, curr_cut_subsets in config['time_cuts'].items():
            curr_cut_train_path = os.path.join(base_input_path, curr_cut_subsets['train_path'])
            input_file = os.path.basename(curr_cut_train_path)
            nr_printed_examples = 0
            outfile.write('__{}__\n\n'.format(input_file))
            for curr_data_point in open(curr_cut_train_path, 'rt'):
                if nr_printed_examples >= max_examples_per_cut:
                    break
                parsed_line = json.loads(curr_data_point)
                left_context = from_bert_to_text(parsed_line['context_left_bert_tokenized'])
                right_context = from_bert_to_text(parsed_line['context_right_bert_tokenized'])
                target_text = from_bert_to_text(parsed_line['target_bert_tokenized'])
                target_orig_title = parsed_line['target_orig_title']
                mention = parsed_line['mention']

                outfile.write('__{}__ \n\n'.format(nr_printed_examples + 1))
                outfile.write('__*Anchor context*__: \n\n ' + left_context[-200:] + ' __' + mention + '__ ' +
                              right_context[:200] + '\n')
                outfile.write('\n')
                outfile.write('__*Target entity text*__: \n\n __' + target_orig_title.replace('_', ' ') +
                              '__' + target_text[:400] + '\n')
                outfile.write('\n')
                outfile.write('__Target entity triples__: TODO' + '\n')
                outfile.write('\n')
                del parsed_line['context_left_bert_tokenized']
                del parsed_line['context_right_bert_tokenized']
                del parsed_line['target_bert_tokenized']
                del parsed_line['target_orig_title']
                del parsed_line['target_title_bert_tokenized']
                del parsed_line['mention']
                del parsed_line['mention_bert_tokenized']
                del parsed_line['target_title_2022']
                outfile.write('__*Attributes*__: \n\n' + str(parsed_line) + '\n')
                outfile.write('\n')
                outfile.write('* * * \n')
                nr_printed_examples += 1


def print_latex_subset_distribution_v2(config):
    """
    Just another table distribution which seems better given that currently the number of entities is shared
    between the subsets, and there are also shared entities across cuts (random), to not generate too much
    repetition in the table.
    :param config:
    :return:
    """

    # \begin{table}[t]
    #     \centering
    #     \resizebox{0.7\columnwidth}{!}
    #     {\begin{tabular}{l cc c ccc}
    #         \toprule
    #           & \multicolumn{2}{c}{\#Entities} && \multicolumn{3}{c}{\#Mentions}  \\
    #            \cmidrule(lr){2-3}\cmidrule(lr){5-7}
    #           Cut & Shared & New && \# Train & \# Validation & \# Test \\
    #          \midrule
    #         2013 & ? & ? && ? & ? & ? \\
    # 	 \bottomrule
    #     \end{tabular}}
    #     \caption{Distribution across subsets.}
    #     \label{tab:dataset}
    # \end{table}
    base_input_path = config['base_input_path']

    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)

    final_output_str = '\\begin{table}[t]\n'
    final_output_str += '\\centering\n'
    final_output_str += '\\resizebox{0.7\\columnwidth}{!}\n'
    final_output_str += '{\\begin{tabular}{l cc c ccc}\n'
    final_output_str += '\t\\toprule\n'
    final_output_str += '\t & \\multicolumn{2}{c}{\#Entities} && \\multicolumn{3}{c}{\#Mentions} \\\\ \n'
    final_output_str += '\t\\cmidrule(lr){2-3}\\cmidrule(lr){5-7}\n'
    final_output_str += '\tCut & \\ Shared & \\ New && \\# Train & \\# Validation & \\# Test \\\\ \n'
    final_output_str += '\t\\midrule\n'
    for curr_year in sorted_cuts:
        logger.info('processing year %s' % curr_year)
        year_config = config['time_cuts'][curr_year]
        train_path = os.path.join(base_input_path, year_config['train_path'])
        validation_path = os.path.join(base_input_path, year_config['validation_path'])
        test_path = os.path.join(base_input_path, year_config['test_path'])
        subsets_distribution = get_subsets_distribution(train_path=train_path,
                                                        validation_path=validation_path,
                                                        test_path=test_path,
                                                        config=config)
        final_output_str += '\t\t{} & {} & {} && {} & {} & {} \\\\ \n' \
            .format(curr_year,
                    subsets_distribution['train_distribution']['nr_entities_shared'],
                    subsets_distribution['train_distribution']['nr_entities_new'],
                    subsets_distribution['train_distribution']['nr_mentions'],
                    subsets_distribution['validation_distribution']['nr_mentions'],
                    subsets_distribution['test_distribution']['nr_mentions'])
    final_output_str += '\t\\bottomrule\n'
    final_output_str += '\t\\end{tabular}}\n'
    final_output_str += '\t\\caption{Distribution across subsets.}\n'
    final_output_str += '\t\\label{tab:dataset}\n'
    final_output_str += '\\end{table}\n'
    logger.info(final_output_str)


def print_latex_subset_distribution_tuples_v2(config, train_tuples, validation_tuples, test_tuples):
    """
    Just another table distribution which seems better given that currently the number of entities is shared
    between the subsets, and there are also shared entities across cuts (random), to not generate too much
    repetition in the table.
    :param config:
    :return:
    """

    # \begin{table}[t]
    #     \centering
    #     \resizebox{0.7\columnwidth}{!}
    #     {\begin{tabular}{l cc c ccc}
    #         \toprule
    #           & \multicolumn{2}{c}{\#Entities} && \multicolumn{3}{c}{\#Mentions}  \\
    #            \cmidrule(lr){2-3}\cmidrule(lr){5-7}
    #           Cut & Shared & New && \# Train & \# Validation & \# Test \\
    #          \midrule
    #         2013 & ? & ? && ? & ? & ? \\
    # 	 \bottomrule
    #     \end{tabular}}
    #     \caption{Distribution across subsets.}
    #     \label{tab:dataset}
    # \end{table}
    base_input_path = config['base_input_path']

    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)

    final_output_str = '\\begin{table}[t]\n'
    final_output_str += '\\centering\n'
    final_output_str += '\\resizebox{0.7\\columnwidth}{!}\n'
    final_output_str += '{\\begin{tabular}{l cc c ccc}\n'
    final_output_str += '\t\\toprule\n'
    final_output_str += '\t & \\multicolumn{2}{c}{\#Entities} && \\multicolumn{3}{c}{\#Mentions} \\\\ \n'
    final_output_str += '\t\\cmidrule(lr){2-3}\\cmidrule(lr){5-7}\n'
    final_output_str += '\tCut & \\ Shared & \\ New && \\# Train & \\# Validation & \\# Test \\\\ \n'
    final_output_str += '\t\\midrule\n'
    for curr_year in sorted_cuts:
        logger.info('processing year %s' % curr_year)
        subsets_distribution = get_subsets_distribution_v2(train_tuples=train_tuples[curr_year],
                                                           validation_tuples=validation_tuples[curr_year],
                                                           test_tuples=test_tuples[curr_year],
                                                           config=config)
        final_output_str += '\t\t{} & {} & {} && {} & {} & {} \\\\ \n' \
            .format(curr_year,
                    subsets_distribution['train_distribution']['nr_entities_shared'],
                    subsets_distribution['train_distribution']['nr_entities_new'],
                    subsets_distribution['train_distribution']['nr_mentions'],
                    subsets_distribution['validation_distribution']['nr_mentions'],
                    subsets_distribution['test_distribution']['nr_mentions'])
    final_output_str += '\t\\bottomrule\n'
    final_output_str += '\t\\end{tabular}}\n'
    final_output_str += '\t\\caption{Distribution across subsets.}\n'
    final_output_str += '\t\\label{tab:dataset}\n'
    final_output_str += '\\end{table}\n'
    # just print since the timestamp better not to print it [?]
    print(final_output_str)


def print_latex_subset_distribution_v3(config):
    """
    The one that divides mentions in new and shared.
    :param config:
    :return:
    """

    # \begin{table}[t]
    #     \centering
    #     \resizebox{0.7\columnwidth}{!}
    #     {\begin{tabular}{l cc c ccc}
    #         \toprule
    #           & \multicolumn{2}{c}{\#Entities} && \multicolumn{3}{c}{\#Mentions}  \\
    #            \cmidrule(lr){2-3}\cmidrule(lr){5-7}
    #           Cut & Shared & New && \# Train & \# Validation & \# Test \\
    #          \midrule
    #         2013 & ? & ? && ? & ? & ? \\
    # 	 \bottomrule
    #     \end{tabular}}
    #     \caption{Distribution across subsets.}
    #     \label{tab:dataset}
    # \end{table}
    base_input_path = config['base_input_path']

    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)

    final_output_str = '\\begin{table}[t]\n'
    final_output_str += '\\centering\n'
    final_output_str += '\\resizebox{0.7\\columnwidth}{!}\n'
    final_output_str += '{\\begin{tabular}{l cc c ccc c ccc}\n'
    final_output_str += '\t\\toprule\n'
    final_output_str += '\t & \\multicolumn{2}{c}{\#Entities} && \\multicolumn{3}{c}{\#Mentions Shared} && \\multicolumn{3}{c}{\#Mentions New} \\\\ \n'
    final_output_str += '\t\\cmidrule(lr){2-3}\\cmidrule(lr){5-7}\\cmidrule(lr){9-11}\n'
    final_output_str += '\tCut & \\ Shared & \\ New && \\# Train & \\# Validation & \\# Test && \\# Train & \\# Validation & \\# Test \\\\ \n'
    final_output_str += '\t\\midrule\n'
    for curr_year in sorted_cuts:
        logger.info('processing year %s' % curr_year)
        year_config = config['time_cuts'][curr_year]
        train_path = os.path.join(base_input_path, year_config['train_path'])
        validation_path = os.path.join(base_input_path, year_config['validation_path'])
        test_path = os.path.join(base_input_path, year_config['test_path'])
        subsets_distribution = get_subsets_distribution(train_path=train_path,
                                                        validation_path=validation_path,
                                                        test_path=test_path,
                                                        config=config)
        final_output_str += '\t\t{} & {} & {} && {} & {} & {} && {} & {} & {} \\\\ \n' \
            .format(curr_year,
                    subsets_distribution['train_distribution']['nr_entities_shared'],
                    subsets_distribution['train_distribution']['nr_entities_new'],
                    subsets_distribution['train_distribution']['nr_mentions_shared'],
                    subsets_distribution['validation_distribution']['nr_mentions_shared'],
                    subsets_distribution['test_distribution']['nr_mentions_shared'],
                    subsets_distribution['train_distribution']['nr_mentions_new'],
                    subsets_distribution['validation_distribution']['nr_mentions_new'],
                    subsets_distribution['test_distribution']['nr_mentions_new'])
    final_output_str += '\t\\bottomrule\n'
    final_output_str += '\t\\end{tabular}}\n'
    final_output_str += '\t\\caption{Distribution across subsets.}\n'
    final_output_str += '\t\\label{tab:dataset}\n'
    final_output_str += '\\end{table}\n'
    print(final_output_str)


def print_latex_subset_distribution_tuples_v3(config, train_tuples, validation_tuples, test_tuples):
    """
    The one that divides mentions in new and shared.
    :param config:
    :return:
    """

    # \begin{table}[t]
    #     \centering
    #     \resizebox{0.7\columnwidth}{!}
    #     {\begin{tabular}{l cc c ccc}
    #         \toprule
    #           & \multicolumn{2}{c}{\#Entities} && \multicolumn{3}{c}{\#Mentions}  \\
    #            \cmidrule(lr){2-3}\cmidrule(lr){5-7}
    #           Cut & Shared & New && \# Train & \# Validation & \# Test \\
    #          \midrule
    #         2013 & ? & ? && ? & ? & ? \\
    # 	 \bottomrule
    #     \end{tabular}}
    #     \caption{Distribution across subsets.}
    #     \label{tab:dataset}
    # \end{table}

    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)

    final_output_str = '\\begin{table}[t]\n'
    final_output_str += '\\centering\n'
    final_output_str += '\\resizebox{0.7\\columnwidth}{!}\n'
    final_output_str += '{\\begin{tabular}{l cc c ccc c ccc}\n'
    final_output_str += '\t\\toprule\n'
    final_output_str += '\t & \\multicolumn{2}{c}{\#Entities} && \\multicolumn{3}{c}{\#Mentions Shared} && \\multicolumn{3}{c}{\#Mentions New} \\\\ \n'
    final_output_str += '\t\\cmidrule(lr){2-3}\\cmidrule(lr){5-7}\\cmidrule(lr){9-11}\n'
    final_output_str += '\tCut & \\ Shared & \\ New && \\# Train & \\# Validation & \\# Test && \\# Train & \\# Validation & \\# Test \\\\ \n'
    final_output_str += '\t\\midrule\n'
    for curr_year in sorted_cuts:
        logger.info('processing year %s' % curr_year)
        subsets_distribution = get_subsets_distribution_v2(train_tuples=train_tuples[curr_year],
                                                           validation_tuples=validation_tuples[curr_year],
                                                           test_tuples=test_tuples[curr_year],
                                                           config=config)
        final_output_str += '\t\t{} & {} & {} && {} & {} & {} && {} & {} & {} \\\\ \n' \
            .format(curr_year,
                    subsets_distribution['train_distribution']['nr_entities_shared'],
                    subsets_distribution['train_distribution']['nr_entities_new'],
                    subsets_distribution['train_distribution']['nr_mentions_shared'],
                    subsets_distribution['validation_distribution']['nr_mentions_shared'],
                    subsets_distribution['test_distribution']['nr_mentions_shared'],
                    subsets_distribution['train_distribution']['nr_mentions_new'],
                    subsets_distribution['validation_distribution']['nr_mentions_new'],
                    subsets_distribution['test_distribution']['nr_mentions_new'])
    final_output_str += '\t\\bottomrule\n'
    final_output_str += '\t\\end{tabular}}\n'
    final_output_str += '\t\\caption{Distribution across subsets.}\n'
    final_output_str += '\t\\label{tab:dataset}\n'
    final_output_str += '\\end{table}\n'
    print(final_output_str)


def print_latex_subset_distribution(config):
    base_input_path = config['base_input_path']
    #     \begin{table}[t]
    #     \centering
    #     \resizebox{1.0\columnwidth}{!}
    #     {\begin{tabular}{l cc c cc c cc}
    #         \toprule
    #           & \multicolumn{2}{c}{Train} && \multicolumn{2}{c}{Validation} && \multicolumn{2}{c}{Test} \\
    #           \cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){8-9}
    #           Snapshot & \# Entities & \# Mentions && \# Entities & \# Mentions && \# Entities & \# Mentions \\
    #          \midrule
    # 	 2013 & ? & ? && ? & ? && ? & ? \\
    # 	 2014 & ? & ? && ? & ? && ? & ? \\
    # 	 2015 & ? & ? && ? & ? && ? & ? \\
    # 	 2016 & ? & ? && ? & ? && ? & ? \\
    # 	 2017 & ? & ? && ? & ? && ? & ? \\
    # 	 2018 & ? & ? && ? & ? && ? & ? \\
    # 	 2019 & ? & ? && ? & ? && ? & ? \\
    # 	 2020 & ? & ? && ? & ? && ? & ? \\
    # 	 2021 & ? & ? && ? & ? && ? & ? \\
    # 	 2022 & ? & ? && ? & ? && ? & ? \\
    # 	 \bottomrule
    #     \end{tabular}}
    #     \caption{Distribution across subsets.}
    #     \label{tab:dataset}
    # \end{table}
    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)

    final_output_str = '\\begin{table}[t]\n'
    final_output_str += '\\centering\n'
    final_output_str += '\\resizebox{1.0\\columnwidth}{!}\n'
    final_output_str += '\\{\\begin{tabular}{l cc c cc c cc}\n'
    final_output_str += '\t\\toprule\n'
    final_output_str += '\t& \\multicolumn{2}{c}{Train} && \\multicolumn{2}{c}{Validation} && ' \
                        '\\multicolumn{2}{c}{Test} \\\\ \n'
    final_output_str += '\t\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}\n'
    final_output_str += '\tSnapshot & \\# Entities & \\# Mentions && \\# Entities & \\# Mentions && \\# Entities ' \
                        '& \\# Mentions \\\\ \n'
    final_output_str += '\t\\midrule\n'
    for curr_year in sorted_cuts:
        logger.info('processing year %s' % curr_year)
        year_config = config['time_cuts'][curr_year]
        train_path = os.path.join(base_input_path, year_config['train_path'])
        validation_path = os.path.join(base_input_path, year_config['validation_path'])
        test_path = os.path.join(base_input_path, year_config['test_path'])
        subsets_distribution = get_subsets_distribution(train_path=train_path,
                                                        validation_path=validation_path,
                                                        test_path=test_path,
                                                        config=config)
        final_output_str += '\t\t{} & {} & {} && {} & {} && {} & {} \\\\ \n' \
            .format(curr_year,
                    subsets_distribution['train_distribution']['nr_entities'],
                    subsets_distribution['train_distribution']['nr_mentions'],
                    subsets_distribution['validation_distribution']['nr_entities'],
                    subsets_distribution['validation_distribution']['nr_mentions'],
                    subsets_distribution['test_distribution']['nr_entities'],
                    subsets_distribution['test_distribution']['nr_mentions'])
    final_output_str += '\t\\bottomrule\n'
    final_output_str += '\t\\end{tabular}}\n'
    final_output_str += '\t\\caption{Distribution across subsets.}\n'
    final_output_str += '\t\\label{tab:dataset}\n'
    final_output_str += '\\end{table}\n'
    print(final_output_str)
