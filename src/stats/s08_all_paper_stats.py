# displays all the paper stats
import argparse
import json
import logging
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate biencoder\'s output')
    parser.add_argument('--config_file', required=False, type=str,
                        # TODO
                        default='TODO',
                        help='The config file that contains all the parameters')
    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))
    path_jaccard_candidate_similarity = config['path_jaccard_candidate_similarity']
    path_jaccard_entities = config['path_jaccard_entities']
    path_jaccard_mentions = config['path_jaccard_mentions']
    path_results = config['path_results']
    path_results_new_type = config['path_results_new_type']

    path_results_new_type_trained_year = config['path_results_new_type_trained_year']
    path_results_new_type_other_years = config['path_results_new_type_other_years']

    path_nr_wiki_candidates = config['path_nr_wiki_candidates']
    path_mentions_per_entity = config['path_mentions_per_entity']

    path_filter_stats = config['path_filter_stats']

    path_detail_results = config['path_detail_results']
    #
    #
    output_path_jaccard_cand_similarity = config['output_path_jaccard_cand_similarity']
    output_path_jaccard_mentions = config['output_path_jaccard_mentions']
    output_path_jaccard_entities = config['output_path_jaccard_entities']
    output_path_edit_distance_entities = config['output_path_edit_distance_entities']
    output_path_results2_new = config['output_path_results_new']
    output_path_results_new_type_s1 = config['output_path_results_new_type_s1']
    output_path_results_new_type_s2 = config['output_path_results_new_type_s2']
    output_path_results_new_type_s3 = config['output_path_results_new_type_s3']
    output_path_results_new_type_s4 = config['output_path_results_new_type_s4']
    output_path_results_new_vs_shared = config['output_path_results_new_vs_shared']
    output_path_results_train_type_s1 = config['output_path_results_train_type_s1']
    output_path_results_train_type_s2 = config['output_path_results_train_type_s2']
    output_path_results_train_type_s3 = config['output_path_results_train_type_s3']
    output_path_results_train_type_s4 = config['output_path_results_train_type_s4']
    output_path_results_train_type_delta = config['output_path_results_train_type_delta']
    output_path_nr_wiki_candidates = config['output_path_nr_wiki_candidates']
    output_path_mentions_per_entity = config['output_path_mentions_per_entity']
    output_path_filter_stats = config['output_path_filter_stats']
    output_path_len_stats = config['output_path_len_stats']
    output_path_results_k_type = config['output_path_results_k_type']
    output_path_results_k_type_delta = config['output_path_results_k_type_delta']
    output_path_results_temporal_offset = config['output_path_results_temporal_offset']
    output_path_results_acc_k_all = config['output_path_results_acc_k_all']
    #
    #
    men_per_entity_dir = os.path.dirname(path_mentions_per_entity)

    df_detail_results_stats = pd.read_csv(open(path_detail_results, 'rt'), sep='\t', header=[0])

    df_jaccard_candidate_similarity = pd.read_csv(open(path_jaccard_candidate_similarity, 'rt'), sep=',', header=[0])
    df_jaccard_entities = pd.read_csv(open(path_jaccard_entities, 'rt'), sep='\t', header=[0])
    df_jaccard_mentions = pd.read_csv(open(path_jaccard_mentions, 'rt'), sep='\t', header=[0])
    df_nr_wiki_candidates = pd.read_csv(open(path_nr_wiki_candidates, 'rt'), sep='\t', header=[0])
    df_nr_wiki_candidates['nr_candidates'] /= 1000000
    df_results = pd.read_csv(open(path_results, 'rt'), sep='\t', header=[0])

    df_results_temporal_offset = df_results.copy(deep=True)

    df_results_temporal_offset['offset'] = \
        df_results_temporal_offset['year_finetune'] - df_results_temporal_offset['year_predicted']

    df_results_temporal_offset['offset_delta'] = 0.0
    for curr_idxrow, curr_row in df_results_temporal_offset.iterrows():
        y_pred = curr_row['year_predicted']
        df_res_in_snapshot = \
            df_results_temporal_offset[
                (df_results_temporal_offset['year_predicted'] == y_pred) &
                (df_results_temporal_offset['year_finetune'] == y_pred) &
                (df_results_temporal_offset['category'] == curr_row['category'])
                ]
        res_in_snapshot = df_res_in_snapshot.iloc[0]['acc@64']
        res_curr = curr_row['acc@64']
        offset_delta = res_curr - res_in_snapshot
        df_results_temporal_offset.iloc[
            curr_idxrow, df_results_temporal_offset.columns.get_loc('offset_delta')] = offset_delta

    df_results_k_in_rows = df_results.melt(id_vars=['year_finetune', 'subset', 'category', 'year_predicted', 'model'],
                                           value_vars=['acc@1', 'acc@2', 'acc@4', 'acc@8', 'acc@16', 'acc@32',
                                                       'acc@64'],
                                           value_name='accuracy',
                                           var_name='k')
    df_results_k_in_rows['k'] = df_results_k_in_rows['k'].str.replace('acc@', '')
    df_results_k_in_rows['finetune_type'] = ''

    df_results_k_in_rows.loc[df_results_k_in_rows['year_predicted'] != df_results_k_in_rows['year_finetune'],
                             'finetune_type'] = 'Out-of-Snapshot'

    df_results_k_in_rows.loc[df_results_k_in_rows['year_predicted'] == df_results_k_in_rows['year_finetune'],
                             'finetune_type'] = 'In-Snapshot'

    df_results_k_in_rows.loc[df_results_k_in_rows['category'] == 'shared', 'category'] = 'Continual'
    df_results_k_in_rows.loc[df_results_k_in_rows['category'] == 'new_entities', 'category'] = 'New'

    # this is the one that works only on the year 2021 (see filter further in the script)
    df_new2_by_type_acc_k = pd.read_csv(open(path_results_new_type, 'rt'), sep=',', header=[0])

    # this is the one that takes the average across all the years, but to this end the data must come from
    # micro-results (i.e., from path_detail_results)
    df_by_type_acc_k = df_detail_results_stats
    #
    df_results_new_type_oy = pd.read_csv(open(path_results_new_type_other_years, 'rt'), sep=',', header=[0])
    df_results_new_type_ty = pd.read_csv(open(path_results_new_type_trained_year, 'rt'), sep=',', header=[0])
    #
    df_results_new_type_oy.rename(columns={'filter-covid': 'filter-covid-other-years',
                                           'filter-other': 'filter-other-other-years'}, inplace=True)
    df_results_new_type_ty.rename(columns={'filter-covid': 'filter-covid-trained-year',
                                           'filter-other': 'filter-other-trained-year'}, inplace=True)

    #
    # merge df_results_new_type_oy and df_results_new_type_ty
    df_results_new_type = pd.merge(df_results_new_type_oy, df_results_new_type_ty,
                                   left_on=('snapshot', 'k'), right_on=('snapshot', 'k'), how='inner')
    df_results_new_type = df_results_new_type[['k', 'filter-covid-other-years', 'filter-other-other-years',
                                               'filter-covid-trained-year', 'filter-other-trained-year']]
    df_results_new_type['k'] = df_results_new_type['k'].astype(str)

    #
    #
    df_filter_stats = pd.read_csv(open(path_filter_stats, 'rt'), sep='\t', header=[0])
    df_mentions_per_entities = pd.read_csv(open(path_mentions_per_entity, 'rt'), sep='\t', header=[0])

    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)

    fig_plt_ent_jaccard_similarity, axs_ent_jaccard_similarity = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_men_jaccard_similarity, axs_men_jaccard_similarity = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_ent_ed_dist_similarity, axs_ent_ed_dist_similarity = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_cand_similarity, axs_cand_similarity = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_new_vs_shared, axs_results_new_vs_shared = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_train_type_s1, axs_results_train_type_s1 = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_train_type_s2, axs_results_train_type_s2 = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_train_type_s3, axs_results_train_type_s3 = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_train_type_s4, axs_results_train_type_s4 = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_train_type_delta, axs_results_train_type_delta = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_k_type, axs_results_k_type = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_k_type_delta, axs_results_k_type_delta = plt.subplots(1, 1, figsize=(4, 4))

    #
    fig_plt_results_new2_acc_k, axs_results_new2_acc_k = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_acc_k_all, axs_results_acc_k_all = plt.subplots(1, 1, figsize=(6, 6))
    fig_plt_results_new_type_s1, axs_results_new_type_s1 = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_new_type_s2, axs_results_new_type_s2 = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_new_type_s3, axs_results_new_type_s3 = plt.subplots(1, 1, figsize=(4, 4))
    fig_plt_results_new_type_s4, axs_results_new_type_s4 = plt.subplots(1, 1, figsize=(4, 4))
    #

    fig_plt_filter_stats, axs_filter_stats = plt.subplots(1, 1, figsize=(7, 4))
    fig_plt_nr_wiki_candidates, axs_nr_wiki_candidates = plt.subplots(1, 1, figsize=(4, 4))

    fig_plt_len_stats, axs_len_stats = plt.subplots(1, 1, figsize=(4, 4))

    fig_results_temporal_offset, axs_results_temporal_offset = plt.subplots(1, 1, figsize=(4, 4))

    axs_len_stats.set_ylabel('TODO y len stats', fontsize=14)
    axs_len_stats.set_title('TODO title len stats', fontsize=14, pad=10)

    axs_nr_wiki_candidates.set_ylabel('Nr. Wiki Entities (Millions)', fontsize=14)
    axs_nr_wiki_candidates.set_title('Number of Wikipedia Entities', fontsize=14, pad=10)

    axs_results_new2_acc_k.set_ylabel('Accuracy@K', fontsize=14)

    axs_results_acc_k_all.set_ylabel('Accuracy@K', fontsize=14)
    #
    axs_results_new_type_s1.set_ylabel('Accuracy@K', fontsize=14)
    axs_results_new_type_s2.set_ylabel('Accuracy@K', fontsize=14)
    axs_results_new_type_s3.set_ylabel('Accuracy@K', fontsize=14)
    axs_results_new_type_s4.set_ylabel('Accuracy@K', fontsize=14)

    axs_filter_stats.set_ylabel('Filter', fontsize=14)
    axs_filter_stats.set_title('Filter Statistics', fontsize=14)
    #
    axs_results_new_vs_shared.set_ylabel('Accuracy@64', fontsize=14)
    axs_results_new_vs_shared.set_title('Average Performance', fontsize=14, pad=10)
    #
    axs_results_train_type_s1.set_ylabel('Accuracy@64', fontsize=14)
    axs_results_train_type_s1.set_title('Average Performance', fontsize=14, pad=10)
    axs_results_train_type_s2.set_ylabel('Accuracy@64', fontsize=14)
    axs_results_train_type_s2.set_title('Average Performance', fontsize=14, pad=10)
    axs_results_train_type_s3.set_ylabel('Accuracy@64', fontsize=14)
    axs_results_train_type_s3.set_title('Average Performance', fontsize=14, pad=10)
    axs_results_train_type_s4.set_ylabel('Accuracy@64', fontsize=14)
    axs_results_train_type_s4.set_title('Average Performance', fontsize=14, pad=10)

    axs_results_train_type_delta.set_ylabel(r'$\Delta$ in Accuracy@64', fontsize=14)

    axs_results_k_type.set_ylabel('Accuracy?', fontsize=14)
    axs_results_k_type.set_title('Impact of Finetuning', fontsize=14, pad=10)
    #
    axs_results_k_type_delta.set_ylabel(r'$\Delta$ in Accuracy@K', fontsize=14)
    #
    axs_results_temporal_offset.set_ylabel(r'$\Delta$ in Accuracy@64', fontsize=14)
    #
    axs_cand_similarity.set_ylabel('Jaccard Similarity', fontsize=14)
    axs_cand_similarity.set_title('Candidate Similarity', fontsize=14, pad=10)
    #
    axs_men_jaccard_similarity.set_ylabel('Jaccard Similarity', fontsize=14)
    axs_men_jaccard_similarity.set_title('Mention Similarity', fontsize=14, pad=10)
    #
    axs_ent_jaccard_similarity.set_ylabel('Jaccard Similarity', fontsize=14)
    axs_ent_jaccard_similarity.set_title('Entity Similarity', fontsize=14, pad=10)
    #
    axs_ent_ed_dist_similarity.set_ylabel('Normalized Edit Distance', fontsize=14)
    axs_ent_ed_dist_similarity.set_title('Entity Edit Distance', fontsize=14, pad=10)
    #
    #
    x_ticks = list()
    sorted_cuts = config['sorted_cuts']
    for curr_cut in sorted_cuts:
        x_ticks.append(int(curr_cut))

    df_results_temporal_offset = df_results_temporal_offset[df_results_temporal_offset['subset'] == 'test']
    df_results_temporal_offset = df_results_temporal_offset[df_results_temporal_offset['model'] == 'biencoder']

    df_results_temporal_offset = df_results_temporal_offset.groupby(['category', 'offset']) \
        .mean().reset_index()[['category', 'offset', 'offset_delta']]

    #
    df_detail_len_stats = df_detail_results_stats[
        df_detail_results_stats['finetune_year'] != df_detail_results_stats['subset_year']]
    df_detail_len_stats['rank_target_len'] = pd.qcut(df_detail_len_stats['target_length'], 15, labels=False)
    df_detail_len_stats = df_detail_len_stats.groupby(['target_entity_type', 'rank_target_len']) \
        .mean().reset_index()[['target_entity_type', 'rank_target_len', 'acc@64']]

    df_detail_len_stats = df_detail_len_stats.pivot_table('acc@64', ['rank_target_len'], 'target_entity_type') \
        .reset_index()

    df_detail_len_stats.plot(ax=axs_len_stats, x='rank_target_len', grid=True)

    df_jaccard_mentions.plot(ax=axs_men_jaccard_similarity, x='year', y='jackard_similarity', marker='o', grid=True)
    axs_men_jaccard_similarity.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)

    df_nr_wiki_candidates.plot.bar(ax=axs_nr_wiki_candidates, x='year', y='nr_candidates', ylim=(3.5, 6.5), grid=True)
    axs_nr_wiki_candidates.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)

    df_jaccard_entities.plot(ax=axs_ent_jaccard_similarity, x='year', y='jackard_similarity', marker='o', grid=True)
    axs_ent_jaccard_similarity.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)

    df_jaccard_candidate_similarity.plot(ax=axs_cand_similarity, x='Year', y='Jaccard Similarity', marker='o',
                                         grid=True)
    axs_cand_similarity.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)

    df_jaccard_entities.plot(ax=axs_ent_ed_dist_similarity, x='year', y='edit_distance', marker='o', grid=True)
    axs_ent_ed_dist_similarity.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)

    # BEGIN massaging df_results to get the right thing
    df_results_k_in_rows = df_results_k_in_rows[df_results_k_in_rows['subset'] == 'test']
    df_results_k_in_rows = df_results_k_in_rows[df_results_k_in_rows['model'] == 'biencoder']
    df_results_k_in_rows['category_type'] = \
        df_results_k_in_rows['category'] + ' ' + df_results_k_in_rows['finetune_type']

    df_results_k_in_rows = df_results_k_in_rows[['category_type', 'k', 'accuracy']]
    df_results_k_in_rows = df_results_k_in_rows.groupby(['category_type', 'k']).mean() \
        .reset_index()[['category_type', 'k', 'accuracy']]

    df_results_k_in_rows = df_results_k_in_rows.pivot_table('accuracy', ['k'], 'category_type').reset_index()
    df_results_k_in_rows['k'] = df_results_k_in_rows['k'].astype(int)

    df_results_k_in_rows.sort_values(by=['k'], inplace=True)
    df_results_k_in_rows['k'] = df_results_k_in_rows['k'].astype(str)

    color_dict_train_type = {'Continual Out-of-Snapshot': '#1f77b4', 'New Out-of-Snapshot': '#ff7f0e',
                             'Continual In-Snapshot': '#1f77b4', 'New In-Snapshot': '#ff7f0e'}

    colors = [color_dict_train_type.get(x, '#333333') for x in df_results_k_in_rows.columns if x != 'k']

    df_results_k_in_rows.plot(ax=axs_results_k_type, x='k', marker='o', grid=True, color=colors)
    linestyles = ['-', '--', '-', '--']
    markers = ['o', 'x', 'o', 'x']
    for i, line in enumerate(axs_results_k_type.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])

    df_results_k_in_rows_delta = df_results_k_in_rows.copy(deep=True)
    df_results_k_in_rows_delta['delta continual'] = df_results_k_in_rows_delta['Continual In-Snapshot'] - \
                                                    df_results_k_in_rows_delta['Continual Out-of-Snapshot']
    df_results_k_in_rows_delta['delta new'] = df_results_k_in_rows_delta['New In-Snapshot'] - \
                                              df_results_k_in_rows_delta['New Out-of-Snapshot']

    df_results_k_in_rows_delta = df_results_k_in_rows_delta[['k', 'delta continual', 'delta new']]
    df_results_k_in_rows_delta.rename(columns={'delta continual': 'Continual Entities', 'delta new': 'New Entities'},
                                      inplace=True)

    df_results_k_in_rows_delta.plot.bar(ax=axs_results_k_type_delta, x='k', grid=True, rot=0,
                                        color=['#ff7f0e', '#1f77b4'])
    axs_results_k_type_delta.legend(title=None)
    fig_plt_results_k_type_delta.tight_layout()
    ###
    # probably pivoting needed
    df_results_temporal_offset = df_results_temporal_offset.pivot_table('offset_delta', ['offset'], 'category') \
        .reset_index()
    df_results_temporal_offset.rename(columns={'shared': 'Continual Entities',
                                               'new_entities': 'New Entities'}, inplace=True)

    df_results_temporal_offset.plot.bar(ax=axs_results_temporal_offset, x='offset', grid=True,
                                        color=['#1f77b4', '#ff7f0e'])
    axs_results_temporal_offset.legend(title=None)

    df_results = df_results[df_results['subset'] == 'test']
    df_results = df_results[df_results['model'] == 'biencoder']

    df_results = df_results[['acc@64', 'year_finetune', 'category', 'year_predicted']]

    df_results_train_not_on_predicted = df_results[df_results['year_finetune'] != df_results['year_predicted']]
    df_results_train_on_predicted = df_results[df_results['year_finetune'] == df_results['year_predicted']]

    df_results_train_not_on_predicted = df_results_train_not_on_predicted \
        .groupby(['category', 'year_predicted']).mean().reset_index()[['category', 'year_predicted', 'acc@64']]

    df_results_train_on_predicted = df_results_train_on_predicted \
        .groupby(['category', 'year_predicted']).mean().reset_index()[['category', 'year_predicted', 'acc@64']]

    df_results_train_not_on_predicted['type_train'] = 'different_year'
    df_results_train_on_predicted['type_train'] = 'same_year'
    df_results_train_type_s4 = pd.concat([df_results_train_on_predicted, df_results_train_not_on_predicted])
    df_results_train_type_s4['category_type_train'] = ''

    df_results_train_type_s4.loc[((df_results_train_type_s4['category'] == 'new_entities') &
                                  (df_results_train_type_s4['type_train'] == 'same_year')),
                                 'category_type_train'] = 'New In-Snapshot'

    df_results_train_type_s4.loc[((df_results_train_type_s4['category'] == 'new_entities') &
                                  (df_results_train_type_s4['type_train'] == 'different_year')),
                                 'category_type_train'] = 'New Out-of-Snapshot'

    df_results_train_type_s4.loc[((df_results_train_type_s4['category'] == 'shared') &
                                  (df_results_train_type_s4['type_train'] == 'different_year')),
                                 'category_type_train'] = 'Continual Out-of-Snapshot'

    df_results_train_type_s4.loc[((df_results_train_type_s4['category'] == 'shared') &
                                  (df_results_train_type_s4['type_train'] == 'same_year')),
                                 'category_type_train'] = 'Continual In-Snapshot'

    df_results_train_type_s4 = df_results_train_type_s4[['year_predicted', 'acc@64', 'category_type_train']]
    df_results_train_type_s4 = df_results_train_type_s4.pivot_table('acc@64', ['year_predicted'], 'category_type_train') \
        .reset_index()

    color_dict_train_type = {'Continual Out-of-Snapshot': '#ff7f0e', 'New Out-of-Snapshot': '#1f77b4',
                             'Continual In-Snapshot': '#ff7f0e', 'New In-Snapshot': '#1f77b4'}

    # changes the order of the columns, so no need to do extra manipulations on axs_results_train_type
    df_results_train_type_s4 = df_results_train_type_s4[['year_predicted',
                                                         'Continual Out-of-Snapshot',
                                                         'New Out-of-Snapshot',
                                                         'Continual In-Snapshot',
                                                         'New In-Snapshot'
                                                         ]]

    df_results_train_type_s1 = df_results_train_type_s4[['year_predicted',
                                                         'Continual Out-of-Snapshot']]
    df_results_train_type_s2 = df_results_train_type_s4[['year_predicted',
                                                         'Continual Out-of-Snapshot',
                                                         'New Out-of-Snapshot']]
    df_results_train_type_s3 = df_results_train_type_s4[['year_predicted',
                                                         'Continual Out-of-Snapshot',
                                                         'New Out-of-Snapshot',
                                                         'Continual In-Snapshot']]

    colors = [color_dict_train_type.get(x, '#333333') for x in df_results_train_type_s4.columns if
              x != 'year_predicted']

    df_results_train_type_s1.plot(ax=axs_results_train_type_s1, x='year_predicted', marker='o', grid=True, color=colors)
    df_results_train_type_s2.plot(ax=axs_results_train_type_s2, x='year_predicted', marker='o', grid=True, color=colors)
    df_results_train_type_s3.plot(ax=axs_results_train_type_s3, x='year_predicted', marker='o', grid=True, color=colors)
    df_results_train_type_s4.plot(ax=axs_results_train_type_s4, x='year_predicted', marker='o', grid=True, color=colors)

    #
    linestyles = ['--']
    markers = ['x']
    for i, line in enumerate(axs_results_train_type_s1.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])
    #
    linestyles = ['--', '--']
    markers = ['x', 'x']
    for i, line in enumerate(axs_results_train_type_s2.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])
    #
    linestyles = ['--', '--', '-']
    markers = ['x', 'x', 'o']
    for i, line in enumerate(axs_results_train_type_s3.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])
    #
    linestyles = ['--', '--', '-', '-']
    markers = ['x', 'x', 'o', 'o']
    for i, line in enumerate(axs_results_train_type_s4.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])

    df_results_train_type_s4['Continual Entities'] = df_results_train_type_s4['Continual In-Snapshot'] - \
                                                     df_results_train_type_s4['Continual Out-of-Snapshot']
    df_results_train_type_s4['New Entities'] = df_results_train_type_s4['New In-Snapshot'] - df_results_train_type_s4[
        'New Out-of-Snapshot']

    df_results_train_type_delta = df_results_train_type_s4[['year_predicted', 'Continual Entities', 'New Entities']]

    color_dict_train_type2 = {'Continual Entities': '#1f77b4', 'New Entities': '#ff7f0e'}

    colors2 = [color_dict_train_type2.get(x, '#333333') \
               for x in df_results_train_type_delta.columns if x != 'year_predicted']

    df_results_train_type_delta \
        .plot.bar(ax=axs_results_train_type_delta, x='year_predicted', rot=0, grid=True,
                  color=['#ff7f0e', '#1f77b4'])

    axs_results_train_type_delta.legend(title=None)

    #
    df_results = df_results.groupby(['category', 'year_predicted']).mean().reset_index()[
        ['category', 'year_predicted', 'acc@64']]

    df_results_pivoted = df_results.pivot_table('acc@64', ['year_predicted'], 'category').reset_index()
    df_results_pivoted = df_results_pivoted[['year_predicted', 'new_entities', 'shared']]
    df_results_pivoted.rename(columns={'shared': 'Continual Entities', 'new_entities': 'New Entities'}, inplace=True)
    df_results_pivoted.plot(ax=axs_results_new_vs_shared, x='year_predicted', marker='o', grid=True)
    os.makedirs(os.path.dirname(config['output_path_results_new_vs_shared_csv']), exist_ok=True)
    df_results_pivoted.to_csv(config['output_path_results_new_vs_shared_csv'], sep='\t', index=False)
    # END  massaging df_results to get the right thing

    ####### BEGIN df_new3_by_type_acc_k
    df_by_type_acc_k = df_by_type_acc_k[['target_entity_type', 'subset', 'model', 'finetune_year', 'subset_year',
                                         'target_entity_filter', 'acc@1', 'acc@2', 'acc@4', 'acc@8', 'acc@16',
                                         'acc@32', 'acc@64']]
    df_by_type_acc_k['finetune_type'] = ''
    df_by_type_acc_k.loc[df_by_type_acc_k['subset_year'] != df_by_type_acc_k['finetune_year'],
                         'finetune_type'] = 'Out-of-Snapshot'

    df_by_type_acc_k.loc[df_by_type_acc_k['subset_year'] == df_by_type_acc_k['finetune_year'],
                         'finetune_type'] = 'In-Snapshot'
    df_by_type_acc_k = df_by_type_acc_k[df_by_type_acc_k['subset'] == 'test']
    df_by_type_acc_k = df_by_type_acc_k[df_by_type_acc_k['model'] == 'biencoder']

    #
    df_by_type_acc_k = df_by_type_acc_k[['target_entity_type', 'finetune_type', 'target_entity_filter',
                                         'acc@1', 'acc@2', 'acc@4', 'acc@8', 'acc@16',
                                         'acc@32', 'acc@64']]
    df_by_type_acc_k_new = df_by_type_acc_k[df_by_type_acc_k['target_entity_type'] == 'new_entities']
    df_by_type_acc_k_new = df_by_type_acc_k_new.groupby(['target_entity_type', 'finetune_type',
                                                         'target_entity_filter']).mean().reset_index()
    #
    df_by_type_acc_k_continual = df_by_type_acc_k[df_by_type_acc_k['target_entity_type'] == 'shared']
    df_by_type_acc_k_continual['target_entity_filter'] = ''
    df_by_type_acc_k_continual = df_by_type_acc_k_continual.groupby(['target_entity_type', 'finetune_type',
                                                                     'target_entity_filter']).mean().reset_index()

    logger.info('df_by_type_acc_k_new.shape: %s' % str(df_by_type_acc_k_new.shape))
    logger.info('df_by_type_acc_k_new.columns: %s' % str(df_by_type_acc_k_new.columns))
    logger.info('df_by_type_acc_k_continual.shape: %s' % str(df_by_type_acc_k_continual.shape))
    logger.info('df_by_type_acc_k_continual.columns: %s' % str(df_by_type_acc_k_continual.columns))

    df_by_type_acc_k_continual['target_entity_filter'] = 'filter-continual'

    df_by_type_acc_k_all = pd.concat([df_by_type_acc_k_continual, df_by_type_acc_k_new])
    df_by_type_acc_k_all = df_by_type_acc_k_all.melt(id_vars=['target_entity_type', 'finetune_type',
                                                              'target_entity_filter'],
                                                     value_vars=['acc@1', 'acc@2', 'acc@4', 'acc@8', 'acc@16', 'acc@32',
                                                                 'acc@64'],
                                                     value_name='Accuracy@K',
                                                     var_name='K')
    df_by_type_acc_k_all['K'] = df_by_type_acc_k_all['K'].str.replace('acc@', '')

    #
    df_by_type_acc_k_all.loc[(df_by_type_acc_k_all['target_entity_type'] == 'shared'),
                             'target_entity_type'] = 'Continual'

    df_by_type_acc_k_all.loc[(df_by_type_acc_k_all['target_entity_type'] == 'new_entities'),
                             'target_entity_type'] = 'New'

    df_by_type_acc_k_all.loc[(df_by_type_acc_k_all['target_entity_filter'] == 'filter-covid'),
                             'target_entity_filter'] = 'COVID-19'

    df_by_type_acc_k_all.loc[(df_by_type_acc_k_all['target_entity_filter'] == 'filter-other'),
                             'target_entity_filter'] = 'Other'

    df_by_type_acc_k_all.loc[(df_by_type_acc_k_all['target_entity_filter'] == 'filter-recurrent'),
                             'target_entity_filter'] = 'Recurrent'

    df_by_type_acc_k_all.loc[(df_by_type_acc_k_all['target_entity_filter'] == 'filter-continual'),
                             'target_entity_filter'] = ''

    df_by_type_acc_k_all['Legend'] = df_by_type_acc_k_all['target_entity_type'] + ' ' + \
                                     df_by_type_acc_k_all['target_entity_filter'] + ' ' + \
                                     df_by_type_acc_k_all['finetune_type']

    df_by_type_acc_k_all.Legend = df_by_type_acc_k_all.Legend.replace(r'\s+', ' ', regex=True)
    #
    #
    df_by_type_acc_k_all = df_by_type_acc_k_all[['K', 'Accuracy@K', 'Legend']]

    df_by_type_acc_k_all = df_by_type_acc_k_all.pivot_table('Accuracy@K', ['K'], 'Legend').reset_index()

    df_by_type_acc_k_all['K'] = df_by_type_acc_k_all['K'].astype(int)

    df_by_type_acc_k_all.sort_values(by=['K'], inplace=True)
    df_by_type_acc_k_all['K'] = df_by_type_acc_k_all['K'].astype(str)

    df_by_type_acc_k_all = df_by_type_acc_k_all[['Continual In-Snapshot', 'Continual Out-of-Snapshot',
                                                 'New COVID-19 In-Snapshot', 'New COVID-19 Out-of-Snapshot',
                                                 'New Recurrent In-Snapshot', 'New Recurrent Out-of-Snapshot',
                                                 'New Other In-Snapshot', 'New Other Out-of-Snapshot', 'K'
                                                 ]]
    color_dict = {'Continual In-Snapshot': '#1F77B4',
                  'Continual Out-of-Snapshot': '#1F77B4',
                  'New COVID-19 In-Snapshot': '#FF7F0E',
                  'New COVID-19 Out-of-Snapshot': '#FF7F0E',
                  'New Other In-Snapshot': '#2DA384',
                  'New Other Out-of-Snapshot': '#2DA384',
                  'New Recurrent In-Snapshot': '#F0564A',
                  'New Recurrent Out-of-Snapshot': '#F0564A'
                  }

    colors = [color_dict.get(x, '#333333') for x in df_by_type_acc_k_all.columns if x != 'K']

    df_by_type_acc_k_all.plot(ax=axs_results_acc_k_all, x='K', marker='o', grid=True, color=colors)
    axs_results_acc_k_all.set_ylim(ymin=0.05, ymax=0.91)

    linestyles = ['-', '--', '-', '--', '-', '--', '-', '--']
    markers = ['o', 'x', 'o', 'x', 'o', 'x', 'o', 'x']
    for i, line in enumerate(axs_results_acc_k_all.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])

    axs_results_acc_k_all.legend(title=None)

    ####### END df_new3_by_type_acc_k
    df_new2_by_type_acc_k = df_new2_by_type_acc_k[df_new2_by_type_acc_k['snapshot'] == 2021]
    df_new2_by_type_acc_k = df_new2_by_type_acc_k[['k', 'filter-covid', 'filter-recurrent', 'filter-other']]
    df_new2_by_type_acc_k['k'] = df_new2_by_type_acc_k['k'].astype(str)

    df_new2_by_type_acc_k.rename(
        columns={'filter-covid': 'COVID-related',
                 'filter-recurrent': 'Recurrent',
                 'filter-other': 'Other'}, inplace=True)

    color_dict = {'COVID-related': '#1f77b4', 'Recurrent': '#ff7f0e', 'Extracted': '#2ca02c'}

    colors = [color_dict.get(x, '#333333') for x in df_new2_by_type_acc_k.columns if x != 'k']
    df_new2_by_type_acc_k.plot(ax=axs_results_new2_acc_k, x='k', marker='o', grid=True, color=colors)
    axs_results_new2_acc_k.set_ylim(ymin=0.03, ymax=0.91)
    #
    # just reorders the columns
    df_results_new_type = df_results_new_type[['filter-covid-other-years', 'filter-covid-trained-year',
                                               'filter-other-other-years', 'filter-other-trained-year', 'k']]
    #
    df_results_new_type.rename(columns={'filter-covid-other-years': 'COVID-19 Out-of-Snapshot',
                                        'filter-covid-trained-year': 'COVID-19 In-Snapshot',
                                        'filter-other-other-years': 'Other Out-of-Snapshot',
                                        'filter-other-trained-year': 'Other In-Snapshot'
                                        }, inplace=True)
    #
    color_dict2 = {'COVID-19 Out-of-Snapshot': '#1f77b4', 'Other Out-of-Snapshot': '#ff7f0e',
                   'COVID-19 In-Snapshot': '#1f77b4', 'Other In-Snapshot': '#ff7f0e'}

    df_results_new_type_s1 = df_results_new_type[['COVID-19 Out-of-Snapshot', 'k']]
    #
    df_results_new_type_s2 = df_results_new_type[['COVID-19 Out-of-Snapshot', 'Other Out-of-Snapshot', 'k']]
    #
    df_results_new_type_s3 = df_results_new_type[['COVID-19 Out-of-Snapshot', 'Other Out-of-Snapshot',
                                                  'COVID-19 In-Snapshot', 'k']]
    #
    df_results_new_type = df_results_new_type[['COVID-19 Out-of-Snapshot', 'Other Out-of-Snapshot',
                                               'COVID-19 In-Snapshot', 'Other In-Snapshot', 'k']]
    #
    colors = [color_dict2.get(x, '#333333') for x in df_results_new_type_s1.columns if x != 'k']
    df_results_new_type_s1.plot(ax=axs_results_new_type_s1, x='k', grid=True, color=colors)
    #
    #
    colors = [color_dict2.get(x, '#333333') for x in df_results_new_type_s2.columns if x != 'k']
    df_results_new_type_s2.plot(ax=axs_results_new_type_s2, x='k', grid=True, color=colors)
    #
    #
    colors = [color_dict2.get(x, '#333333') for x in df_results_new_type_s3.columns if x != 'k']
    df_results_new_type_s3.plot(ax=axs_results_new_type_s3, x='k', grid=True, color=colors)
    #
    #
    colors = [color_dict2.get(x, '#333333') for x in df_results_new_type.columns if x != 'k']
    df_results_new_type.plot(ax=axs_results_new_type_s4, x='k', grid=True, color=colors)
    #

    linestyles = ['--']
    markers = ['x']
    for i, line in enumerate(axs_results_new_type_s1.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])

    linestyles = ['--', '--']
    markers = ['x', 'x']
    for i, line in enumerate(axs_results_new_type_s2.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])

    linestyles = ['--', '--', '-']
    markers = ['x', 'x', 'o']
    for i, line in enumerate(axs_results_new_type_s3.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])

    linestyles = ['--', '--', '-', '-']
    markers = ['x', 'x', 'o', 'o']
    for i, line in enumerate(axs_results_new_type_s4.get_lines()):
        line.set_marker(markers[i])
        line.set_linestyle(linestyles[i])

    axs_results_new_type_s1.set_ylim(ymin=0.1, ymax=0.85)
    axs_results_new_type_s2.set_ylim(ymin=0.1, ymax=0.85)
    axs_results_new_type_s3.set_ylim(ymin=0.1, ymax=0.85)
    axs_results_new_type_s4.set_ylim(ymin=0.1, ymax=0.85)

    #######
    handles, labels = axs_results_new2_acc_k.get_legend_handles_labels()
    labels, handles = zip(*reversed(list(zip(labels, handles))))
    axs_results_new2_acc_k.legend(handles, labels)

    axs_results_new2_acc_k.set_xlabel('K', fontsize=14, labelpad=6)
    axs_results_new2_acc_k.set_xticks(range(df_new2_by_type_acc_k['k'].size))
    axs_results_new2_acc_k.set_xticklabels(df_new2_by_type_acc_k['k'])
    #######
    #######
    axs_results_acc_k_all.set_xlabel('K', fontsize=14, labelpad=6)
    axs_results_acc_k_all.set_xticks(range(df_by_type_acc_k_all['K'].size))
    axs_results_acc_k_all.set_xticklabels(df_by_type_acc_k_all['K'])
    #######
    #
    handles, labels = axs_results_new_type_s1.get_legend_handles_labels()
    axs_results_new_type_s1.legend(handles, labels, loc='upper left')
    axs_results_new_type_s1.set_xlabel('K', fontsize=14, labelpad=6)
    axs_results_new_type_s1.set_xticks(range(df_results_new_type['k'].size))
    axs_results_new_type_s1.set_xticklabels(df_results_new_type['k'])
    #
    #
    handles, labels = axs_results_new_type_s2.get_legend_handles_labels()
    axs_results_new_type_s2.legend(handles, labels, loc='upper left')
    axs_results_new_type_s2.set_xlabel('K', fontsize=14, labelpad=6)
    axs_results_new_type_s2.set_xticks(range(df_results_new_type['k'].size))
    axs_results_new_type_s2.set_xticklabels(df_results_new_type['k'])
    #
    #
    handles, labels = axs_results_new_type_s3.get_legend_handles_labels()
    axs_results_new_type_s3.legend(handles, labels, loc='upper left')
    axs_results_new_type_s3.set_xlabel('K', fontsize=14, labelpad=6)
    axs_results_new_type_s3.set_xticks(range(df_results_new_type['k'].size))
    axs_results_new_type_s3.set_xticklabels(df_results_new_type['k'])
    #
    #
    handles, labels = axs_results_new_type_s4.get_legend_handles_labels()
    axs_results_new_type_s4.legend(handles, labels, loc='upper left')
    axs_results_new_type_s4.set_xlabel('K', fontsize=14, labelpad=6)
    axs_results_new_type_s4.set_xticks(range(df_results_new_type['k'].size))
    axs_results_new_type_s4.set_xticklabels(df_results_new_type['k'])
    #
    #######

    #
    df_filter_stats.sort_values(by=['Percentage'], inplace=True)
    df_filter_stats.plot.barh(x='Filter', y='Percentage', ax=axs_filter_stats, grid=True)
    axs_filter_stats.set_xlabel('Fraction of Filtered Wikipedia Mentions', fontsize=14, labelpad=6)
    axs_filter_stats.get_legend().remove()
    #

    axs_results_new_vs_shared.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)
    axs_results_train_type_s1.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)
    axs_results_train_type_s2.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)
    axs_results_train_type_s3.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)
    axs_results_train_type_s4.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)
    axs_results_k_type.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)

    y_ticks = np.arange(0.0, 0.040, 0.005)
    y_ticklabels = [str('{:.3f}'.format(yt)) for yt in y_ticks]
    axs_results_k_type_delta.set_yticks(y_ticks)
    axs_results_k_type_delta.set_yticklabels(y_ticklabels)

    axs_results_k_type_delta.set_xlabel('K', fontsize=14, labelpad=5)

    handles, labels = axs_results_k_type_delta.get_legend_handles_labels()
    labels, handles = zip(*reversed(list(zip(labels, handles))))
    axs_results_k_type_delta.legend(handles, labels)
    #
    #
    y_ticks = np.arange(-0.060, 0.020, 0.01)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]

    axs_results_temporal_offset.set_yticks(y_ticks)
    axs_results_temporal_offset.set_yticklabels(y_ticklabels)

    axs_results_temporal_offset.set_xlabel('Temporal Offset', fontsize=14, labelpad=5)

    x_ticks_offset = np.arange(0, 20, 2)
    x_ticks_offset2 = np.arange(-9, 10, 2)
    x_ticklabels = [str('{:0.0f}'.format(xt)) for xt in x_ticks_offset2]

    axs_results_temporal_offset.set_xticks(x_ticks_offset)
    axs_results_temporal_offset.set_xticklabels(x_ticklabels, rotation=0)

    #
    axs_results_train_type_delta.set_xlabel('Temporal Snapshot', fontsize=14, labelpad=8)
    y_ticks = np.arange(-0.01, 0.06, 0.01)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]
    axs_results_train_type_delta.set_yticks(y_ticks)
    axs_results_train_type_delta.set_yticklabels(y_ticklabels)

    #
    axs_nr_wiki_candidates.set_xticklabels(x_ticks, rotation=45)
    axs_nr_wiki_candidates.get_legend().remove()
    #

    axs_ent_jaccard_similarity.set_xticklabels(x_ticks, rotation=45)
    axs_ent_jaccard_similarity.set_xticks(x_ticks)
    axs_ent_jaccard_similarity.get_legend().remove()

    axs_men_jaccard_similarity.set_xticklabels(x_ticks, rotation=45)
    axs_men_jaccard_similarity.set_xticks(x_ticks)
    axs_men_jaccard_similarity.get_legend().remove()

    axs_cand_similarity.set_xticklabels(x_ticks, rotation=45)
    axs_cand_similarity.set_xticks(x_ticks)
    y_ticks = np.arange(0.163, 0.178, 0.002)
    y_ticklabels = [str('{:.3f}'.format(yt)) for yt in y_ticks]
    axs_cand_similarity.set_yticks(y_ticks)
    axs_cand_similarity.set_yticklabels(y_ticklabels)
    axs_cand_similarity.get_legend().remove()

    axs_ent_ed_dist_similarity.set_xticklabels(x_ticks, rotation=45)
    axs_ent_ed_dist_similarity.set_xticks(x_ticks)
    y_ticks = np.arange(0.0, 0.51, 0.1)
    y_ticklabels = [str('{:.1f}'.format(yt)) for yt in y_ticks]
    axs_ent_ed_dist_similarity.set_yticks(y_ticks)
    axs_ent_ed_dist_similarity.set_yticklabels(y_ticklabels)
    axs_ent_ed_dist_similarity.get_legend().remove()

    y_ticks = np.arange(0.6, 1.01, 0.1)
    y_ticklabels = [str('{:.1f}'.format(yt)) for yt in y_ticks]
    axs_ent_jaccard_similarity.set_yticks(y_ticks)
    axs_ent_jaccard_similarity.set_yticklabels(y_ticklabels)
    # axs_ent_jaccard_similarity.get_legend().remove()

    axs_results_new_vs_shared.set_xticklabels(x_ticks, rotation=45)
    axs_results_new_vs_shared.set_xticks(x_ticks)
    # y_ticks = np.arange(0.65, 0.95, 0.03)
    y_ticks = np.arange(0.55, 0.95, 0.03)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]
    axs_results_new_vs_shared.set_yticks(y_ticks)
    axs_results_new_vs_shared.set_yticklabels(y_ticklabels)
    axs_results_new_vs_shared.legend(title='')
    ##
    axs_results_train_type_delta.set_xticklabels(x_ticks, rotation=45)
    #######
    handles, labels = axs_results_train_type_delta.get_legend_handles_labels()
    labels, handles = zip(*reversed(list(zip(labels, handles))))
    axs_results_train_type_delta.legend(handles, labels)

    #######

    # s1
    axs_results_train_type_s1.set_xticklabels(x_ticks, rotation=45)
    axs_results_train_type_s1.set_xticks(x_ticks)
    y_ticks = np.arange(0.60, 0.96, 0.05)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]
    axs_results_train_type_s1.set_yticks(y_ticks)
    axs_results_train_type_s1.set_yticklabels(y_ticklabels)
    axs_results_train_type_s1.legend(title='', loc='lower left')
    # s2
    axs_results_train_type_s2.set_xticklabels(x_ticks, rotation=45)
    axs_results_train_type_s2.set_xticks(x_ticks)
    y_ticks = np.arange(0.60, 0.96, 0.05)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]
    axs_results_train_type_s2.set_yticks(y_ticks)
    axs_results_train_type_s2.set_yticklabels(y_ticklabels)
    axs_results_train_type_s2.legend(title='', loc='lower left')
    # s3
    axs_results_train_type_s3.set_xticklabels(x_ticks, rotation=45)
    axs_results_train_type_s3.set_xticks(x_ticks)
    y_ticks = np.arange(0.60, 0.96, 0.05)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]
    axs_results_train_type_s3.set_yticks(y_ticks)
    axs_results_train_type_s3.set_yticklabels(y_ticklabels)
    axs_results_train_type_s3.legend(title='', loc='lower left')
    # s4
    axs_results_train_type_s4.set_xticklabels(x_ticks, rotation=45)
    axs_results_train_type_s4.set_xticks(x_ticks)
    y_ticks = np.arange(0.60, 0.96, 0.05)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]
    axs_results_train_type_s4.set_yticks(y_ticks)
    axs_results_train_type_s4.set_yticklabels(y_ticklabels)
    axs_results_train_type_s4.legend(title='', loc='lower left')

    #
    y_ticks = np.arange(0.20, 0.95, 0.1)
    y_ticklabels = [str('{:.2f}'.format(yt)) for yt in y_ticks]
    axs_results_k_type.set_yticks(y_ticks)
    axs_results_k_type.set_yticklabels(y_ticklabels)
    axs_results_k_type.legend(title='')
    #
    y_ticks = np.arange(3.5, 6.6, 0.5)
    y_ticklabels = [str('{:.1f}'.format(yt)) for yt in y_ticks]
    #
    axs_nr_wiki_candidates.set_yticks(y_ticks)
    axs_nr_wiki_candidates.set_yticklabels(y_ticklabels)
    #
    fig_plt_ent_jaccard_similarity.subplots_adjust(bottom=0.13, left=0.17)
    fig_plt_men_jaccard_similarity.subplots_adjust(bottom=0.13, left=0.17)
    fig_plt_ent_ed_dist_similarity.subplots_adjust(bottom=0.13, left=0.20)

    fig_plt_filter_stats.subplots_adjust(left=0.40, bottom=0.14)

    fig_plt_cand_similarity.subplots_adjust(bottom=0.13, left=0.22)
    fig_plt_results_new_vs_shared.subplots_adjust(bottom=0.13, left=0.19)
    #
    fig_plt_results_train_type_s1.subplots_adjust(bottom=0.13, left=0.19)
    fig_plt_results_train_type_s2.subplots_adjust(bottom=0.13, left=0.19)
    fig_plt_results_train_type_s3.subplots_adjust(bottom=0.13, left=0.19)
    fig_plt_results_train_type_s4.subplots_adjust(bottom=0.13, left=0.19)
    fig_plt_results_train_type_delta.subplots_adjust(bottom=0.13, left=0.23, top=0.96, right=0.96)
    #
    fig_plt_results_new2_acc_k.subplots_adjust(bottom=0.13, left=0.19)

    fig_plt_results_new_type_s1.subplots_adjust(bottom=0.13, left=0.19, top=0.96, right=0.96)
    fig_plt_results_new_type_s2.subplots_adjust(bottom=0.13, left=0.19, top=0.96, right=0.96)
    fig_plt_results_new_type_s3.subplots_adjust(bottom=0.13, left=0.19, top=0.96, right=0.96)
    fig_plt_results_new_type_s4.subplots_adjust(bottom=0.13, left=0.19, top=0.96, right=0.96)
    fig_plt_nr_wiki_candidates.subplots_adjust(bottom=0.13, left=0.19)

    fig_results_temporal_offset.subplots_adjust(bottom=0.13, left=0.21, right=0.98, top=0.96)

    fig_plt_results_acc_k_all.subplots_adjust(bottom=0.09, left=0.12, right=0.98, top=0.98)

    ######### BEGIN mentions per entity statistics

    fig_plt, axs_plts = plt.subplots(2, 3, figsize=(15, 8))

    cat_id_to_cat_name = {'shared': 'New Entities',
                          'new_entities': 'Continual Entities'}

    subset_id_to_subset_name = {'train': 'Train',
                                'test': 'Test',
                                'validation': 'Validation'}
    dist_categories = ['shared', 'new_entities']
    dist_subsets = ['train', 'validation', 'test']
    for idx_category, curr_category in enumerate(dist_categories):
        for idx_subset, curr_subset in enumerate(dist_subsets):
            df_to_plot_curr = df_mentions_per_entities.loc[((df_mentions_per_entities['category'] == curr_category) &
                                                            (df_mentions_per_entities['subset'] == curr_subset)
                                                            ), ['cumsum', 'nr_mentions', 'cut']]
            df_pivotet = df_to_plot_curr.pivot_table('cumsum', ['nr_mentions'], 'cut').reset_index()

            df_pivotet.fillna(method='ffill', inplace=True)
            for curr_column in df_pivotet.columns:
                if curr_column != 'nr_mentions':
                    df_pivotet[curr_column] = df_pivotet[curr_column].astype('int')

            path_output_mentions_per_entity = os.path.join(men_per_entity_dir, 'stat_mentions_per_entity_{}_{}.csv'
                                                           .format(curr_subset, curr_category))

            df_pivotet.to_csv(path_output_mentions_per_entity, sep='\t', index=False)

            df_pivotet.plot(ax=axs_plts[idx_category][idx_subset], x='nr_mentions', grid=True)

            min_nr_mentions = 1
            max_nr_mentions = df_to_plot_curr[['nr_mentions']].max().item()
            interval_size = math.ceil(max_nr_mentions / 10)
            x_ticks = list(range(min_nr_mentions, max_nr_mentions, interval_size))
            axs_plts[idx_category][idx_subset].set_ylabel('Nr. of Linked Entities', fontsize=14)
            axs_plts[idx_category][idx_subset].set_xlabel('Nr. of Mentions', fontsize=14)

            axs_plts[idx_category][idx_subset].set_xticks(x_ticks)

            axs_plts[idx_category][idx_subset].yaxis.set_major_locator(MaxNLocator(integer=True))
            axs_plts[idx_category][idx_subset].xaxis.set_major_locator(MaxNLocator(integer=True))
            curr_title = subset_id_to_subset_name[curr_subset] + ': ' + cat_id_to_cat_name[curr_category]
            axs_plts[idx_category][idx_subset].set_title(curr_title, fontsize=14, pad=8)
            axs_plts[idx_category][idx_subset].get_legend().remove()

    handles, labels = axs_plts[1][2].get_legend_handles_labels()
    fig_plt.legend(handles, labels, loc='upper right')

    fig_plt.tight_layout()
    fig_plt.subplots_adjust(wspace=0.3)
    fig_plt.subplots_adjust(hspace=0.3)
    plt.show()
    fig_plt.savefig(output_path_mentions_per_entity, format='pdf')
    ######### END mention per entity statistics

    # plt.show()

    fig_plt_ent_jaccard_similarity.savefig(output_path_jaccard_entities, format='pdf')
    fig_plt_men_jaccard_similarity.savefig(output_path_jaccard_mentions, format='pdf')
    fig_plt_cand_similarity.savefig(output_path_jaccard_cand_similarity, format='pdf')
    fig_plt_ent_ed_dist_similarity.savefig(output_path_edit_distance_entities, format='pdf')
    fig_plt_results_new_vs_shared.savefig(output_path_results_new_vs_shared, format='pdf')
    fig_plt_results_train_type_s1.savefig(output_path_results_train_type_s1, format='pdf')
    fig_plt_results_train_type_s2.savefig(output_path_results_train_type_s2, format='pdf')
    fig_plt_results_train_type_s3.savefig(output_path_results_train_type_s3, format='pdf')
    fig_plt_results_train_type_s4.savefig(output_path_results_train_type_s4, format='pdf')
    fig_plt_results_train_type_delta.savefig(output_path_results_train_type_delta, format='pdf')
    fig_plt_nr_wiki_candidates.savefig(output_path_nr_wiki_candidates, format='pdf')
    fig_plt_results_new2_acc_k.savefig(output_path_results2_new, format='pdf')
    #
    fig_plt_results_new_type_s1.savefig(output_path_results_new_type_s1, format='pdf')
    fig_plt_results_new_type_s2.savefig(output_path_results_new_type_s2, format='pdf')
    fig_plt_results_new_type_s3.savefig(output_path_results_new_type_s3, format='pdf')
    fig_plt_results_new_type_s4.savefig(output_path_results_new_type_s4, format='pdf')
    #
    fig_plt_filter_stats.savefig(output_path_filter_stats, format='pdf')
    fig_plt_results_k_type.savefig(output_path_results_k_type, format='pdf')
    fig_plt_results_k_type_delta.savefig(output_path_results_k_type_delta, format='pdf')
    fig_results_temporal_offset.savefig(output_path_results_temporal_offset, format='pdf')
    fig_plt_results_acc_k_all.savefig(output_path_results_acc_k_all, format='pdf')
