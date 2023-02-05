# The script to create the dataset

import argparse
import json
import logging
import os
import pickle
import random
import time
from datetime import datetime, timedelta
from math import isclose
from multiprocessing import Process, Value
from threading import Thread

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm

from src.tempel_creation.misc.article_queue import ArticleReadingQueue
from src.tempel_creation.misc.s03_final_dataset_creator_utils import item_generator, apply_filters, put_mention_in_cluster, \
    get_evenly_distributed_mentions_all, read_from_input_file, process_line_input_file_parallel, write_to_output_file, \
    display
from src.utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/dataset_creation/data_20220515/config/s03_dataset_creator.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--only_dataset_file_merging', action='store_true',
                        help='If mention and entity files have been already created, then this parameter '
                             'can be passed, so the script only generates the dataset file. ')

    parser.add_argument('--nr_threads', required=False, type=int,
                        default=30,
                        help='Nr of threads that will process the data, ideally should be close to the number of the '
                             'available CPUs.')

    args = parser.parse_args()
    logger.info('getting the final dataset with the following parameters: %s' % args)
    config = json.load(open(args.config_file, 'rt'))

    config['nr_threads'] = args.nr_threads

    # setting the seed to ensure reproducibility
    if config['seed'] != -1:
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])

    import git

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir, 'commit_hash_dataset_creator.txt'), 'wt') as outfile:
        outfile.write(sha)

    only_dataset_file_merging = args.only_dataset_file_merging

    # base_experiment_path = config['base_experiment_path']
    # output_dir = os.path.join(base_experiment_path, config['output_dir'])
    input_dir_stats = config['input_dir_stats']
    # input_dir_stats = os.path.join(base_experiment_path, input_dir_stats)
    input_dir_stats = os.path.join(input_dir_stats)

    prefix_prior_link_stats = config['prefix_prior_link_stats']
    prefix_page_info = config['prefix_page_info']
    min_same_surface_form_mentions_to_entity = config['common_filters']['min_same_surface_form_mentions_to_entity']
    # input_pages_with_redirects_path = os.path.join(base_experiment_path, config['input_pages_with_redirects_path'])
    input_pages_with_redirects_path = os.path.join(config['input_pages_with_redirects_path'])

    prefix_dict_inverse_index = config['prefix_dict_inverse_index']
    debug = config['debug']

    time_cuts = config['time_cuts']

    max_len_context_tokenized = config['max_len_context_tokenized']
    #########
    # loads the mentions with the pages they appear in
    #########
    all_subset_names = [curr_subset['name'] for curr_category in config['categories']
                        for curr_subset in curr_category['subsets']]
    all_subset_names = set(all_subset_names)

    cat_to_subset_info = dict()
    for curr_category in config['categories']:
        cat_to_subset_info[curr_category['name']] = dict()
        for curr_subset in curr_category['subsets']:
            cat_to_subset_info[curr_category['name']][curr_subset['name']] = curr_subset

    if not only_dataset_file_merging:
        output_data_paths = dict()
        df_pages_with_redirects = pd.read_csv(open(input_pages_with_redirects_path, 'rt'), sep=',', header=[0])
        wikidata_qids_with_redirects_in_history = set(df_pages_with_redirects['wikidata_qid'])
        entities_per_subset = dict()  # subset --> entities ; in order to avoid overlaps
        all_category_types = [curr_category['name'] for curr_category in config['categories']]
        all_category_types = set(all_category_types)

        for curr_subset_name in all_subset_names:
            entities_per_subset[curr_subset_name] = set()

        sub_types = item_generator(config, 'sub_types')

        enforce_shared_target_entities_across_cuts = item_generator(config,
                                                                    'enforce_shared_target_entities_across_cuts')
        if any(enforce_shared_target_entities_across_cuts):
            enforce_shared_target_entities = True
        else:
            enforce_shared_target_entities = False

        df_common_filtered_target_qids = None
        # date cut to df of prior stats shared
        filtered_anchor_target_stats = dict()

        df_page_infos_per_cut = dict()
        df_target_qid_stats_per_cut = dict()  # TODO -we are here
        logger.info('loading necessary dataframes from .csv and filtering for each of the time_cuts')
        start_loading = time.time()
        process = psutil.Process(os.getpid())

        for curr_time_cut in time_cuts:
            logger.info('00 memory usage (MB) before %s: %s' % (curr_time_cut, process.memory_info().rss / 1024 / 1024))
            logger.info('01 memory usage (MB): %s' % (process.memory_info().rss / 1024 / 1024))
            curr_page_info_path = '{}{}.csv'.format(prefix_page_info, curr_time_cut)
            curr_page_info_path = os.path.join(input_dir_stats, curr_page_info_path)

            df_page_info = pd.read_csv(open(curr_page_info_path, 'rt'), sep='\t', header=[0])

            #
            df_page_info['wikipedia_creation_time'] = \
                pd.to_datetime(df_page_info['wikipedia_creation_time'], format='%Y-%m-%dT%H:%M:%SZ')

            df_page_infos_per_cut[curr_time_cut] = df_page_info
            logger.info('02 memory usage (MB): %s' % (process.memory_info().rss / 1024 / 1024))

            parsed_curr_date = datetime.strptime(curr_time_cut, '%Y-%m-%dT%H:%M:%SZ')
            curr_year = parsed_curr_date.year
            start = time.time()
            curr_prior_stats_path = '{}{}.csv'.format(prefix_prior_link_stats, curr_time_cut)
            curr_prior_stats_path = os.path.join(input_dir_stats, curr_prior_stats_path)
            df_prior_stats = pd.read_csv(open(curr_prior_stats_path, 'rt'), sep='\t', header=[0])
            logger.info('03 memory usage (MB): %s' % (process.memory_info().rss / 1024 / 1024))

            logger.info('BEFORE filtering out List_of the df_prior_stats.shape is %s' % str(df_prior_stats.shape))
            df_prior_stats = df_prior_stats[
                df_prior_stats['target_wikipedia_title_orig'].str.contains('^List_of_', regex=True) == False]
            logger.info('AFTER filtering out List_of the df_prior_stats.shape is %s' % str(df_prior_stats.shape))

            df_prior_stats.rename(
                columns={'nr_links': 'nr_links_mention_per_entity'}, inplace=True)

            if config['debug']:
                bug_list_of = df_prior_stats[
                    df_prior_stats['target_wikipedia_title_orig'].str.contains('^List_of_', regex=True)]

                lst_target_list_of = bug_list_of['target_wikipedia_title_orig'].tolist()
                logger.info('---BEGIN: List of targets of type "List_of_..."')
                for curr_target in lst_target_list_of:
                    logger.info(curr_target)
                logger.info('---END: List of targets of type "List_of_..."')

                logger.info('shape of List_of_...: %s' % str(bug_list_of.shape))
                logger.info('---')

            logger.info('04 memory usage (MB): %s' % (process.memory_info().rss / 1024 / 1024))

            df_prior_stats_filtered = apply_filters(config['common_filters'], df_prior_stats, 'generic common filter',
                                                    wikidata_qids_with_redirects_in_history, debug=config['debug'])

            filtered_anchor_target_stats[curr_time_cut] = df_prior_stats_filtered
            # it is not the same as inlinks, since this one is based only on filtered mentions
            # columns df_prior_stats_filtered:
            #
            filtered_targets_w_nr_inlinks: pd.DataFrame = df_prior_stats_filtered[
                ['target_qid', 'nr_links_mention_per_entity']] \
                .groupby('target_qid').agg({'nr_links_mention_per_entity': 'sum'})

            df_distinct_mentions_per_target = df_prior_stats_filtered.groupby('target_qid') \
                .anchor_mention_text.nunique().reset_index()

            col_dist_mentions = 'nr_distinct_mentions'
            df_distinct_mentions_per_target.rename(
                columns={'anchor_mention_text': col_dist_mentions}, inplace=True)

            col_nr_inlinks = 'nr_inlinks_after_filter'
            filtered_targets_w_nr_inlinks.rename(columns={'nr_links_mention_per_entity': col_nr_inlinks}, inplace=True)

            min_nr_distinct_mentions_per_entity = config['common_filters']['min_nr_distinct_mentions_per_entity']

            # this min_nr_distinct_mentions_per_entity might not be necessary
            df_distinct_mentions_per_target = \
                df_distinct_mentions_per_target[df_distinct_mentions_per_target[col_dist_mentions] >=
                                                min_nr_distinct_mentions_per_entity]

            # is this drop_duplicates necessary?
            df_distinct_mentions_per_target = df_distinct_mentions_per_target.drop_duplicates()

            # can this merge produce cartesian stuff? CHECK WHAT THE PRINT BELOW OUTPUTS!
            logger.info('%s is the dimension of filtered_targets... BEFORE joining with '
                        'df_distinct_mentions_per_target' % str(filtered_targets_w_nr_inlinks.shape))
            filtered_targets_w_nr_inlinks = pd.merge(filtered_targets_w_nr_inlinks, df_distinct_mentions_per_target,
                                                     on=('target_qid',), how='inner')
            logger.info('%s is the dimension of filtered_targets... AFTER joining with df_distinct_mentions_per_target'
                        % str(filtered_targets_w_nr_inlinks.shape))

            df_target_qid_stats_per_cut[curr_time_cut] = filtered_targets_w_nr_inlinks

            if df_common_filtered_target_qids is None:
                df_common_filtered_target_qids = filtered_targets_w_nr_inlinks[['target_qid']].copy()
            else:
                df_common_filtered_target_qids = pd.merge(df_common_filtered_target_qids,
                                                          filtered_targets_w_nr_inlinks[['target_qid']],
                                                          on=('target_qid',), how='inner')

            logger.info('%s : df_common_filtered_targets.shape after %s ' %
                        (str(df_common_filtered_target_qids.shape), curr_time_cut))
            logger.info('05 memory usage (MB): %s' % (process.memory_info().rss / 1024 / 1024))

        logger.info('%s is the final df_common_filtered_targets.shape' % str(df_common_filtered_target_qids.shape))

        end_loading = time.time()
        logger.info('%s minutes for loading necessary dataframes from .csv and filtering for each of the time_cuts' %
                    str((end_loading - start_loading) / 60))

        # now that we have common ids, we leave in each of the stats the rows of these common ids
        start_shared_merge = time.time()
        filtered_shared_anchor_target_stats = dict()
        for curr_time_cut in filtered_anchor_target_stats.keys():
            parsed_curr_date = datetime.strptime(curr_time_cut, '%Y-%m-%dT%H:%M:%SZ')
            curr_year = parsed_curr_date.year
            filtered_anchor_target_stats[curr_time_cut] = pd.merge(filtered_anchor_target_stats[curr_time_cut],
                                                                   df_target_qid_stats_per_cut[curr_time_cut],
                                                                   on=('target_qid',), how='inner')
        end_shared_merge = time.time()
        logger.info('%s minutes for loading for performing shared merge to getshared entities across all the cuts' %
                    ((end_shared_merge - start_shared_merge) / 60))

        # BEGIN (characteristic of this _v2 version) and now we concatenate across all the cuts in a SINGLE DATAFRAME,
        # this has to be done do later distribute the mentions so there is no overlap between subsets
        # "ACROSS ALL THE CUTS" (see implementation notes).
        to_concat = list(filtered_anchor_target_stats.values())
        df_anchor_and_target_stats_across_cuts = pd.concat(to_concat)
        # END (characteristic of this _v2 version)

        nr_entities = 0
        dict1_target_pages_to_search = dict()
        dict2_anchor_pages_to_search = dict()

        arq = ArticleReadingQueue(article_queue_size=config['queues']['article_queue_size'],
                                  process_file_queue_size=config['queues']['process_file_queue_size'])
        input_file_path = config['input_file_path']
        # input_file_path = os.path.join(base_experiment_path, input_file_path)

        already_processed_target_qids_per_category = dict()

        mentions_to_process_per_category = dict()
        for curr_category in config['categories']:
            if curr_category['name'] not in already_processed_target_qids_per_category:
                already_processed_target_qids_per_category[curr_category['name']] = set()
            if curr_category['name'] == 'continual':
                #### BEGIN first filters out the target entities that were already processed in other subset or category
                df_shared_target_qids_not_processed = df_common_filtered_target_qids
                for curr_cat in already_processed_target_qids_per_category.keys():
                    if curr_cat == curr_category['name']:
                        continue
                    df_shared_target_qids_not_processed = df_shared_target_qids_not_processed[
                        df_shared_target_qids_not_processed.apply
                        (lambda x: x['target_qid'] not in
                                   already_processed_target_qids_per_category[curr_cat], axis=1)]

                logger.info('%s: df_common_filtered_targets.shape VS. %s: '
                            'df_common_filtered_targets_not_processed.shape' %
                            (str(df_common_filtered_target_qids.shape), str(df_shared_target_qids_not_processed.shape)))

                #### END first filters out the target entities that were already processed in other subset or category

                df_entities_to_process_w_mentions = pd.merge(
                    df_shared_target_qids_not_processed[['target_qid']],
                    df_anchor_and_target_stats_across_cuts,
                    on=('target_qid',), how='inner')

                # TODO - check if this sort_values is really necessary
                # creates groups of target pages, each of which is sorted by number of mentions:
                # I guess nr_links_mention_per_entity should give less popular mentions first for a particular entity
                #
                df_entities_to_process_w_mentions = df_entities_to_process_w_mentions.sort_values(
                    ['target_qid', 'filtered_date', 'nr_links_mention_per_entity'],
                    ascending=[True, True, True])

                groups = [df for _, df in df_entities_to_process_w_mentions.groupby(by=['target_qid'])]
                random.shuffle(groups)
                #
                df_entities_to_process_w_mentions = pd.concat(groups).reset_index(drop=True)
                # adding subset_name and bucket_name columns
                df_entities_to_process_w_mentions['subset_name'] = ''
                prev_target_wikidata_qid = None
                prev_target_wikipedia_title_orig = None
                subset_clusters_dict = {curr_sbset_name: set() for curr_sbset_name in all_subset_names}
                distinct_mentions = set()
                for curr_tuple in df_entities_to_process_w_mentions.itertuples():
                    curr_target_wikidata_qid = curr_tuple.target_wikidata_qid
                    curr_target_wikipedia_title_orig = curr_tuple.target_wikipedia_title_orig
                    if prev_target_wikidata_qid is None or prev_target_wikidata_qid != curr_target_wikidata_qid:
                        prev_target_wikipedia_title_orig = curr_tuple.target_wikipedia_title_orig
                        prev_target_wikidata_qid = curr_target_wikidata_qid
                        distinct_mentions = set()
                        subset_clusters_dict = {curr_sbset_name: set() for curr_sbset_name in all_subset_names}
                    mention = curr_tuple.anchor_mention_text
                    subset_clusters_dict, assigned_subset = \
                        put_mention_in_cluster(subset_clusters_dict=subset_clusters_dict,
                                               config=config, mention=mention)
                    distinct_mentions.add(mention)
                    df_entities_to_process_w_mentions.at[curr_tuple.Index, 'subset_name'] = assigned_subset

                df_nr_distinct_subsets = df_entities_to_process_w_mentions \
                    .groupby(['target_qid', 'filtered_date']).subset_name.nunique().reset_index()
                df_nr_distinct_subsets.rename(columns={'subset_name': 'nr_distinct_subsets'}, inplace=True)

                ####### BEGIN WIP min nr distinct subsets ACROSS TARGET AND CUT
                # TODO - here min nr distinct subsets across target and cut!
                df_min_nr_distinct_subsets = df_nr_distinct_subsets \
                    .groupby(['target_qid']) \
                    .nr_distinct_subsets.min().reset_index()
                ####### END WIP
                #
                logger.info('%s the df_entities_to_process_w_mentions.shape BEFORE applying filter on '
                            'nr_distinct_subsets  for SHARED' % str(df_entities_to_process_w_mentions.shape))
                df_min_nr_distinct_subsets = df_min_nr_distinct_subsets[
                    df_min_nr_distinct_subsets['nr_distinct_subsets'] == len(all_subset_names)]
                df_min_nr_distinct_subsets = df_min_nr_distinct_subsets[['target_qid']].drop_duplicates()
                df_entities_to_process_w_mentions = pd.merge(df_entities_to_process_w_mentions,
                                                             df_min_nr_distinct_subsets, on=('target_qid',),
                                                             how='inner')
                logger.info('%s the df_entities_to_process_w_mentions.shape AFTER applying filter on '
                            'nr_distinct_subsets  for SHARED' % str(df_entities_to_process_w_mentions.shape))
                # step 02 - similarly to step 01 calculate min_nr_mentions for each of the subsets, across the cuts
                # this will allow to assign a particular target_wikidata_qid to a specific bucket
                # at this point the columns of df_entities_to_process_w_mentions are:
                # Index(['target_qid', 'anchor_mention_text',
                #        'target_wikipedia_title_orig', 'target_page_id', 'filtered_date',
                #        'count_mentions', 'nr_links_mention_per_entity', 'prior',
                #        'target_title_2022', 'prior_rank', 'target_wikipedia_title_orig_lower',
                #        'edit_distance_mention_title', 'ratio_edit_distance_mention_title',
                #        'avg_prior_rank', 'avg_prior', 'nr_inlinks', 'nr_inlinks_after_filter',
                #        'nr_distinct_mentions', 'subset_name', 'bucket_name'],
                #       dtype='object')

                df_nr_mentions_per_subset_per_cut = df_entities_to_process_w_mentions \
                    .groupby(['target_qid', 'filtered_date', 'subset_name']) \
                    .nr_links_mention_per_entity.sum().reset_index()
                df_nr_mentions_per_subset_per_cut.rename(
                    columns={'nr_links_mention_per_entity': 'nr_links_per_cut_per_sset'}, inplace=True)
                df_entities_to_process_w_mentions = pd.merge(df_entities_to_process_w_mentions,
                                                             df_nr_mentions_per_subset_per_cut,
                                                             on=('target_qid', 'filtered_date', 'subset_name'),
                                                             how='inner')
                df_min_nr_links_per_subset = df_nr_mentions_per_subset_per_cut \
                    .groupby(['target_qid', 'subset_name']) \
                    .nr_links_per_cut_per_sset.min().reset_index()
                df_min_nr_links_per_subset.rename(
                    columns={'nr_links_per_cut_per_sset': 'min_nr_links_per_subset'}, inplace=True)
                df_entities_to_process_w_mentions = pd.merge(df_entities_to_process_w_mentions,
                                                             df_min_nr_links_per_subset,
                                                             on=('target_qid', 'subset_name'),
                                                             how='inner')
                # at this point the columns of df_entities_to_process_w_mentions are:
                #
                # Index(['target_qid', 'anchor_mention_text',
                #        'target_wikipedia_title_orig', 'target_page_id', 'filtered_date',
                #        'count_mentions', 'nr_links_mention_per_entity', 'prior',
                #        'target_title_2022', 'prior_rank', 'target_wikipedia_title_orig_lower',
                #        'edit_distance_mention_title', 'ratio_edit_distance_mention_title',
                #        'avg_prior_rank', 'avg_prior', 'nr_inlinks', 'nr_inlinks_after_filter',
                #        'nr_distinct_mentions', 'subset_name', 'bucket_name',
                #        'min_nr_links_per_subset'],
                #       dtype='object')
                #

                # step 03 - similarly to steps 01 and 02, calculate the min_nr_distinct_mentions for each of the subsets,
                # across the cuts. Optionally, this will allow to focus on target_wikidata_qid with mentions that have
                # some diversity in surface form. THIS STEP IS OPTIONAL.
                # =========
                # here just decided to calculate the nr of distinct mentions for each of the subsets for each of the cut
                # for each of the target wikidata qid. Later can be extended to calculate min_nr_distinct_mentions
                # as required above.
                # =========
                df_dist_men = df_entities_to_process_w_mentions.groupby(['target_qid', 'filtered_date',
                                                                         'subset_name']) \
                    .anchor_mention_text.nunique().reset_index()
                col_dist_mentions = 'nr_dist_men_per_cut_per_sset'
                df_dist_men.rename(columns={'anchor_mention_text': col_dist_mentions}, inplace=True)

                df_entities_to_process_w_mentions = pd.merge(df_entities_to_process_w_mentions, df_dist_men,
                                                             on=('target_qid', 'filtered_date',
                                                                 'subset_name'),
                                                             how='inner')

                cols_to_select_men = df_entities_to_process_w_mentions.columns.values.tolist()
                category_filter = curr_category['category_filter']
                #
                max_entities = category_filter['max_entities']

                df_entities_to_process_w_mentions = pd.merge(left=df_entities_to_process_w_mentions,
                                                             right=df_shared_target_qids_not_processed,
                                                             on='target_qid', how='inner')

                cols_to_select = df_shared_target_qids_not_processed.columns.values.tolist()
                df_shared_target_qids_not_processed = df_entities_to_process_w_mentions[cols_to_select] \
                    .drop_duplicates()

                if max_entities < df_shared_target_qids_not_processed.shape[0]:
                    df_shared_target_qids_not_processed = df_shared_target_qids_not_processed.sample(n=max_entities)

                logger.info('%s df_entities_to_process_w_mentions.shape BEFORE filtering on max_entities ' %
                            str(df_entities_to_process_w_mentions.shape))
                df_entities_to_process_w_mentions = pd.merge(left=df_entities_to_process_w_mentions,
                                                             right=df_shared_target_qids_not_processed,
                                                             on='target_qid', how='inner')[cols_to_select_men]
                logger.info('%s df_entities_to_process_w_mentions.shape AFTER filtering on max_entities '
                            % str(df_entities_to_process_w_mentions.shape))

                logger.info('%s shape df_shared_targets_not_processed' % str(df_shared_target_qids_not_processed.shape))
                logger.info('nr of distinct entities in df_entities_to_process_w_mentions (shared): %s' %
                            len(df_entities_to_process_w_mentions['target_qid'].unique()))

                ############################## BEGIN V3 algorithm
                logger.info('V3 columns of df_entities_to_process_w_mentions: %s' %
                            str(df_entities_to_process_w_mentions.columns))
                # columns of df_entities_to_process_w_mentions:
                # Index(['target_qid', 'anchor_mention_text',
                #        'target_wikipedia_title_orig', 'target_page_id', 'filtered_date',
                #        'count_mentions', 'nr_links_mention_per_entity', 'prior',
                #        'target_title_2022', 'prior_rank', 'target_wikipedia_title_orig_lower',
                #        'edit_distance_mention_title', 'edit_distance_mention_title_ent',
                #        'ratio_edit_distance_mention_title',
                #        'ratio_edit_distance_mention_title_ent', 'overlap_type',
                #        'avg_prior_rank', 'avg_prior', 'nr_inlinks', 'nr_inlinks_after_filter',
                #        'nr_distinct_mentions', 'subset_name', 'bucket_name',
                #        'nr_links_per_cut_per_sset', 'min_nr_links_per_subset',
                #        'nr_dist_men_per_cut_per_sset'],
                #       dtype='object')
                # # nr links per cut per sset
                #
                df_entities_to_proc = df_entities_to_process_w_mentions \
                    .groupby(['target_qid', 'filtered_date', 'subset_name']) \
                    .nr_links_per_cut_per_sset.mean().reset_index()

                df_entities_to_process_adjusted = None

                df_entities_to_proc['nr_links_per_cut_per_sset'] = \
                    df_entities_to_proc['nr_links_per_cut_per_sset'].astype(int)
                for curr_subset_name in all_subset_names:
                    ent_mentions_by_subset_by_cut = dict()

                    min_nr_mentions_in_cut = -1
                    min_nr_mentions_cut = None
                    for curr_time_cut in config['time_cuts']:
                        parsed_curr_date = datetime.strptime(curr_time_cut, '%Y-%m-%dT%H:%M:%SZ')
                        curr_year = parsed_curr_date.year
                        logger.info('v3 processing curr_time_cut: %s' % curr_time_cut)
                        ent_mentions_by_subset_by_cut[curr_time_cut] = \
                            df_entities_to_proc[(df_entities_to_proc['filtered_date'] == curr_time_cut) &
                                                (df_entities_to_proc['subset_name'] == curr_subset_name)]
                        logger.info('ent_mentions_by_subset_by_cut[curr_time_cut].shape: %s' %
                                    str(ent_mentions_by_subset_by_cut[curr_time_cut].shape))
                        col_nr_links_name = 'nr_links_per_{}_per_{}'.format(curr_year, curr_subset_name)
                        ent_mentions_by_subset_by_cut[curr_time_cut] = \
                            ent_mentions_by_subset_by_cut[curr_time_cut] \
                                .rename(columns={'nr_links_per_cut_per_sset': col_nr_links_name,
                                                 'target_qid': 'target_wikidata_qid_{}'.format(curr_year)})

                        # sorts by nr links, doesn't matter if ascending is True or False
                        ent_mentions_by_subset_by_cut[curr_time_cut] = \
                            ent_mentions_by_subset_by_cut[curr_time_cut]. \
                                sort_values([col_nr_links_name], ascending=[False])
                        #
                        ent_mentions_by_subset_by_cut[curr_time_cut] \
                            .insert(0, 'new_id', range(0, 0 + ent_mentions_by_subset_by_cut[curr_time_cut].shape[0]))

                        max_mentions_per_entity = \
                            cat_to_subset_info[curr_category['name']][curr_subset_name]['max_mentions_per_entity']

                        logger.info('BEFORE max_mentions_per_entity for cut %s for SHARED for subset %s : %s'
                                    % (curr_time_cut, curr_subset_name, ent_mentions_by_subset_by_cut[curr_time_cut] \
                            [col_nr_links_name].sum()))

                        if max_mentions_per_entity > 0:
                            ent_mentions_by_subset_by_cut[curr_time_cut] \
                                .loc[ent_mentions_by_subset_by_cut[curr_time_cut][col_nr_links_name] > \
                                     max_mentions_per_entity, col_nr_links_name] = max_mentions_per_entity

                        logger.info('AFTER max_mentions_per_entity for cut %s for SHARED for subset %s: %s' %
                                    (curr_time_cut, curr_subset_name, ent_mentions_by_subset_by_cut[curr_time_cut] \
                                        [col_nr_links_name].sum()))

                        # cat_to_subset_info
                        nr_elements_sset_cut = ent_mentions_by_subset_by_cut[curr_time_cut] \
                            [col_nr_links_name].sum()
                        if min_nr_mentions_in_cut == -1:
                            min_nr_mentions_in_cut = nr_elements_sset_cut
                            min_nr_mentions_cut = curr_time_cut
                        elif nr_elements_sset_cut < min_nr_mentions_in_cut:
                            min_nr_mentions_in_cut = nr_elements_sset_cut
                            min_nr_mentions_cut = curr_time_cut

                    logger.info('min_nr_mentions_cut for SHARED for subset %s: '
                                '%s for cut %s' % (curr_subset_name, min_nr_mentions_in_cut, min_nr_mentions_cut))

                    parsed_curr_date = datetime.strptime(min_nr_mentions_cut, '%Y-%m-%dT%H:%M:%SZ')
                    min_year = parsed_curr_date.year
                    min_nr_links_name = 'nr_links_per_{}_per_{}'.format(min_year, curr_subset_name)
                    min_target_qid_name = 'target_wikidata_qid_{}'.format(min_year)
                    df_joined = ent_mentions_by_subset_by_cut[min_nr_mentions_cut]
                    df_joined = df_joined.loc[:, ['new_id', min_target_qid_name, min_nr_links_name]]

                    for curr_time_cut in config['time_cuts']:
                        parsed_curr_date = datetime.strptime(curr_time_cut, '%Y-%m-%dT%H:%M:%SZ')
                        curr_year = parsed_curr_date.year
                        col_nr_links_name = 'nr_links_per_{}_per_{}'.format(curr_year, curr_subset_name)
                        col_target_wdata_qid = 'target_wikidata_qid_{}'.format(curr_year)
                        if curr_time_cut != min_nr_mentions_cut:
                            right_join = ent_mentions_by_subset_by_cut[curr_time_cut] \
                                [['new_id', col_target_wdata_qid, col_nr_links_name]]

                            df_joined = pd.merge(left=df_joined, right=right_join, on='new_id')
                            diff_min_col_name = 'diff_min_{}'.format(curr_year)
                            df_joined[diff_min_col_name] = \
                                df_joined[col_nr_links_name] - df_joined[min_nr_links_name]

                            # https://kanoki.org/2019/07/17/pandas-how-to-replace-values-based-on-conditions/
                            df_joined.loc[(df_joined[diff_min_col_name] < 0), diff_min_col_name] = 0

                            # builds a list based on index to pick randomly from, indices are repeated diff times
                            # so the chance is higher for mentions with higher diff.
                            lst_pick_df_idx_from = []
                            for curr_tpl in df_joined.itertuples():
                                # print('curr_tpl is: ', curr_tpl)
                                curr_idx = curr_tpl.Index
                                curr_diff = getattr(curr_tpl, diff_min_col_name)
                                if curr_diff > 0:
                                    curr_idx_lst = [curr_idx] * curr_diff
                                    lst_pick_df_idx_from += curr_idx_lst

                            tot_curr_cut_nr_mentions = df_joined[col_nr_links_name].sum()

                            nr_mentions_to_reduce = tot_curr_cut_nr_mentions - min_nr_mentions_in_cut

                            logger.info('!!REDUCTION TO BALANCE SHARED!! subset name: %s'
                                        ' timestamp snapshot: %s tot_curr_cut_nr_mentions: %s'
                                        ' nr_mentions_to_reduce: %s min_nr_mentions_in_cut: %s' %
                                        (curr_subset_name, curr_time_cut, tot_curr_cut_nr_mentions,
                                         nr_mentions_to_reduce, min_nr_mentions_in_cut))

                            random.shuffle(lst_pick_df_idx_from)

                            lst_idxs_to_reduce = random.sample(lst_pick_df_idx_from, nr_mentions_to_reduce)

                            for curr_idx_to_reduce in lst_idxs_to_reduce:
                                df_joined.at[curr_idx_to_reduce, col_nr_links_name] = \
                                    df_joined.at[curr_idx_to_reduce, col_nr_links_name] - 1

                        df_to_cat = df_joined.loc[:, [col_target_wdata_qid, col_nr_links_name]]
                        df_to_cat.loc[:, 'subset_name'] = curr_subset_name
                        df_to_cat.loc[:, 'filtered_date'] = curr_time_cut
                        df_to_cat = df_to_cat.rename(columns={col_target_wdata_qid: 'target_qid',
                                                              col_nr_links_name: 'nr_links_per_cut_per_sset'})
                        # check that the nr of mentions to extract equals to the nr of mentions in the cut with less
                        # mentions
                        assert df_to_cat['nr_links_per_cut_per_sset'].sum() == min_nr_mentions_in_cut
                        if df_entities_to_process_adjusted is None:
                            df_entities_to_process_adjusted = df_to_cat.copy()
                        else:
                            df_entities_to_process_adjusted = pd.concat([df_entities_to_process_adjusted, df_to_cat])

                df_entities_to_process_adjusted = df_entities_to_process_adjusted.rename(
                    columns={'nr_links_per_cut_per_sset': 'nr_mentions_to_extract_per_cut_per_sset'})

                # controls that after the adjustments, at least one mentions of each entity has to be extracted
                assert (df_entities_to_process_adjusted['nr_mentions_to_extract_per_cut_per_sset'] > 0).all()

                df_entities_to_proc = pd.merge(
                    left=df_entities_to_proc[['target_qid', 'subset_name', 'filtered_date']],
                    right=df_entities_to_process_adjusted,
                    on=['target_qid', 'subset_name', 'filtered_date'], how='inner')

                df_entities_to_process_w_mentions = pd.merge(
                    left=df_entities_to_proc,
                    right=df_entities_to_process_w_mentions,
                    on=['target_qid', 'subset_name', 'filtered_date'], how='inner')

                # the nr of mentions to extract has not to be more than the actual nr of mentions available
                assert ((df_entities_to_process_w_mentions['nr_links_per_cut_per_sset'] -
                         df_entities_to_process_w_mentions['nr_mentions_to_extract_per_cut_per_sset']) >= 0).all()

                ############################## END V3 algorithm

                logger.info('df_entities_to_process_w_mentions.columns: %s' %
                            str(df_entities_to_process_w_mentions.columns))

                processed_wikidata_qids = set(df_entities_to_process_w_mentions.target_wikidata_qid.unique())
                already_processed_target_qids_per_category[curr_category['name']] = \
                    already_processed_target_qids_per_category[curr_category['name']].union(processed_wikidata_qids)
                mentions_to_process_per_category[curr_category['name']] = df_entities_to_process_w_mentions

            elif curr_category['name'] == 'new':
                logger.info('in the category of "new_entities": %s' % curr_category)
                df_entities_to_process_w_mentions_all_cuts = None
                start_new_entities = time.time()
                # The following is a sketch of possible steps of this part of algorithm:
                for idx_time_cut, curr_time_cut in enumerate(time_cuts):

                    if idx_time_cut == 0:
                        first_cut_from = curr_category['first_cut_from']
                    else:
                        first_cut_from = time_cuts[idx_time_cut - 1]

                    first_cut_from = datetime.strptime(first_cut_from, '%Y-%m-%dT%H:%M:%SZ')

                    start = time.time()
                    df_prior_w_page_info = pd.merge(filtered_anchor_target_stats[curr_time_cut],
                                                    df_page_infos_per_cut[curr_time_cut],
                                                    left_on=('target_qid',),
                                                    right_on=('wikidata_qid',), how='inner')

                    if debug:
                        # just creates a new page_info based on the rows in df_prior_stats
                        artificial_df_content = list()
                        already_added_wikidata_qids = set()
                        for idx_row, curr_row in filtered_anchor_target_stats[curr_time_cut].iterrows():
                            val_page_id = curr_row['target_page_id']
                            val_wikidata_qid = curr_row['target_qid']
                            if val_wikidata_qid in already_added_wikidata_qids:
                                continue
                            val_wikipedia_title = curr_row['target_title_2022']
                            content_length = 123
                            # nr_days = 10 * get_random_one()
                            nr_days = 10
                            wikipedia_creation_time = first_cut_from + timedelta(days=nr_days)
                            wikipedia_revision_time = wikipedia_creation_time
                            artificial_df_content.append({
                                'page_id': val_page_id,
                                'wikipedia_title': val_wikipedia_title,
                                'wikidata_qid': val_wikidata_qid,
                                'wikipedia_creation_time': wikipedia_creation_time,
                                'wikipedia_revision_time': wikipedia_revision_time,
                                'content_length': content_length
                            })
                            already_added_wikidata_qids.add(val_wikidata_qid)
                        df_page_info = pd.DataFrame(artificial_df_content)
                        df_page_info['wikipedia_creation_time'] = \
                            pd.to_datetime(df_page_info['wikipedia_creation_time'], format='%Y-%m-%dT%H:%M:%SZ')

                        # and now repeats the merge with the extra data
                        df_prior_w_page_info = pd.merge(filtered_anchor_target_stats[curr_time_cut],
                                                        df_page_info,
                                                        left_on=('target_qid',),
                                                        right_on=('wikidata_qid',), how='inner')

                    end = time.time()

                    logger.info('%s minutes (new_entities) took to merge with df_page_info, the final shape is: %s'
                                % (((end - start) / 60), str(df_prior_w_page_info.shape)))
                    start = time.time()
                    df_prior_w_page_info_new = df_prior_w_page_info[
                        df_prior_w_page_info['wikipedia_creation_time'] > first_cut_from]
                    df_prior_w_page_info_old = df_prior_w_page_info[
                        df_prior_w_page_info['wikipedia_creation_time'] <= first_cut_from]
                    end = time.time()
                    logger.info('%s minutes (new_entities) took to filter by first_cut_from of %s'
                                ' , the final shape of df_prior_w_page_info_new is: %s'
                                ' , the final shape of df_prior_w_page_info_old is: %s'
                                ' the current cut is %s' % ((end - start) / 60, curr_time_cut,
                                                            str(df_prior_w_page_info_new.shape),
                                                            str(df_prior_w_page_info_old.shape),
                                                            first_cut_from))

                    start = time.time()

                    df_entities_to_process_w_mentions = df_prior_w_page_info_new[['target_qid']] \
                        .drop_duplicates()

                    #### BEGIN first filters out the target entities that were already processed in other subset or category
                    logger.info('%s df_entities_to_process_w_mentions.shape BEFORE cleaning already processed' %
                                str(df_entities_to_process_w_mentions.shape))
                    for curr_cat in already_processed_target_qids_per_category.keys():
                        if curr_cat == curr_category['name']:
                            continue
                        df_entities_to_process_w_mentions = df_entities_to_process_w_mentions[
                            df_entities_to_process_w_mentions.apply
                            (lambda x: x['target_qid'] not in
                                       already_processed_target_qids_per_category[curr_cat], axis=1)]

                    logger.info('%s df_entities_to_process_w_mentions.shape AFTER cleaning already processed' %
                                str(df_entities_to_process_w_mentions.shape))
                    #### END first filters out the target entities that were already processed in other subset or category

                    df_entities_to_process_w_mentions = pd.merge(df_prior_w_page_info_new,
                                                                 df_entities_to_process_w_mentions,
                                                                 on=('target_qid',),
                                                                 how='inner')

                    # df_entities_to_process_w_mentions.columns:
                    # Index(['anchor_mention_text', 'target_qid',
                    #        'target_wikipedia_title_orig', 'target_page_id', 'filtered_date',
                    #        'count_mentions', 'nr_links_mention_per_entity', 'prior',
                    #        'target_title_2022', 'prior_rank', 'target_wikipedia_title_orig_lower',
                    #        'edit_distance_mention_title', 'ratio_edit_distance_mention_title',
                    #        'avg_prior_rank', 'avg_prior', 'nr_inlinks',
                    #        'nr_inlinks_after_filter_2013', 'nr_distinct_mentions_2013', 'page_id',
                    #        'wikipedia_title', 'wikidata_qid', 'wikipedia_creation_time',
                    #        'wikipedia_revision_time', 'content_length'],
                    #       dtype='object')

                    df_entities_to_process_w_mentions = df_entities_to_process_w_mentions.sample(frac=1) \
                        .reset_index(drop=True)
                    df_entities_to_process_w_mentions = df_entities_to_process_w_mentions.sort_values(
                        ['target_qid'],
                        ascending=[True])

                    groups = [df for _, df in df_entities_to_process_w_mentions.groupby('target_qid')]
                    random.shuffle(groups)

                    #
                    df_entities_to_process_w_mentions = pd.concat(groups).reset_index(drop=True)
                    #
                    #
                    df_entities_to_process_w_mentions['subset_name'] = ''
                    subset_clusters_dict = {curr_sbset_name: set() for curr_sbset_name in all_subset_names}
                    prev_target_wikidata_qid = None
                    prev_target_wikipedia_title_orig = None
                    distinct_mentions = set()

                    for curr_tuple in df_entities_to_process_w_mentions.itertuples():
                        curr_target_wikidata_qid = curr_tuple.target_wikidata_qid
                        if prev_target_wikidata_qid is None or prev_target_wikidata_qid != curr_target_wikidata_qid:
                            #
                            prev_target_wikipedia_title_orig = curr_tuple.target_wikipedia_title_orig
                            prev_target_wikidata_qid = curr_target_wikidata_qid
                            distinct_mentions = set()
                            subset_clusters_dict = {curr_sbset_name: set() for curr_sbset_name in all_subset_names}
                        mention = curr_tuple.anchor_mention_text
                        subset_clusters_dict, assigned_subset = \
                            put_mention_in_cluster(subset_clusters_dict=subset_clusters_dict,
                                                   config=config,
                                                   mention=mention)
                        distinct_mentions.add(mention)

                        df_entities_to_process_w_mentions.at[curr_tuple.Index, 'subset_name'] = assigned_subset

                    # step 01 - similar as in "random", but without the grouping on the cut ("filtered_date"), since
                    # for new entities it is not needed (no shared entities with previous cuts)
                    # df_entities_to_process_w_mentions.shape --> (2598, 29)
                    df_nr_distinct_subsets = df_entities_to_process_w_mentions \
                        .groupby(['target_qid']).subset_name.nunique().reset_index()

                    df_nr_distinct_subsets.rename(columns={'subset_name': 'nr_distinct_subsets'}, inplace=True)

                    logger.info('%s the shape BEFORE applying filter on nr_distinct_subsets' %
                                str(df_nr_distinct_subsets.shape))
                    df_nr_distinct_subsets = df_nr_distinct_subsets[df_nr_distinct_subsets['nr_distinct_subsets'] ==
                                                                    len(all_subset_names)]
                    logger.info('%s the shape AFTER applying filter on nr_distinct_subsets' %
                                str(df_nr_distinct_subsets.shape))
                    # df_nr_distinct_subsets.shape --> (534, 2)
                    df_nr_distinct_subsets = df_nr_distinct_subsets[['target_qid']].drop_duplicates()
                    # after .drop_duplicates --> shape --> (534, 1) |||
                    # df_entities_to_process_w_mentions[['target_qid']].drop_duplicates().shape --> 587,1 ||
                    # in other words we lose (587-534) target entities;
                    # those can be put only in one subset (ex: in test; or validation and test).
                    df_entities_to_process_w_mentions = pd.merge(df_entities_to_process_w_mentions,
                                                                 df_nr_distinct_subsets,
                                                                 on=('target_qid',),
                                                                 how='inner')
                    logger.info('%s the df_entities_to_process_w_mentions.shape AFTER applying filter on '
                                'nr_distinct_subsets for NEW' % str(df_entities_to_process_w_mentions.shape))
                    #
                    # similar as in "random", but without the grouping on the cut ("filtered_date"), since
                    # for new entities it is not needed (no shared entities with previous cuts)
                    #
                    df_nr_mentions_per_subset = df_entities_to_process_w_mentions \
                        .groupby(['target_qid', 'subset_name']) \
                        .nr_links_mention_per_entity.sum().reset_index()
                    df_nr_mentions_per_subset.rename(
                        columns={'nr_links_mention_per_entity': 'nr_links_per_subset'}, inplace=True)
                    df_entities_to_process_w_mentions = pd.merge(df_entities_to_process_w_mentions,
                                                                 df_nr_mentions_per_subset,
                                                                 on=('target_qid', 'subset_name'),
                                                                 how='inner')
                    #
                    # Similar as in "random", but without the grouping on the cut ("filtered_date"), since
                    # for new entities it is not needed (no shared entities with previous cuts).
                    #
                    df_dist_men = df_entities_to_process_w_mentions.groupby(['target_qid', 'subset_name']) \
                        .anchor_mention_text.nunique().reset_index()
                    col_dist_mentions = 'nr_dist_men_per_sset'
                    df_dist_men.rename(columns={'anchor_mention_text': col_dist_mentions}, inplace=True)

                    df_entities_to_process_w_mentions = pd.merge(df_entities_to_process_w_mentions, df_dist_men,
                                                                 on=('target_qid', 'subset_name'),
                                                                 how='inner')
                    #
                    # Inside itertuples, assign maximum possible bucket limited by the min_nr_mentions for each
                    # subset across cuts for each wikidata_target_qid.
                    #
                    mentions_per_entity_per_subset = dict()  # df_entities_to_process_w_mentions.shape --> (2425, 31)

                    processed_wikidata_qids = set(df_entities_to_process_w_mentions.target_wikidata_qid.unique())
                    already_processed_target_qids_per_category[curr_category['name']] = \
                        already_processed_target_qids_per_category[curr_category['name']].union(processed_wikidata_qids)

                    if df_entities_to_process_w_mentions_all_cuts is None:
                        df_entities_to_process_w_mentions_all_cuts = df_entities_to_process_w_mentions.copy()
                    else:
                        df_entities_to_process_w_mentions_all_cuts = \
                            pd.concat([df_entities_to_process_w_mentions_all_cuts, df_entities_to_process_w_mentions])

                ### BEGIN V3 - same number of entities and mentions across cuts
                #
                # df_entities_to_process_w_mentions_all_cuts.columns:
                #       Index(['anchor_mention_text', 'target_qid',
                #           'target_wikipedia_title_orig', 'target_page_id', 'filtered_date',
                #           'count_mentions', 'nr_links_mention_per_entity', 'prior',
                #           'target_title_2022', 'prior_rank', 'target_wikipedia_title_orig_lower',
                #           'edit_distance_mention_title', 'edit_distance_mention_title_ent',
                #           'ratio_edit_distance_mention_title',
                #           'ratio_edit_distance_mention_title_ent', 'overlap_type',
                #           'avg_prior_rank', 'avg_prior', 'nr_inlinks', 'nr_inlinks_after_filter',
                #           'nr_distinct_mentions', 'page_id', 'wikipedia_title', 'wikidata_qid',
                #           'wikipedia_creation_time', 'wikipedia_revision_time', 'content_length',
                #           'subset_name', 'nr_links_per_subset', 'nr_dist_men_per_sset'],
                #           dtype='object')
                #
                min_nr_entities = -1
                min_nr_entities_cut = None
                entities_to_proc_per_cut = dict()

                #
                df_entities_to_proc = df_entities_to_process_w_mentions_all_cuts.drop_duplicates(
                    subset=['target_qid', 'filtered_date']
                ).loc[:, ['target_qid', 'filtered_date']]
                #
                df_entities_to_process_adjusted = None

                for curr_time_cut in config['time_cuts']:
                    parsed_curr_date = datetime.strptime(curr_time_cut, '%Y-%m-%dT%H:%M:%SZ')
                    curr_year = parsed_curr_date.year

                    df_entities_to_process_cut = \
                        df_entities_to_proc.loc[(df_entities_to_proc['filtered_date'] == curr_time_cut),
                                                ['target_qid', 'filtered_date']]
                    entities_to_proc_per_cut[curr_time_cut] = df_entities_to_process_cut

                    if min_nr_entities_cut is None:
                        min_nr_entities_cut = curr_time_cut
                        min_nr_entities = df_entities_to_process_cut.shape[0]
                    else:
                        cs = df_entities_to_process_cut.shape[0]
                        if cs < min_nr_entities:
                            min_nr_entities = cs
                            min_nr_entities_cut = curr_time_cut

                logger.info('min_nr_entities for NEW_ENTITIES : %s for cut %s' %
                            (min_nr_entities, min_nr_entities_cut))

                for curr_time_cut in config['time_cuts']:
                    if curr_time_cut != min_nr_entities_cut:
                        entities_to_proc_per_cut[curr_time_cut] = \
                            entities_to_proc_per_cut[curr_time_cut].sample(n=min_nr_entities)
                    logger.info('new_entities shape for cut %s: %s' %
                                (curr_time_cut, str(entities_to_proc_per_cut[curr_time_cut].shape)))
                # concatenates everything
                df_entities_to_proc = pd.concat(list(entities_to_proc_per_cut.values()))
                df_entities_to_process = df_entities_to_process_w_mentions_all_cuts \
                    .groupby(['target_qid', 'filtered_date', 'subset_name']) \
                    .nr_links_per_subset.mean().reset_index()
                # now merges directly with entities_to_proc_per_cut to reduce the nr of entities (downsampled)
                df_entities_to_process = pd.merge(left=df_entities_to_process, right=df_entities_to_proc,
                                                  on=['target_qid', 'filtered_date'], how='inner')

                df_entities_to_process['nr_links_per_subset'] = \
                    df_entities_to_process['nr_links_per_subset'].astype(int)
                entities_to_process_per_cut = dict()
                for curr_subset_name in all_subset_names:
                    # now makes the nr of entities equal by removing (randomly?) some entities
                    # from other cuts to make it equal
                    min_nr_mentions_cut = None
                    min_nr_mentions_in_cut = -1
                    for curr_time_cut in config['time_cuts']:
                        entities_to_process_per_cut[curr_time_cut] = \
                            df_entities_to_process.loc[(df_entities_to_process['filtered_date'] == curr_time_cut) &
                                                       (df_entities_to_process['subset_name'] == curr_subset_name)]
                        parsed_curr_date = datetime.strptime(curr_time_cut, '%Y-%m-%dT%H:%M:%SZ')
                        curr_year = parsed_curr_date.year

                        logger.info('new_entities shape for cut %s for subset %s: %s' %
                                    (curr_time_cut, curr_subset_name,
                                     str(entities_to_process_per_cut[curr_time_cut].shape)))
                        # sorts by nr of mentions and adds extra column to later join on (similar to 'shared' case)
                        col_nr_links_name = 'nr_links_per_subset_{}'.format(curr_year)
                        entities_to_process_per_cut[curr_time_cut] = \
                            entities_to_process_per_cut[curr_time_cut] \
                                .rename(columns={'nr_links_per_subset': col_nr_links_name,
                                                 'target_qid': 'target_wikidata_qid_{}'.format(curr_year)})
                        entities_to_process_per_cut[curr_time_cut] = \
                            entities_to_process_per_cut[curr_time_cut]. \
                                sort_values([col_nr_links_name], ascending=[False])
                        #
                        entities_to_process_per_cut[curr_time_cut] \
                            .insert(0, 'new_id', range(0, 0 + entities_to_process_per_cut[curr_time_cut].shape[0]))

                        logger.info('BEFORE max_mentions_per_entity for cut %s for NEW for subset %s: %s' %
                                    (curr_time_cut, curr_subset_name, entities_to_process_per_cut[curr_time_cut] \
                                        [col_nr_links_name].sum()))
                        max_mentions_per_entity = \
                            cat_to_subset_info[curr_category['name']][curr_subset_name]['max_mentions_per_entity']
                        if max_mentions_per_entity > 0:
                            entities_to_process_per_cut[curr_time_cut] \
                                .loc[entities_to_process_per_cut[curr_time_cut][col_nr_links_name] > \
                                     max_mentions_per_entity, col_nr_links_name] = max_mentions_per_entity

                        logger.info('AFTER max_mentions_per_entity for cut %s for NEW for subset %s: %s' %
                                    (curr_time_cut, curr_subset_name, entities_to_process_per_cut[curr_time_cut] \
                                        [col_nr_links_name].sum()))

                        nr_elements_sset_cut = entities_to_process_per_cut[curr_time_cut] \
                            [col_nr_links_name].sum()
                        if min_nr_mentions_in_cut == -1:
                            min_nr_mentions_in_cut = nr_elements_sset_cut
                            min_nr_mentions_cut = curr_time_cut
                        elif nr_elements_sset_cut < min_nr_mentions_in_cut:
                            min_nr_mentions_in_cut = nr_elements_sset_cut
                            min_nr_mentions_cut = curr_time_cut

                    logger.info('min_nr_mentions_cut for NEW_ENTITIES for subset %s %s: for cut %s' %
                                (curr_subset_name, min_nr_mentions_in_cut, min_nr_mentions_cut))

                    parsed_curr_date = datetime.strptime(min_nr_mentions_cut, '%Y-%m-%dT%H:%M:%SZ')
                    min_year = parsed_curr_date.year
                    min_nr_links_name = 'nr_links_per_subset_{}'.format(min_year)
                    min_target_qid_name = 'target_wikidata_qid_{}'.format(min_year)
                    df_joined = entities_to_process_per_cut[min_nr_mentions_cut]
                    df_joined = df_joined[['new_id', min_target_qid_name, min_nr_links_name]]

                    for curr_time_cut in config['time_cuts']:
                        parsed_curr_date = datetime.strptime(curr_time_cut, '%Y-%m-%dT%H:%M:%SZ')
                        curr_year = parsed_curr_date.year
                        col_nr_links_name = 'nr_links_per_subset_{}'.format(curr_year)
                        col_target_wdata_qid = 'target_wikidata_qid_{}'.format(curr_year)
                        if curr_time_cut != min_nr_mentions_cut:
                            right_join = entities_to_process_per_cut[curr_time_cut] \
                                [['new_id', col_target_wdata_qid, col_nr_links_name]]

                            df_joined = pd.merge(left=df_joined, right=right_join, on='new_id')
                            diff_min_col_name = 'diff_min_{}'.format(curr_year)
                            df_joined[diff_min_col_name] = \
                                df_joined[col_nr_links_name] - df_joined[min_nr_links_name]

                            # df_joined['diff_min_{}'.format(curr_year)]
                            # https://kanoki.org/2019/07/17/pandas-how-to-replace-values-based-on-conditions/
                            df_joined.loc[(df_joined[diff_min_col_name] < 0), diff_min_col_name] = 0

                            # builds a list based on index to pick randomly from, indices are repeated diff times
                            # so the chance is higher for mentions with higher diff.
                            lst_pick_df_idx_from = []
                            for curr_tpl in df_joined.itertuples():
                                curr_idx = curr_tpl.Index
                                curr_diff = getattr(curr_tpl, diff_min_col_name)
                                if curr_diff > 0:
                                    curr_idx_lst = [curr_idx] * curr_diff
                                    lst_pick_df_idx_from += curr_idx_lst

                            tot_curr_cut_nr_mentions = df_joined[col_nr_links_name].sum()

                            nr_mentions_to_reduce = tot_curr_cut_nr_mentions - min_nr_mentions_in_cut
                            logger.info('!!REDUCTION TO BALANCE new_entities!! subset name: %s '
                                        ' timestamp snapshot: %s tot_curr_cut_nr_mentions: %s'
                                        ' nr_mentions_to_reduce: %s min_nr_mentions_in_cut: %s' %
                                        (curr_subset_name, curr_time_cut, tot_curr_cut_nr_mentions,
                                         nr_mentions_to_reduce, min_nr_mentions_in_cut))

                            random.shuffle(lst_pick_df_idx_from)

                            lst_idxs_to_reduce = random.sample(lst_pick_df_idx_from, nr_mentions_to_reduce)

                            for curr_idx_to_reduce in lst_idxs_to_reduce:
                                df_joined.at[curr_idx_to_reduce, col_nr_links_name] = \
                                    df_joined.at[curr_idx_to_reduce, col_nr_links_name] - 1

                        df_to_cat = df_joined.loc[:, [col_target_wdata_qid, col_nr_links_name]]
                        df_to_cat.loc[:, 'subset_name'] = curr_subset_name
                        df_to_cat.loc[:, 'filtered_date'] = curr_time_cut
                        df_to_cat = df_to_cat.rename(columns={col_target_wdata_qid: 'target_qid',
                                                              col_nr_links_name: 'nr_links_per_subset'})
                        logger.info('new_entities, nr of mentions to extract for sset %s for curr_time_cut %s: %s' %
                                    (curr_subset_name, curr_time_cut, df_to_cat['nr_links_per_subset'].sum()))
                        # check that the nr of mentions to extract equals to the nr of mentions in the cut with less
                        # mentions
                        assert df_to_cat['nr_links_per_subset'].sum() == min_nr_mentions_in_cut
                        if df_entities_to_process_adjusted is None:
                            df_entities_to_process_adjusted = df_to_cat.copy()
                        else:
                            df_entities_to_process_adjusted = pd.concat([df_entities_to_process_adjusted, df_to_cat])

                df_entities_to_process_adjusted = df_entities_to_process_adjusted.rename(
                    columns={'nr_links_per_subset': 'nr_mentions_to_extract_per_subset'})

                # controls that after the adjustments, at least one mentions of each entity has to be extracted
                assert (df_entities_to_process_adjusted['nr_mentions_to_extract_per_subset'] > 0).all()

                df_entities_to_process = pd.merge(
                    left=df_entities_to_process[['target_qid', 'filtered_date', 'subset_name']],
                    right=df_entities_to_process_adjusted,
                    on=['target_qid', 'subset_name', 'filtered_date'])

                df_entities_to_process_w_mentions_all_cuts = pd.merge(left=df_entities_to_process,
                                                                      right=df_entities_to_process_w_mentions_all_cuts,
                                                                      on=['target_qid', 'subset_name',
                                                                          'filtered_date'])
                #
                # the nr of mentions to extract has not to be more than the actual nr of mentions available
                assert ((df_entities_to_process_w_mentions_all_cuts['nr_links_per_subset'] -
                         df_entities_to_process_w_mentions_all_cuts['nr_mentions_to_extract_per_subset']) >= 0).all()
                # same number of entities and mentions across cuts
                #
                mentions_to_process_per_category[curr_category['name']] = df_entities_to_process_w_mentions_all_cuts

            else:
                raise RuntimeError('not recognized curr_category[\'type\']: ' + str(curr_category['name']))

        # invoke here get_evenly_distributed_mentions_all for each of the date cuts, this way it also doesn't need
        # to have all the inverse dictionaries of all the cuts loaded in the memory
        for idx_time_cut, curr_time_cut in enumerate(time_cuts):
            curr_dict_inverse_path = '{}{}.pickle'.format(prefix_dict_inverse_index, curr_time_cut)
            curr_dict_inverse_path = os.path.join(input_dir_stats, curr_dict_inverse_path)
            start = time.time()
            logger.info('loading inverse dictionary: %s' % curr_dict_inverse_path)
            dict_inverse_index = pickle.load(open(curr_dict_inverse_path, 'rb'))

            end = time.time()
            logger.info('06 memory usage after loading the inverse dictionary (MB): %s' %
                        (process.memory_info().rss / 1024 / 1024))
            logger.info('%s minutes to load inverse dictionary' % ((end - start) / 60))

            for curr_category in config['categories']:
                start_to_process = time.time()
                df_to_process = mentions_to_process_per_category[curr_category['name']]
                df_to_process = df_to_process[df_to_process['filtered_date'] == curr_time_cut]
                end_to_process = time.time()
                logger.info('%s minutes to filter for %s for the category %s' %
                            ((end_to_process - start_to_process) / 60, curr_time_cut, curr_category['name']))
                dict1_target_pages_to_search, dict2_anchor_pages_to_search = \
                    get_evenly_distributed_mentions_all(
                        df_entities_to_process_w_mentions=df_to_process,
                        curr_time_cut=curr_time_cut,
                        dict1_target_pages_to_search=dict1_target_pages_to_search,
                        dict2_anchor_pages_to_search=dict2_anchor_pages_to_search,
                        curr_dict_inverse_index=dict_inverse_index,
                        category_name=curr_category['name'])

            del dict_inverse_index
        entities_dict1 = set()
        for curr_cut, curr_entities in dict1_target_pages_to_search.items():
            for curr_entity_qid in curr_entities.keys():
                entities_dict1.add(curr_entity_qid)

        entities_dict2 = set()
        for curr_cut, curr_entities in dict2_anchor_pages_to_search.items():
            for curr_entity_qid in curr_entities.keys():
                entities_dict2.add(curr_entity_qid)

        len_entities_dict1 = len(entities_dict1)
        len_entities_dict2 = len(entities_dict2)
        len_entities_both = len(entities_dict1 | entities_dict2)
        expected_nr_entities_all = item_generator(config, 'nr_entities')
        expected_nr_entities_all = sum(expected_nr_entities_all)

        process_read_input_file = list()
        start_time = time.time()
        for i in range(1):
            t = Process(target=read_from_input_file,
                        args=(input_file_path, dict1_target_pages_to_search, dict2_anchor_pages_to_search, arq))
            t.start()
            process_read_input_file.append(t)

        nr_threads = config['nr_threads']
        v_shutdown_file_processor = Value('i', 0)
        v_shutdown_writer = Value('i', 0)
        v_nr_entities = Value('i', 0)
        v_nr_entities_found = Value('i', 0)
        v_nr_mentions_found = Value('i', 0)

        process_file = list()

        for i in range(nr_threads):
            t = Process(target=process_line_input_file_parallel,
                        args=(config, dict1_target_pages_to_search, dict2_anchor_pages_to_search, arq,
                              v_shutdown_file_processor, start_time, v_nr_entities, v_nr_entities_found,
                              v_nr_mentions_found))
            t.start()
            process_file.append(t)

        process_write_output_file = list()

        for i in range(1):
            t = Process(target=write_to_output_file, args=(arq, v_shutdown_writer, config,
                                                           list(all_subset_names)))
            t.start()
            process_write_output_file.append(t)

        if nr_threads >= 1:
            logger.info('LAUNCHING THREAD DISPLAY!')
            thread = Thread(target=display, args=(process_write_output_file, process_file, v_nr_entities,
                                                  v_nr_entities_found, v_nr_mentions_found, arq, start_time,
                                                  len_entities_dict1, len_entities_dict2, len_entities_both))
            thread.daemon = True  # Daemonize thread
            thread.start()  # Start the execution

        for curr_proc in process_read_input_file:
            curr_proc.join()

        v_shutdown_file_processor.value = 1

        for curr_proc in process_file:
            curr_proc.join()

        v_shutdown_writer.value = 1

        for curr_proc in process_write_output_file:
            curr_proc.join()

    loaded_target_entities = dict()
    # here loads the target entities
    for curr_time_cut in config['time_cuts']:
        loaded_target_entities[curr_time_cut] = dict()
        loaded_target_entities[curr_time_cut] = dict()
        path_to_target_entities = os.path.join(output_dir, '{}_entities.jsonl'.format(curr_time_cut))
        logger.info('reading entities from path_to_target_entities V3: %s' % path_to_target_entities)

        for curr_target_entity in tqdm(open(path_to_target_entities, 'rt')):
            curr_json_target_entity = json.loads(curr_target_entity)
            curr_target_wikidata_qid = curr_json_target_entity['wikidata_qid']
            loaded_target_entities[curr_time_cut][curr_target_wikidata_qid] = curr_json_target_entity

    # here iterates over anchor mentions and joins them with the respective target entity and produces the final
    # dataset
    output_dataset_path = os.path.join(output_dir, 'final_output')
    os.makedirs(output_dataset_path, exist_ok=True)

    for curr_time_cut in config['time_cuts']:
        for curr_subset in all_subset_names:
            path_to_anchor_mentions = os.path.join(output_dir,
                                                   '{}_{}_mentions.jsonl'.format(curr_time_cut, curr_subset))
            output_dataset_file_path = os.path.join(output_dataset_path,
                                                    '{}_{}_all.jsonl'.format(curr_time_cut, curr_subset))
            output_dataset_file_path_not_tokenized = os.path.join(output_dataset_path,
                                                                  '{}_{}_all_not_tokenized.jsonl'.format(curr_time_cut,
                                                                                                         curr_subset))
            logger.info('writing to (tokenized) %s' % output_dataset_file_path)
            with open(output_dataset_file_path, 'wt', encoding='utf8') as outfile_tokenized, \
                    open(output_dataset_file_path_not_tokenized, 'wt', encoding='utf8') as outfile_not_tokenized:
                logger.info('reading mentions from %s' % path_to_anchor_mentions)
                for curr_anchor_mention in tqdm(open(path_to_anchor_mentions, 'rt')):
                    curr_json_anchor_mention = json.loads(curr_anchor_mention)

                    target_wikidata_qid = curr_json_anchor_mention['target_qid']
                    target_details = loaded_target_entities[curr_time_cut][target_wikidata_qid]

                    target_title_2022_source2 = target_details['target_title_2022_source2']
                    target_orig_title_source2 = target_details['target_orig_title_source2']

                    target_page_id_source2 = target_details['target_page_id_source2']
                    target_wikidata_qid_source2 = target_details['target_wikidata_qid_source2']
                    target_wikipedia_creation_date = target_details['target_wikipedia_creation_date']
                    target_wikipedia_revision_date = target_details['target_wikipedia_revision_date']

                    curr_json_anchor_mention['target_orig_title_source2'] = target_orig_title_source2
                    curr_json_anchor_mention['target_title_2022_source2'] = target_title_2022_source2
                    curr_json_anchor_mention['target_page_id_source2'] = target_page_id_source2
                    curr_json_anchor_mention['target_wikidata_qid_source2'] = target_wikidata_qid_source2

                    curr_json_anchor_mention['target_wikipedia_creation_date'] = target_wikipedia_creation_date
                    curr_json_anchor_mention['target_wikipedia_revision_date'] = target_wikipedia_revision_date

                    if target_orig_title_source2 != curr_json_anchor_mention['target_orig_title']:
                        logger.warning('!!WARNING!!, target_orig_title mismatch: %s' % curr_json_anchor_mention)
                    if target_title_2022_source2 != curr_json_anchor_mention['target_title_2022']:
                        logger.warning('!!WARNING!!, target_title_2022 mismatch: %s' % curr_json_anchor_mention)
                    if target_wikidata_qid_source2 != curr_json_anchor_mention['target_qid']:
                        logger.warning('!!WARNING!!, target_wikidata_qid mismatch: %s' % curr_json_anchor_mention)
                    if target_page_id_source2 != curr_json_anchor_mention['target_page_id']:
                        logger.warning('!!WARNING!!, target_page_id mismatch: %s' % curr_json_anchor_mention)

                    curr_json_anchor_mention_tokenized = curr_json_anchor_mention.copy()
                    curr_json_anchor_mention_not_tokenized = curr_json_anchor_mention.copy()
                    del curr_json_anchor_mention_tokenized['context_left']
                    del curr_json_anchor_mention_tokenized['context_right']
                    del curr_json_anchor_mention_tokenized['mention']

                    del curr_json_anchor_mention_not_tokenized['context_right_bert']
                    del curr_json_anchor_mention_not_tokenized['context_left_bert']
                    del curr_json_anchor_mention_not_tokenized['mention_bert']

                    curr_json_anchor_mention_tokenized['target_bert'] = target_details['label_bert_tokenized']

                    curr_json_anchor_mention_tokenized['target_title_bert'] = target_details[
                        'label_title_bert_tokenized']

                    # limits in the nr of tokens for tokenized, if not gets too big
                    len_mention_context = int(max_len_context_tokenized / 2)
                    curr_json_anchor_mention_tokenized['context_right_bert'] = \
                        curr_json_anchor_mention_tokenized['context_right_bert'][:len_mention_context]
                    curr_json_anchor_mention_tokenized['context_left_bert'] = \
                        curr_json_anchor_mention_tokenized['context_left_bert'][-len_mention_context:]
                    curr_json_anchor_mention_tokenized['target_bert_'] = \
                        curr_json_anchor_mention_tokenized['target_bert'][:max_len_context_tokenized]
                    #

                    curr_json_anchor_mention_not_tokenized['target_text'] = target_details['target_text']
                    curr_json_anchor_mention_not_tokenized['target_title'] = target_details['target_title']

                    curr_json_anchor_mention_not_tokenized['target_len'] = target_details['target_len']
                    curr_json_anchor_mention_tokenized['target_len'] = target_details['target_len']

                    assert isclose(curr_json_anchor_mention['avg_prior_rank'],
                                   target_details['avg_prior_rank'], rel_tol=1e-7, abs_tol=0.0)
                    assert isclose(curr_json_anchor_mention['avg_prior'], target_details['avg_prior'],
                                   rel_tol=1e-7, abs_tol=0.0)
                    if curr_json_anchor_mention['nr_inlinks'] != target_details['nr_inlinks']:
                        logger.info('WARNING nr_inlinks DO NOT MATCH {} (anchor mention) vs {} (target): '
                                    'curr_json_anchor_mention: ' \
                                    .format(curr_json_anchor_mention['nr_inlinks'], target_details['nr_inlinks']),
                                    curr_json_anchor_mention, ' target_details: ', target_details)
                    assert curr_json_anchor_mention['nr_inlinks'] == target_details['nr_inlinks']

                    outfile_tokenized.write(json.dumps(curr_json_anchor_mention_tokenized, ensure_ascii=False) + '\n')
                    outfile_tokenized.flush()

                    outfile_not_tokenized.write(
                        json.dumps(curr_json_anchor_mention_not_tokenized, ensure_ascii=False) + '\n')
                    outfile_not_tokenized.flush()
