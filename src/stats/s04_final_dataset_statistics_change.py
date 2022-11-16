# based on src/wiki/wikipedia/jobs/dataset_creation/wikipedia_get_top_changes.py
# statistics of the final dataset related to change in the content of entities/context around the mentions
import argparse
import json
import logging
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from stats.utils.s04_final_dataset_statistics_change_utils import get_mappings, get_similarities_dataframe
from tempel_creation.misc.utils import from_bert_to_text, get_ratio_edit_distance_v2
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        # TODO
                        default='TODO',
                        help='The config file that contains all the parameters')

    args = parser.parse_args()
    logger.info('getting the final dataset with the following parameters: %s' % args)
    config = json.load(open(args.config_file, 'rt'))
    stats_to_run = set(config['stats_to_run'])
    output_plot_path = config['output_plot_path']
    os.makedirs(output_plot_path, exist_ok=True)
    gpu_id = 0
    device = 'cpu'
    if not config['no_cuda']:
        device = torch.device('cuda:{}'.format(gpu_id))

    base_input_path = config['base_input_path']
    shared_category_name = config['shared_category_name']

    max_len_edit_distance = config['max_len_edit_distance']
    max_len_context_around_mention = config['max_len_context_around_mention']

    cache_path = config['cache_path']
    cache_path = os.path.join(base_input_path, cache_path)
    os.makedirs(cache_path, exist_ok=True)

    cache_file_path = os.path.join(cache_path, 's06_final_dataset_data.pickle')

    shared_qid_to_cut_to_content = dict()
    shared_qid_to_cut_to_tokens = dict()
    shared_qid_to_cut_to_mentions_to_tokens = dict()
    shared_qid_to_cut_to_subset_to_mentions_to_tokens = dict()

    sorted_cuts = list(config['time_cuts'].keys())
    sorted_cuts = sorted(sorted_cuts)

    if os.path.exists(cache_file_path):
        start_dump = time.time()
        logger.info('reading cache_file_path: %s' % cache_file_path)
        cache_loaded = pickle.load(open(cache_file_path, 'rb'))
        shared_qid_to_cut_to_content = cache_loaded['shared_qid_to_cut_to_content']
        shared_qid_to_cut_to_tokens = cache_loaded['shared_qid_to_cut_to_tokens']
        shared_qid_to_cut_to_mentions_to_tokens = cache_loaded['shared_qid_to_cut_to_mentions_to_tokens']
        end_dump = time.time()
        logger.info('%s mins to END dumping the data to pickle' % ((end_dump - start_dump) / 60))
    else:
        # goes directly to where the TARGET entities are saved
        for curr_cut in sorted_cuts:
            curr_year_config = config['time_cuts'][curr_cut]
            target_entities_path = os.path.join(base_input_path, curr_year_config['target_entities_only_path'])
            logger.info('processing curr_cut for target %s' % curr_cut)
            for curr_line in tqdm(open(target_entities_path)):
                parsed_line = json.loads(curr_line)
                target_wikidata_qid = parsed_line['wikidata_qid']
                if shared_category_name in parsed_line['category']:
                    if target_wikidata_qid not in shared_qid_to_cut_to_content:
                        shared_qid_to_cut_to_content[target_wikidata_qid] = dict()
                        shared_qid_to_cut_to_tokens[target_wikidata_qid] = dict()
                    target_content = parsed_line['label_bert_tokenized']
                    target_content = from_bert_to_text(target_content).strip()
                    shared_qid_to_cut_to_content[target_wikidata_qid][curr_cut] = target_content
                    shared_qid_to_cut_to_tokens[target_wikidata_qid][curr_cut] = target_content.split(' ')

        for curr_cut in sorted_cuts:
            curr_year_config = config['time_cuts'][curr_cut]
            test_path = os.path.join(base_input_path, curr_year_config['test_path'])
            validation_path = os.path.join(base_input_path, curr_year_config['validation_path'])
            train_path = os.path.join(base_input_path, curr_year_config['train_path'])
            subset_paths = [train_path, validation_path, test_path]

            for curr_subset_path in subset_paths:
                logger.info('processing %s' % curr_subset_path)
                for curr_line in tqdm(open(curr_subset_path)):
                    curr_line_json = json.loads(curr_line)
                    context_right_bert_tokenized = curr_line_json['context_right_bert_tokenized']
                    context_left_bert_tokenized = curr_line_json['context_left_bert_tokenized']
                    context_right = from_bert_to_text(context_right_bert_tokenized).strip()
                    context_left = from_bert_to_text(context_left_bert_tokenized).strip()

                    target_wikidata_qid = curr_line_json['target_wikidata_qid']
                    if target_wikidata_qid not in shared_qid_to_cut_to_mentions_to_tokens:
                        shared_qid_to_cut_to_mentions_to_tokens[target_wikidata_qid] = dict()
                        shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid] = dict()
                    if curr_cut not in shared_qid_to_cut_to_mentions_to_tokens[target_wikidata_qid]:
                        shared_qid_to_cut_to_mentions_to_tokens[target_wikidata_qid][curr_cut] = list()
                        shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][curr_cut] = dict()
                    shared_qid_to_cut_to_mentions_to_tokens[target_wikidata_qid][curr_cut] \
                        .append({'context_right_tokenized': context_right.split(' ')[:max_len_context_around_mention],
                                 'context_left_tokenized': context_left.split(' ')[(-max_len_context_around_mention):]})
                    if curr_subset_path == train_path:
                        if 'train' not in shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][
                            curr_cut]:
                            shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][curr_cut][
                                'train'] = list()
                        shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][curr_cut]['train'] \
                            .append(
                            {'context_right_tokenized': context_right.split(' ')[:max_len_context_around_mention],
                             'context_left_tokenized': context_left.split(' ')[(-max_len_context_around_mention):]})
                    elif curr_subset_path == validation_path:
                        if 'validation' not in shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][
                            curr_cut]:
                            shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][curr_cut][
                                'validation'] = list()
                        shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][curr_cut]['validation'] \
                            .append(
                            {'context_right_tokenized': context_right.split(' ')[:max_len_context_around_mention],
                             'context_left_tokenized': context_left.split(' ')[(-max_len_context_around_mention):]})
                    elif curr_subset_path == test_path:
                        if 'test' not in shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][
                            curr_cut]:
                            shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][curr_cut][
                                'test'] = list()
                        shared_qid_to_cut_to_subset_to_mentions_to_tokens[target_wikidata_qid][curr_cut]['test'] \
                            .append(
                            {'context_right_tokenized': context_right.split(' ')[:max_len_context_around_mention],
                             'context_left_tokenized': context_left.split(' ')[(-max_len_context_around_mention):]})
        start_dump = time.time()
        logger.info('BEGIN dumping the data to pickle')
        pickle.dump(
            {'shared_qid_to_cut_to_content': shared_qid_to_cut_to_content,
             'shared_qid_to_cut_to_tokens': shared_qid_to_cut_to_tokens,
             'shared_qid_to_cut_to_subset_to_mentions_to_tokens': shared_qid_to_cut_to_subset_to_mentions_to_tokens,
             'shared_qid_to_cut_to_mentions_to_tokens': shared_qid_to_cut_to_mentions_to_tokens},
            open(cache_file_path, 'wb')
        )
        end_dump = time.time()
        logger.info('%s mins to END dumping the data to pickle' % ((end_dump - start_dump) / 60))

    # for a particular target entity I guess not all the cuts will have the content in the debug mode
    if config['debug']:
        for target_wikidata_qid, content_cuts in shared_qid_to_cut_to_content.items():
            first_content = content_cuts[list(content_cuts.keys())[0]]
            for curr_cut in sorted_cuts:
                if curr_cut not in content_cuts:
                    content_cuts[curr_cut] = first_content

    ##### end get_mappings
    ############### BEGIN - dot product (similarities) part -
    df_wikipedia_changes = None
    if 'dot_product_wrt_first' in stats_to_run or 'max_dot_product_examples' in stats_to_run:
        input_encodings_dir = config['input_encodings_dir']

        shared_target_wikidata_qids = set(shared_qid_to_cut_to_content.keys())
        # step 02 - the rest should be more or less the same as in 'dot_product_wrt_first'
        mappings = get_mappings(config=config,
                                sorted_cuts=sorted_cuts,
                                base_input_path=base_input_path,
                                input_encodings_dir=input_encodings_dir,
                                shared_target_wikidata_qids=shared_target_wikidata_qids,
                                device=device)
        cut_to_label_id_wikidata_qid_map = mappings['cut_to_label_id_wikidata_qid_map']
        cut_to_shared_label_ids_to_filtered_ids = mappings['cut_to_shared_label_ids_to_filtered_ids']
        cut_to_filtered_tensors = mappings['cut_to_filtered_tensors']

        cut_wikipedia_changes = \
            get_similarities_dataframe(config=config,
                                       device=device,
                                       shared_target_wikidata_qids=shared_target_wikidata_qids,
                                       cut_to_label_id_wikidata_qid_map=cut_to_label_id_wikidata_qid_map,
                                       sorted_cuts=sorted_cuts,
                                       cut_to_shared_label_ids_to_filtered_ids=cut_to_shared_label_ids_to_filtered_ids,
                                       cut_to_filtered_tensors=cut_to_filtered_tensors)

        df_wikipedia_changes = pd.DataFrame(data=cut_wikipedia_changes)

    if 'dot_product_wrt_first' in stats_to_run:
        ############### END - dot product (similarities) part - TODO: if necessary, can be parallelized

        df_grouped_changes = df_wikipedia_changes[['cut_to', 'dot_product']].groupby(['cut_to']).dot_product.mean()
        df_grouped_changes = df_grouped_changes.reset_index()
        logger.info('df_grouped_changes: %s' % df_grouped_changes.head(10))

        #####################
        fig_plt, axs_plt = plt.subplots(1, 1, figsize=(10, 10))

        df_pivotet = df_grouped_changes
        logger.info('df_pivotet: %s' % df_pivotet)
        df_pivotet.plot(ax=axs_plt, x='cut_to', y='dot_product')
        curr_plot_path = os.path.join(output_plot_path, 'dataset_statistics_change_v01.pdf')
        logger.info('saving the graph into curr_plot_path: %s' % curr_plot_path)
        fig_plt.savefig(curr_plot_path, format='pdf')
        #####################

    if 'max_dot_product_examples' in stats_to_run:
        logger.info('max_dot_product_examples')

        df_wikipedia_changes.sort_values(by='dot_product', inplace=True)
        logger.info('sorted df_wikipedia_changes: %s' % df_wikipedia_changes)
        top_to_show = 10
        for row in df_wikipedia_changes.itertuples():
            target_wikidata_qid = row.wikidata_qid
            cut_from = row.cut_from
            cut_to = row.cut_to

            content_cut_from = shared_qid_to_cut_to_content[target_wikidata_qid][cut_from]
            content_cut_to = shared_qid_to_cut_to_content[target_wikidata_qid][cut_to]

            logger.info('==========================')
            logger.info('cut_from: %s' % cut_from)
            logger.info('content_from: %s' % content_cut_from[:500])
            logger.info('-------')
            logger.info('cut_to: %s' % cut_to)
            logger.info('content_to: %s' % content_cut_to[:500])
            logger.info('==========================')

    if 'dot_product_context_mentions_wrt_first' in stats_to_run:
        # TODO
        # generic - similar as dot_product_wrt_first BUT for mention context
        # step 01 -
        #   for dot product:
        #       get all the text around the mentions (mentions including), basically similar to what is done
        #       in the prediction module s06_main_eval_faiss_biencoder_blink.py, where the bi-encoder encoding is also
        #       done.
        #   for edit distance:
        #       get all the anchor mention text associated with each of the target entities.
        # step 02 -
        #   - edit distance between the mentions of each entity.
        #   - dot product using the same embeddings as in s06_main_eval_faiss_biencoder_blink.py.
        pass

    if 'jackard_vocabulary_mentions_wrt_test' in stats_to_run:
        logger.info('starting jackard_vocabulary_mentions_wrt_test')
        diff_lst = list()
        nr_shared = 0

        ssets_to_compare = ['train', 'validation', 'test']

        for curr_target_qid, cuts_data in tqdm(shared_qid_to_cut_to_subset_to_mentions_to_tokens.items()):

            if len(cuts_data) != len(sorted_cuts):
                continue
            nr_shared += 1
            min_context_len = 99999999
            min_nr_mentions = 99999999

            all_subsets_point_to_target_entity = True
            for curr_cut, context_contents_per_sset in cuts_data.items():

                for curr_subset in ssets_to_compare:
                    if curr_subset not in context_contents_per_sset:
                        logger.info('warning(should not happen since all target qid are supposed to be shared between '
                                    'subsets), qid not shared for %s: %s' % (curr_subset, curr_target_qid))
                        all_subsets_point_to_target_entity = False
                        break

                    min_nr_mentions = min(min_nr_mentions, len(context_contents_per_sset[curr_subset]))
                    random.shuffle(context_contents_per_sset[curr_subset])

                    for curr_context in context_contents_per_sset[curr_subset]:
                        curr_tot_context_len = len(curr_context['context_left_tokenized']) + \
                                               len(curr_context['context_right_tokenized'])
                        min_context_len = min(min_context_len, curr_tot_context_len)
                if not all_subsets_point_to_target_entity:
                    break
            if not all_subsets_point_to_target_entity:
                continue

            cut_to_sset_to_context_tokens = dict()
            for curr_cut, context_contents_per_sset in cuts_data.items():
                for curr_subset in ssets_to_compare:
                    cut_to_sset_to_context_tokens[curr_subset] = dict()
                    random.shuffle(context_contents_per_sset[curr_subset])
                    curr_tokens = []
                    for idx_ctx, curr_context in enumerate(context_contents_per_sset[curr_subset]):
                        if idx_ctx >= min_nr_mentions:
                            break
                        half_min_context_len = int(min_context_len / 2)
                        if len(curr_context['context_left_tokenized']) >= half_min_context_len and \
                                len(curr_context['context_right_tokenized']) >= half_min_context_len:
                            curr_tokens += curr_context['context_left_tokenized'][-half_min_context_len:] + \
                                           curr_context['context_right_tokenized'][:half_min_context_len]
                        elif len(curr_context['context_left_tokenized']) < half_min_context_len < \
                                len(curr_context['context_right_tokenized']):
                            diff = half_min_context_len - len(curr_context['context_left_tokenized'])
                            curr_tokens += curr_context['context_left_tokenized'] + \
                                           curr_context['context_right_tokenized'][:(half_min_context_len + diff)]
                        elif len(curr_context['context_right_tokenized']) < half_min_context_len < \
                                len(curr_context['context_left_tokenized']):
                            diff = half_min_context_len - len(curr_context['context_right_tokenized'])
                            curr_tokens += curr_context['context_left_tokenized'][-(half_min_context_len + diff):] + \
                                           curr_context['context_right_tokenized']
                    cut_to_sset_to_context_tokens[curr_subset][curr_cut] = set(curr_tokens)

            for curr_compare_to_cut in sorted_cuts:
                diff_lst = list()
                toks_curr_cut_1st_sset = cut_to_sset_to_context_tokens[ssets_to_compare[0]][curr_compare_to_cut]
                for curr_cut in sorted_cuts:
                    jackard_dist_per_sset = dict()
                    ignore = False
                    for curr_sset in ssets_to_compare[1:]:
                        curr_cut_toks = set(cut_to_sset_to_context_tokens[curr_sset][curr_cut])
                        intersec = curr_cut_toks.intersection(toks_curr_cut_1st_sset)
                        union = curr_cut_toks.union(toks_curr_cut_1st_sset)
                        if len(union) == 0:
                            logger.info('WARNING union in 0 for %s' % curr_target_qid)
                            ignore = True
                            continue

                        jackard_dist_per_sset[curr_sset] = len(intersec) / len(union)
                    if not ignore:
                        diff_lst.append(
                            {('jackard_%s' % ssets_to_compare[1]): jackard_dist_per_sset[ssets_to_compare[1]],
                             ('jackard_%s' % ssets_to_compare[2]): jackard_dist_per_sset[ssets_to_compare[2]],
                             'year': curr_cut})

                logger.info('NR OF SHARED ENTITIES VERIFIED: %s' % nr_shared)
                df_diff_lst = pd.DataFrame(diff_lst)
                df_diff_lst_gr = df_diff_lst.groupby(['year']).mean().reset_index()
                # on subplots some nice examples:
                # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
                fig_plt, axs_plt1 = plt.subplots(1, 1, figsize=(5, 5))
                axs_plt1.set_ylabel('Jackard Similarity')
                axs_plt1.set_xlabel('Year')
                axs_plt1.set_title('Jackard Vocabulary Similarity (mentions)')
                curr_plot_path = os.path.join(output_plot_path, 'jackard_mentions_years_wrt_test_comp_year{}.pdf'
                                              .format(curr_compare_to_cut))
                x_ticks = list()
                for curr_cut in sorted_cuts:
                    x_ticks.append(int(curr_cut))

                df_diff_lst_gr.plot(ax=axs_plt1, x='year')
                axs_plt1.set_xticks(range(len(x_ticks)))
                axs_plt1.set_xticklabels(x_ticks)
                plt.show()
                fig_plt.savefig(curr_plot_path, format='pdf')

    if 'jackard_vocabulary_mentions_wrt_first' in stats_to_run:
        logger.info('starting jackard_vocabulary_mentions_wrt_first')
        diff_lst = list()
        nr_shared = 0
        for curr_target_qid, cuts_data in tqdm(shared_qid_to_cut_to_mentions_to_tokens.items()):
            if len(cuts_data) != len(sorted_cuts):
                continue
            nr_shared += 1
            min_context_len = 99999999
            min_nr_mentions = 99999999
            for curr_cut, context_contents in cuts_data.items():
                min_nr_mentions = min(min_nr_mentions, len(context_contents))
                random.shuffle(context_contents)
                for curr_context in context_contents:
                    curr_tot_context_len = len(curr_context['context_left_tokenized']) + \
                                           len(curr_context['context_right_tokenized'])
                    min_context_len = min(min_context_len, curr_tot_context_len)
            cut_to_context_tokens = dict()
            for curr_cut, context_contents in cuts_data.items():
                random.shuffle(context_contents)
                curr_tokens = []
                for idx_ctx, curr_context in enumerate(context_contents):
                    if idx_ctx >= min_nr_mentions:
                        break
                    half_min_context_len = int(min_context_len / 2)
                    if len(curr_context['context_left_tokenized']) >= half_min_context_len and \
                            len(curr_context['context_right_tokenized']) >= half_min_context_len:
                        curr_tokens += curr_context['context_left_tokenized'][-half_min_context_len:] + \
                                       curr_context['context_right_tokenized'][:half_min_context_len]
                    elif len(curr_context['context_left_tokenized']) < half_min_context_len < \
                            len(curr_context['context_right_tokenized']):
                        diff = half_min_context_len - len(curr_context['context_left_tokenized'])
                        curr_tokens += curr_context['context_left_tokenized'] + \
                                       curr_context['context_right_tokenized'][:(half_min_context_len + diff)]
                    elif len(curr_context['context_right_tokenized']) < half_min_context_len < \
                            len(curr_context['context_left_tokenized']):
                        diff = half_min_context_len - len(curr_context['context_right_tokenized'])
                        curr_tokens += curr_context['context_left_tokenized'][-(half_min_context_len + diff):] + \
                                       curr_context['context_right_tokenized']
                cut_to_context_tokens[curr_cut] = set(curr_tokens)

            toks_1st_cut = cut_to_context_tokens[sorted_cuts[0]]
            for curr_cut in sorted_cuts:

                curr_cut_toks = set(cut_to_context_tokens[curr_cut])
                intersec = curr_cut_toks.intersection(toks_1st_cut)
                union = curr_cut_toks.union(toks_1st_cut)
                if len(union) == 0:
                    logger.info('WARNING union in 0 for %s' % curr_target_qid)
                    continue
                jackard_dist_1st_cut = len(intersec) / len(union)
                diff_lst.append(
                    {'jackard_similarity': jackard_dist_1st_cut, 'year': curr_cut})
        logger.info('NR OF SHARED ENTITIES VERIFIED: %s' % nr_shared)
        df_diff_lst = pd.DataFrame(diff_lst)
        df_diff_lst_gr = df_diff_lst.groupby(['year']).mean().reset_index()
        # on subplots some nice examples:
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
        fig_plt, axs_plt1 = plt.subplots(1, 1, figsize=(5, 5))
        axs_plt1.set_ylabel('Jackard Similarity')
        axs_plt1.set_xlabel('Year')
        axs_plt1.set_title('Jackard Vocabulary Similarity (mentions)')
        curr_plot_path = os.path.join(output_plot_path, 'jackard_mentions_years.pdf')
        #
        curr_plot_csv_path = os.path.join(output_plot_path, 'jackard_mentions_years.csv')
        df_diff_lst_gr.to_csv(curr_plot_csv_path, sep='\t', index=False)
        #
        x_ticks = list()
        for curr_cut in sorted_cuts:
            x_ticks.append(int(curr_cut))

        df_diff_lst_gr.plot(ax=axs_plt1, x='year', y='jackard_similarity')
        axs_plt1.set_xticks(range(len(x_ticks)))
        axs_plt1.set_xticklabels(x_ticks)
        plt.show()
        fig_plt.savefig(curr_plot_path, format='pdf')

    if 'jackard_vocabulary_entities_wrt_first' in stats_to_run or \
            'edit_distance_entities_wrt_first' in stats_to_run:
        # based on the formulation in
        # https://studymachinelearning.com/jaccard-similarity-text-similarity-metric-in-nlp/
        logger.info('working on jackard_vocabulary_entities_wrt_first')
        diff_lst = list()
        logger.info('processing for jackard distance/edit distance')
        jackard_distance_sorted = list()  # sorts the jackard distance from lowest to highest
        for wikidata_qid, cuts_content in tqdm(shared_qid_to_cut_to_tokens.items()):
            if len(cuts_content) == len(sorted_cuts):
                vocab_per_cut = None
                diff_vocab_wrt_1st_year = None
                min_len_across_cuts = len(cuts_content[sorted_cuts[0]])
                for curr_cut in sorted_cuts:
                    toks_cut = cuts_content[curr_cut]
                    min_len_across_cuts = min(min_len_across_cuts, len(toks_cut))
                min_len_across_cuts_ed_dist = min(max_len_edit_distance, min_len_across_cuts)
                toks_1st_cut = set(cuts_content[sorted_cuts[0]][:min_len_across_cuts])
                str_toks_1st_cut = ' '.join(cuts_content[sorted_cuts[0]][:min_len_across_cuts_ed_dist])
                lowest_jackard_distance = 2.0
                lowest_jackard_distance_year = None
                for curr_cut in sorted_cuts:
                    curr_cut_toks = set(cuts_content[curr_cut][:min_len_across_cuts])
                    curr_cut_str_toks = ' '.join(cuts_content[curr_cut][:min_len_across_cuts_ed_dist])
                    intersec = curr_cut_toks.intersection(toks_1st_cut)
                    union = curr_cut_toks.union(toks_1st_cut)
                    jackard_dist_1st_cut = len(intersec) / len(union)
                    if jackard_dist_1st_cut < lowest_jackard_distance:
                        lowest_jackard_distance = jackard_dist_1st_cut
                        lowest_jackard_distance_year = curr_cut
                    edit_distance = get_ratio_edit_distance_v2(str_toks_1st_cut, curr_cut_str_toks)
                    diff_lst.append(
                        {'jackard_similarity': jackard_dist_1st_cut, 'edit_distance': edit_distance[1],
                         'year': curr_cut})
                jackard_distance_sorted.append({'text_1st_cut': cuts_content[sorted_cuts[0]][:256],
                                                'text_biggest_cut': cuts_content[lowest_jackard_distance_year][:256],
                                                '1st_cut': sorted_cuts[0],
                                                'biggest_cut': lowest_jackard_distance_year,
                                                'jackard_distance': lowest_jackard_distance,
                                                'wikidata_qid': wikidata_qid})

        jackard_distance_sorted = sorted(jackard_distance_sorted, key=lambda x: x['jackard_distance'])
        cache_jackard_distance_list_path = os.path.join(cache_path, 'jackard_distance_sorted_list.pickle')
        logger.info('BEGIN pickling jackard_distance_sorted')
        pickle.dump(jackard_distance_sorted, open(cache_jackard_distance_list_path, 'wb'))
        logger.info('END pickling jackard_distance_sorted')

        jackard_distance_sorted = jackard_distance_sorted[:1000]
        logger.info('LOWEST JACKARD SIMILARITIES FOR ENTITIES: ')
        for curr_jackard_distance in jackard_distance_sorted:
            logger.info('==========================================')
            logger.info('wikidata_qid: %s' % curr_jackard_distance['wikidata_qid'])
            logger.info('jackard_distance: %s' % curr_jackard_distance['jackard_distance'])
            logger.info('date 1st cut: %s' % curr_jackard_distance['1st_cut'])
            logger.info('text_1st_cut: %s' % curr_jackard_distance['text_1st_cut'])
            logger.info('biggest_cut: %s' % curr_jackard_distance['biggest_cut'])
            logger.info('text_biggest_cut: %s' % curr_jackard_distance['text_biggest_cut'])
            logger.info('==========================================')
        df_diff_lst = pd.DataFrame(diff_lst)
        df_diff_lst_gr = df_diff_lst.groupby(['year']).mean().reset_index()

        #
        curr_plot_csv_path = os.path.join(output_plot_path, 'jackard_entities.csv')
        df_diff_lst_gr.to_csv(curr_plot_csv_path, sep='\t', index=False)
        #

        # on subplots some nice examples:
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
        fig_plt, (axs_plt1, axs_plt2) = plt.subplots(1, 2, figsize=(10, 5))
        axs_plt1.set_ylabel('Jackard Similarity')
        axs_plt1.set_xlabel('Year')
        axs_plt1.set_title('Jackard Vocabulary Similarity (entities)')
        axs_plt2.set_ylabel('Edit Distance')
        axs_plt2.set_xlabel('Year')
        axs_plt2.set_title('Edit Distance (entities)')
        curr_plot_path = os.path.join(output_plot_path, 'jackard_entities_years.pdf')
        x_ticks = list()
        for curr_cut in sorted_cuts:
            x_ticks.append(int(curr_cut))

        df_diff_lst_gr.plot(ax=axs_plt1, x='year', y='jackard_similarity')
        df_diff_lst_gr.plot(ax=axs_plt2, x='year', y='edit_distance')

        axs_plt1.set_xticks(range(len(x_ticks)))
        axs_plt1.set_xticklabels(x_ticks)
        axs_plt2.set_xticks(range(len(x_ticks)))
        axs_plt2.set_xticklabels(x_ticks)
        plt.show()
        fig_plt.savefig(curr_plot_path, format='pdf')

    logger.info('done here final')
