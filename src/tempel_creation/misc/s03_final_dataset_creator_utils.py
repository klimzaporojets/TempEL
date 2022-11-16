import io
import json
import logging
import os
import random
import shutil
import time
import traceback
from datetime import datetime
from typing import List, Set, Dict

import pandas as pd
from pytorch_transformers import BertTokenizer

from tempel_creation.misc.article_queue import ArticleReadingQueue
from tempel_creation.misc.cleaning import clean_text_from_link_markers
from tempel_creation.misc.compiled_regexes import compiled_regexes
from tempel_creation.misc.utils import from_bert_to_text
from tempel_creation.s01_wikipedia_clean_and_tokenize import parse_mentions_from_source
from tempel_creation.s02_alias_table_generator import one_string_in_another, get_ratio_edit_distance_v2
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def get_random_mentions_with_context(anchor_mention_text, target_orig_title, source, source_title,
                                     curr_time_cut,
                                     config, bert_tokenizer, nr_mentions_to_extract):
    max_len_context = config['max_len_context']
    compiled_mention_finder = compiled_regexes['compiled_mention_finder']
    compiled_country_in_link = compiled_regexes['compiled_country_in_link']

    mention_links, tot_detected_mentions, tot_links_errors = \
        parse_mentions_from_source(source=source,
                                   source_title=source_title,
                                   filtered_date=None,
                                   compiled_mention_finder=compiled_mention_finder,
                                   compiled_country_in_link=compiled_country_in_link,
                                   anchor_wikidata_qid=None,
                                   get_span_pos=True)
    # first filters, then shuffles
    mention_links_ret = [ml for ml in mention_links if ml['anchor_mention_text'].lower() == anchor_mention_text
                         and ml['target_wikipedia_title_orig'] == target_orig_title]

    if len(mention_links_ret) == 0:
        logger.info('didn\'t find the mention %s with target %s'
                    ' for source_title %s for curr_time_cut of %s' %
                    (anchor_mention_text, target_orig_title, source_title, curr_time_cut))

        # 18/04/2022 - it still looks for the particular mention and if this particular mention is not linked to
        # more than 1 specific page, just returns it.
        # we are here
        mention_links_ret = [ml for ml in mention_links if ml['anchor_mention_text'].lower() == anchor_mention_text]
        all_titles_in_links = set([ml['target_wikipedia_title_orig'] for ml in mention_links_ret])
        if len(all_titles_in_links) > 1 or len(mention_links) == 0:
            # more than 1 target for a particular mention, then we can not be sure what we need
            # or directly no mention found
            return None

    random.shuffle(mention_links_ret)

    if len(mention_links_ret) > nr_mentions_to_extract:
        mention_links_ret = mention_links_ret[:nr_mentions_to_extract]

    to_ret_mentions = list()
    for curr_link in mention_links_ret:
        to_ret_curr_mention = dict()
        context_left = source[:curr_link['span'][0]]
        context_left = clean_text_from_link_markers(context_left)

        context_right = source[curr_link['span'][1]:]
        context_right = clean_text_from_link_markers(context_right)

        # BEGIN: new added in order to not keep the dataset lowercased
        to_ret_curr_mention['context_right'] = context_right
        to_ret_curr_mention['context_left'] = context_left
        to_ret_curr_mention['mention'] = curr_link['anchor_mention_text']
        # END: new added in order to not keep the dataset lowercased

        context_left_short = ' '.join(context_left.split(' ')[-max_len_context:])
        context_right_short = ' '.join(context_right.split(' ')[:max_len_context])

        context_right_bert = bert_tokenizer.tokenize(context_right_short)

        to_ret_curr_mention['context_right_bert_tokenized'] = context_right_bert
        context_left_bert = bert_tokenizer.tokenize(context_left_short)
        to_ret_curr_mention['context_left_bert_tokenized'] = context_left_bert
        mention_bert_tokenized = bert_tokenizer.tokenize(curr_link['anchor_mention_text'])
        to_ret_curr_mention['mention_bert_tokenized'] = mention_bert_tokenized
        to_ret_mentions.append(to_ret_curr_mention)
    return to_ret_mentions


def get_evenly_distributed_mentions_all(
        df_entities_to_process_w_mentions,
        curr_time_cut,
        dict1_target_pages_to_search,
        dict2_anchor_pages_to_search,
        curr_dict_inverse_index: Dict,
        category_name: str):
    """
    The generic version, so there is no code repetition between get_evenly_distributed_mentions_random and
    get_evenly_distributed_mentions_new_entities. The only few places in code where the logic/access fields
    are different should be differenciated based on the category_name variable.

    :return:
    """
    prev_sset = None
    prev_target_qid = None
    list_mentions_in_sset = list()
    # this name might be misleading for category_name=new_entities where the best can be:
    # nr_dist_men_per_sset_calculated
    nr_dist_men_per_cut_per_sset_calculated = 0
    nr_dist_men_per_cut_per_sset_in_df = 0
    nr_covered_mentions = 0
    nr_covered_mentions_of_curr_entity = 0
    set_to_check_no_duplicates = set()

    for idx_row, curr_row in enumerate(df_entities_to_process_w_mentions.itertuples()):

        curr_sset = curr_row.subset_name
        curr_target_wikidata_qid = curr_row.target_wikidata_qid

        is_target_qid_already_in_included = False

        if is_target_qid_already_in_included:
            continue
        curr_time_cut = curr_row.filtered_date

        if curr_time_cut not in dict1_target_pages_to_search:
            dict1_target_pages_to_search[curr_time_cut] = dict()

        if curr_time_cut not in dict2_anchor_pages_to_search:
            dict2_anchor_pages_to_search[curr_time_cut] = dict()

        if prev_sset is None or curr_sset != prev_sset or curr_target_wikidata_qid != prev_target_qid:
            # BEGIN - assert to make sure there is no mixed stuff in the mentions' dataframe
            # (df_entities_to_process_w_mentions)
            to_add_set_no_duplicates = (curr_row.target_wikidata_qid, curr_row.filtered_date, curr_row.subset_name)
            assert to_add_set_no_duplicates not in set_to_check_no_duplicates
            set_to_check_no_duplicates.add(to_add_set_no_duplicates)
            # END - assert

            if len(list_mentions_in_sset) > 0:
                if category_name == 'continual':
                    nr_mentions_per_subset = list_mentions_in_sset[0][
                        'curr_row'].nr_mentions_to_extract_per_cut_per_sset
                elif category_name == 'new':
                    nr_mentions_per_subset = list_mentions_in_sset[0]['curr_row'].nr_mentions_to_extract_per_subset
                else:
                    raise RuntimeError('category_name not recognized! ' + category_name)

                prev_target_qid = list_mentions_in_sset[0]['curr_row'].target_wikidata_qid

                tot_men_extracted = 0
                #
                something_has_been_added = True
                while tot_men_extracted < nr_mentions_per_subset and something_has_been_added:
                    something_has_been_added = False
                    for curr_men_in_subset in list_mentions_in_sset:
                        if tot_men_extracted >= nr_mentions_per_subset:
                            continue
                        max_nr_links_men = curr_men_in_subset['curr_row'].nr_links_mention_per_entity
                        if max_nr_links_men > curr_men_in_subset['nr_mentions_to_extract']:
                            curr_men_in_subset['nr_mentions_to_extract'] += 1
                            tot_men_extracted += 1
                            something_has_been_added = True
                if tot_men_extracted < nr_mentions_per_subset:
                    logger.warning('!!WARNING step 02!! THIS SHOULD NOT HAPPEN, PLEASE TRACK THE BUG!!')

                for curr_men_in_subset in list_mentions_in_sset:
                    curr_anchor_mention = curr_men_in_subset['curr_row'].anchor_mention_text
                    if curr_anchor_mention not in curr_dict_inverse_index or \
                            prev_target_qid not in \
                            curr_dict_inverse_index[curr_anchor_mention]:
                        logger.warning('!!!!WARNING!!!! %s curr_anchor_mention %s or curr_target_wikidata_qid %s'
                                       ' not in inverse dict for cut %s' %
                                       (category_name, curr_anchor_mention, prev_target_qid, curr_time_cut))
                        continue
                    anchor_pages = curr_dict_inverse_index[curr_anchor_mention][prev_target_qid]
                    anchor_pages_lst = list(anchor_pages)
                    random.shuffle(anchor_pages_lst)
                    nr_mentions_covered_dict = {idx_page: None for idx_page, _ in enumerate(anchor_pages_lst)}

                    # this name is weird, basically it is the nr of instances of a particular mention in a set of anchor pages
                    # (anchor_pages_lst)
                    nr_covered_instances_of_curr_mention = 0
                    are_there_more_mentions_in_anchor_pages = True
                    nr_mentions_to_extract = curr_men_in_subset['nr_mentions_to_extract']
                    while are_there_more_mentions_in_anchor_pages and \
                            nr_covered_instances_of_curr_mention < nr_mentions_to_extract:
                        are_there_more_mentions_in_anchor_pages = False
                        for idx_page, curr_anchor_page in enumerate(anchor_pages_lst):
                            # curr_anchor_page[0] --> wikidata qid of anchor page
                            # curr_anchor_page[1] --> nr of mentions curr_anchor_mention pointing to
                            if nr_covered_instances_of_curr_mention >= nr_mentions_to_extract:
                                continue

                            if nr_mentions_covered_dict[idx_page] is None and curr_anchor_page[1] > 0:
                                assert curr_men_in_subset['curr_row'].subset_name == prev_sset
                                #
                                if category_name == 'continual':
                                    nr_links_per_subset = curr_men_in_subset['curr_row'].nr_links_per_cut_per_sset
                                elif category_name == 'new':
                                    nr_links_per_subset = curr_men_in_subset['curr_row'].nr_links_per_subset
                                else:
                                    raise RuntimeError('category_name not recognized! ' + category_name)
                                #
                                if category_name == 'continual':
                                    nr_dist_men_per_cut_per_sset = curr_men_in_subset[
                                        'curr_row'].nr_dist_men_per_cut_per_sset
                                elif category_name == 'new':
                                    nr_dist_men_per_cut_per_sset = curr_men_in_subset['curr_row'].nr_dist_men_per_sset
                                else:
                                    raise RuntimeError('category_name not recognized! ' + category_name)

                                #
                                if category_name == 'continual':
                                    min_nr_links_per_subset = curr_men_in_subset['curr_row'].min_nr_links_per_subset
                                elif category_name == 'new':
                                    min_nr_links_per_subset = nr_links_per_subset
                                else:
                                    raise RuntimeError('category_name not recognized! ' + category_name)

                                entity_prior = None
                                if 'entity_prior' in set(dir(curr_men_in_subset['curr_row'])):
                                    entity_prior = curr_men_in_subset['curr_row'].entity_prior
                                to_append_dict2_anchor = {
                                    'nr_mentions': 1,
                                    'mention': curr_anchor_mention,
                                    # TODO: in case of taking into account ALL the redirects, this 'target_orig_title'
                                    #  will have to become a set of all possible names at this curr_time_cut
                                    'target_orig_title': curr_men_in_subset['curr_row'].target_wikipedia_title_orig,
                                    'target_title_2022': curr_men_in_subset['curr_row'].target_title_2022,
                                    'target_page_id': curr_men_in_subset['curr_row'].target_page_id,
                                    'target_wikidata_qid': curr_men_in_subset['curr_row'].target_wikidata_qid,
                                    'filtered_date': curr_men_in_subset['curr_row'].filtered_date,
                                    # count_mentions: total number this mention appears in wikipedia (check, some
                                    # filters might be applied before calculating it)
                                    'total_count_mentions': curr_men_in_subset['curr_row'].count_mentions,
                                    'nr_links_mention_per_entity':
                                        curr_men_in_subset['curr_row'].nr_links_mention_per_entity,
                                    'prior': curr_men_in_subset['curr_row'].prior,
                                    'prior_rank': curr_men_in_subset['curr_row'].prior_rank,
                                    'edit_distance_mention_title':
                                        curr_men_in_subset['curr_row'].edit_distance_mention_title,
                                    'ratio_edit_distance_mention_title':
                                        curr_men_in_subset['curr_row'].ratio_edit_distance_mention_title,
                                    'edit_distance_mention_title_ent':
                                        curr_men_in_subset['curr_row'].edit_distance_mention_title_ent,
                                    'ratio_edit_distance_mention_title_ent':
                                        curr_men_in_subset['curr_row'].ratio_edit_distance_mention_title_ent,
                                    'overlap_type': curr_men_in_subset['curr_row'].overlap_type,
                                    'avg_prior_rank': curr_men_in_subset['curr_row'].avg_prior_rank,
                                    'avg_prior': curr_men_in_subset['curr_row'].avg_prior,
                                    'nr_inlinks': curr_men_in_subset['curr_row'].nr_inlinks,
                                    'nr_inlinks_after_filter': curr_men_in_subset['curr_row'].nr_inlinks_after_filter,
                                    'nr_distinct_mentions': curr_men_in_subset['curr_row'].nr_distinct_mentions,
                                    'subset': curr_men_in_subset['curr_row'].subset_name,
                                    # here if "random" then "nr_links_per_cut_per_sset", if new_entities, then 'nr_links_per_subset'
                                    #
                                    'nr_links_per_subset': nr_links_per_subset,
                                    'min_nr_links_per_subset': min_nr_links_per_subset,
                                    'nr_dist_men_per_cut_per_sset': nr_dist_men_per_cut_per_sset,
                                    'nr_mentions_to_extract_per_subset': nr_mentions_per_subset,
                                    ###### begin new ones (01/05/2022)
                                    'entity_prior': entity_prior,
                                    ###### end new ones (01/05/2022)
                                    'category': category_name,
                                    'anchor_wikidata_qid': curr_anchor_page[0]
                                }
                                nr_mentions_covered_dict[idx_page] = to_append_dict2_anchor
                                are_there_more_mentions_in_anchor_pages = True
                                nr_covered_mentions += 1
                                nr_covered_instances_of_curr_mention += 1
                                nr_covered_mentions_of_curr_entity += 1
                            elif nr_mentions_covered_dict[idx_page] is not None and \
                                    curr_anchor_page[1] > nr_mentions_covered_dict[idx_page]['nr_mentions']:
                                nr_mentions_covered_dict[idx_page]['nr_mentions'] += 1
                                are_there_more_mentions_in_anchor_pages = True
                                nr_covered_mentions += 1
                                nr_covered_instances_of_curr_mention += 1
                                nr_covered_mentions_of_curr_entity += 1
                    nr_appended = 0
                    for _, to_append_dict2_anchor in nr_mentions_covered_dict.items():
                        if to_append_dict2_anchor is None:
                            continue

                        if to_append_dict2_anchor['anchor_wikidata_qid'] not in \
                                dict2_anchor_pages_to_search[curr_time_cut]:
                            dict2_anchor_pages_to_search[curr_time_cut][
                                to_append_dict2_anchor['anchor_wikidata_qid']] = list()
                        dict2_anchor_pages_to_search[curr_time_cut][to_append_dict2_anchor['anchor_wikidata_qid']] \
                            .append(to_append_dict2_anchor)
                        nr_appended += 1

            list_mentions_in_sset = list()
            prev_sset = curr_sset
            prev_target_qid = curr_target_wikidata_qid

            # just checks that the nrs of distinct mentions are actually the same
            if nr_dist_men_per_cut_per_sset_calculated != nr_dist_men_per_cut_per_sset_in_df:
                logger.warning('!!WARNING nr_dist_men_per_cut_per_sset_calculated != '
                               'nr_dist_men_per_cut_per_sset_in_df !! %s'
                               ' %s vs %s' % (category_name,
                                              nr_dist_men_per_cut_per_sset_calculated,
                                              nr_dist_men_per_cut_per_sset_in_df))
            assert nr_dist_men_per_cut_per_sset_calculated == nr_dist_men_per_cut_per_sset_in_df
            nr_dist_men_per_cut_per_sset_calculated = 0
            nr_covered_mentions_of_curr_entity = 0

        if category_name == 'continual':
            nr_dist_men_per_cut_per_sset_in_df = curr_row.nr_dist_men_per_cut_per_sset
        elif category_name == 'new':
            nr_dist_men_per_cut_per_sset_in_df = curr_row.nr_dist_men_per_sset
        else:
            raise RuntimeError('category_name not recognized: ' + category_name)
        list_mentions_in_sset.append({'curr_row': curr_row, 'nr_mentions_to_extract': 0})

        nr_dist_men_per_cut_per_sset_calculated += 1

        if curr_time_cut not in dict2_anchor_pages_to_search:
            dict2_anchor_pages_to_search[curr_time_cut] = dict()

        if curr_target_wikidata_qid not in dict1_target_pages_to_search[curr_time_cut]:
            dict1_target_pages_to_search[curr_time_cut][curr_target_wikidata_qid] = {
                'subset': [curr_sset],
                'category': [category_name],
                'avg_prior_rank': curr_row.avg_prior_rank,
                'avg_prior': curr_row.avg_prior,
                'nr_inlinks': curr_row.nr_inlinks
            }

        else:
            # just in case adds, should not have more than 1 in each subset/category because a particular target
            # entity can not be used in different subsets / categories
            #   this is O(N)!, but should not have more than 3 in subset
            if curr_sset not in dict1_target_pages_to_search[curr_time_cut][curr_target_wikidata_qid]['subset']:
                dict1_target_pages_to_search[curr_time_cut][curr_target_wikidata_qid]['subset'].append(curr_sset)
            #   this is O(N)!, but should not have more than 5 or so in category
            if category_name not in dict1_target_pages_to_search[curr_time_cut][curr_target_wikidata_qid]['category']:
                dict1_target_pages_to_search[curr_time_cut][curr_target_wikidata_qid]['category'].append(category_name)
    ######## TODO - BEGIN repeated logic - see if can be simplified
    if len(list_mentions_in_sset) > 0:
        prev_target_qid = list_mentions_in_sset[0]['curr_row'].target_wikidata_qid
        if category_name == 'continual':
            nr_mentions_per_subset = list_mentions_in_sset[0]['curr_row'].nr_mentions_to_extract_per_cut_per_sset
        elif category_name == 'new':
            nr_mentions_per_subset = list_mentions_in_sset[0]['curr_row'].nr_mentions_to_extract_per_subset
        else:
            raise RuntimeError('category_name not recognized! ' + category_name)

        tot_men_extracted = 0
        something_has_been_added = True
        while tot_men_extracted < nr_mentions_per_subset and something_has_been_added:
            something_has_been_added = False
            for curr_men_in_subset in list_mentions_in_sset:
                if tot_men_extracted >= nr_mentions_per_subset:
                    continue
                max_nr_links_men = curr_men_in_subset['curr_row'].nr_links_mention_per_entity
                if max_nr_links_men > curr_men_in_subset['nr_mentions_to_extract']:
                    curr_men_in_subset['nr_mentions_to_extract'] += 1
                    tot_men_extracted += 1
                    something_has_been_added = True
        if tot_men_extracted < nr_mentions_per_subset:
            logger.warning('!!WARNING step 02!! THIS SHOULD NOT HAPPEN, PLEASE TRACK THE BUG!!')

        for curr_men_in_subset in list_mentions_in_sset:
            curr_anchor_mention = curr_men_in_subset['curr_row'].anchor_mention_text
            if curr_anchor_mention not in curr_dict_inverse_index or \
                    prev_target_qid not in \
                    curr_dict_inverse_index[curr_anchor_mention]:
                logger.warning('!!!!WARNING!!!! %s curr_anchor_mention %s or curr_target_wikidata_qid %s'
                               ' not in inverse dict for cut %s' %
                               (category_name, curr_anchor_mention, prev_target_qid,
                                curr_men_in_subset['curr_row'].filtered_date))
                continue
            anchor_pages = curr_dict_inverse_index[curr_anchor_mention][prev_target_qid]
            anchor_pages_lst = list(anchor_pages)
            random.shuffle(anchor_pages_lst)
            nr_mentions_covered_dict = {idx_page: None for idx_page, _ in enumerate(anchor_pages_lst)}

            # this name is weird, basically it is the nr of instances of a particular mention in a set of anchor pages
            # (anchor_pages_lst)
            nr_covered_instances_of_curr_mention = 0
            are_there_more_mentions_in_anchor_pages = True
            nr_mentions_to_extract = curr_men_in_subset['nr_mentions_to_extract']
            while are_there_more_mentions_in_anchor_pages and \
                    nr_covered_instances_of_curr_mention < nr_mentions_to_extract:
                are_there_more_mentions_in_anchor_pages = False
                for idx_page, curr_anchor_page in enumerate(anchor_pages_lst):
                    # curr_anchor_page[0] --> wikidata qid of anchor page
                    # curr_anchor_page[1] --> nr of mentions curr_anchor_mention pointing to
                    if nr_covered_instances_of_curr_mention >= nr_mentions_to_extract:
                        continue

                    if nr_mentions_covered_dict[idx_page] is None and curr_anchor_page[1] > 0:
                        assert curr_men_in_subset['curr_row'].subset_name == prev_sset
                        #
                        if category_name == 'continual':
                            nr_links_per_subset = curr_men_in_subset['curr_row'].nr_links_per_cut_per_sset
                        elif category_name == 'new':
                            nr_links_per_subset = curr_men_in_subset['curr_row'].nr_links_per_subset
                        else:
                            raise RuntimeError('category_name not recognized! ' + category_name)
                        #
                        if category_name == 'continual':
                            nr_dist_men_per_cut_per_sset = curr_men_in_subset['curr_row'].nr_dist_men_per_cut_per_sset
                        elif category_name == 'new':
                            nr_dist_men_per_cut_per_sset = curr_men_in_subset['curr_row'].nr_dist_men_per_sset
                        else:
                            raise RuntimeError('category_name not recognized! ' + category_name)
                        #
                        if category_name == 'continual':
                            min_nr_links_per_subset = curr_men_in_subset['curr_row'].min_nr_links_per_subset
                        elif category_name == 'new':
                            min_nr_links_per_subset = nr_links_per_subset
                        else:
                            raise RuntimeError('category_name not recognized! ' + category_name)

                        entity_prior = None
                        if 'entity_prior' in set(dir(curr_men_in_subset['curr_row'])):
                            entity_prior = curr_men_in_subset['curr_row'].entity_prior

                        to_append_dict2_anchor = {
                            'nr_mentions': 1,
                            'mention': curr_anchor_mention,
                            # TODO: in case of taking into account ALL the redirects, this 'target_orig_title'
                            #  will have to become a set of all possible names at this curr_time_cut
                            'target_orig_title': curr_men_in_subset['curr_row'].target_wikipedia_title_orig,
                            'target_title_2022': curr_men_in_subset['curr_row'].target_title_2022,
                            'target_page_id': curr_men_in_subset['curr_row'].target_page_id,
                            'target_wikidata_qid': curr_men_in_subset['curr_row'].target_wikidata_qid,
                            'filtered_date': curr_men_in_subset['curr_row'].filtered_date,
                            # count_mentions: total number this mention appears in wikipedia (check, some
                            # filters might be applied before calculating it)
                            'total_count_mentions': curr_men_in_subset['curr_row'].count_mentions,
                            'nr_links_mention_per_entity':
                                curr_men_in_subset['curr_row'].nr_links_mention_per_entity,
                            'prior': curr_men_in_subset['curr_row'].prior,
                            'prior_rank': curr_men_in_subset['curr_row'].prior_rank,
                            # curr_men_in_subset['curr_row']
                            'edit_distance_mention_title':
                                curr_men_in_subset['curr_row'].edit_distance_mention_title,
                            'ratio_edit_distance_mention_title':
                                curr_men_in_subset['curr_row'].ratio_edit_distance_mention_title,
                            'edit_distance_mention_title_ent':
                                curr_men_in_subset['curr_row'].edit_distance_mention_title_ent,
                            'ratio_edit_distance_mention_title_ent':
                                curr_men_in_subset['curr_row'].ratio_edit_distance_mention_title_ent,
                            'overlap_type': curr_men_in_subset['curr_row'].overlap_type,
                            'avg_prior_rank': curr_men_in_subset['curr_row'].avg_prior_rank,
                            'avg_prior': curr_men_in_subset['curr_row'].avg_prior,
                            'nr_inlinks': curr_men_in_subset['curr_row'].nr_inlinks,
                            'nr_inlinks_after_filter': curr_men_in_subset['curr_row'].nr_inlinks_after_filter,
                            'nr_distinct_mentions': curr_men_in_subset['curr_row'].nr_distinct_mentions,
                            # 'count_mentions': curr_men_in_subset['curr_row'].count_mentions,
                            'subset': curr_men_in_subset['curr_row'].subset_name,
                            # here if "random" then "nr_links_per_cut_per_sset", if new_entities, then 'nr_links_per_subset'
                            #
                            ###### begin new ones (01/05/2022)
                            # 'bucket_name': curr_men_in_subset['curr_row'].bucket_name,
                            'entity_prior': entity_prior,
                            ###### end new ones (01/05/2022)
                            #
                            'nr_links_per_subset': nr_links_per_subset,
                            'min_nr_links_per_subset': min_nr_links_per_subset,
                            'nr_dist_men_per_cut_per_sset': nr_dist_men_per_cut_per_sset,
                            'nr_mentions_to_extract_per_subset': nr_mentions_per_subset,
                            'category': category_name,
                            'anchor_wikidata_qid': curr_anchor_page[0]
                        }
                        nr_mentions_covered_dict[idx_page] = to_append_dict2_anchor
                        are_there_more_mentions_in_anchor_pages = True
                        nr_covered_mentions += 1
                        nr_covered_instances_of_curr_mention += 1
                        nr_covered_mentions_of_curr_entity += 1
                    elif nr_mentions_covered_dict[idx_page] is not None and \
                            curr_anchor_page[1] > nr_mentions_covered_dict[idx_page]['nr_mentions']:
                        nr_mentions_covered_dict[idx_page]['nr_mentions'] += 1
                        are_there_more_mentions_in_anchor_pages = True
                        nr_covered_mentions += 1
                        nr_covered_instances_of_curr_mention += 1
                        nr_covered_mentions_of_curr_entity += 1
            nr_appended = 0
            for _, to_append_dict2_anchor in nr_mentions_covered_dict.items():
                if to_append_dict2_anchor is None:
                    continue
                if to_append_dict2_anchor['anchor_wikidata_qid'] not in dict2_anchor_pages_to_search[curr_time_cut]:
                    dict2_anchor_pages_to_search[curr_time_cut][
                        to_append_dict2_anchor['anchor_wikidata_qid']] = list()
                dict2_anchor_pages_to_search[curr_time_cut][to_append_dict2_anchor['anchor_wikidata_qid']] \
                    .append(to_append_dict2_anchor)
                nr_appended += 1
    ######## TODO - END repeated logic - see if can be simplified

    return dict1_target_pages_to_search, dict2_anchor_pages_to_search


def get_random_one():
    return 1.0 if random.random() < 0.5 else -1.0


def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)


def apply_filters(filter_config, df_to_filter: pd.DataFrame, description_filter,
                  wikidata_qids_with_redirects_in_history: Set, debug=False):
    param_min_inlinks = filter_config['min_inlinks']
    param_max_avg_prior_entity = filter_config['max_avg_prior_entity']
    param_max_prior_mention = filter_config['max_prior_mention']
    param_min_prior_mention = filter_config['min_prior_mention']
    param_min_avg_prior_rank_entity = filter_config['min_avg_prior_rank_entity']
    param_min_prior_rank_mention = filter_config['min_prior_rank_mention']
    min_prior_rank_mention_wrt_men = filter_config['min_prior_rank_mention_wrt_men']
    param_min_edit_distance_mention_title = filter_config['min_edit_distance_mention_title']
    param_min_edit_distance_mention_title_ratio = filter_config['min_edit_distance_mention_title_ratio']
    param_min_same_surface_form_mentions_to_entity = filter_config['min_same_surface_form_mentions_to_entity']

    start = time.time()
    logger.info('df_to_filter in %s' % str(description_filter))
    logger.info('%s is df_to_filter.shape before starting filtering' % str(df_to_filter.shape))
    #
    df_to_filter = df_to_filter[df_to_filter['anchor_mention_text'].notna()]
    logger.info('%s is df_to_filter.shape after df_to_filter[\'anchor_mention_text\'].notna()' %
                str(df_to_filter.shape))
    #
    # Index(['anchor_mention_text', 'target_wikidata_qid',
    #        'target_wikipedia_title_orig', 'target_page_id', 'filtered_date',
    #        'count_mentions', 'nr_links_mention_per_entity', 'prior',
    #        'target_title_2022', 'prior_rank', 'target_wikipedia_title_orig_lower',
    #        'edit_distance_mention_title', 'ratio_edit_distance_mention_title',
    #        'avg_prior_rank', 'avg_prior', 'nr_inlinks'],
    #       dtype='object')
    #
    df_to_filter = df_to_filter[df_to_filter['avg_prior'] <= param_max_avg_prior_entity]
    logger.info('%s is df_to_filter.shape after avg_prior' % str(df_to_filter.shape))
    # param_min_inlinks
    df_to_filter = df_to_filter[df_to_filter['nr_inlinks'] >= param_min_inlinks]
    logger.info('%s is df_to_filter.shape after nr_inlinks' % str(df_to_filter.shape))

    df_to_filter = df_to_filter[df_to_filter['prior'] >= param_min_prior_mention]
    logger.info('%s is df_to_filter.shape after MIN prior' % str(df_to_filter.shape))
    # param_min_avg_prior_rank_entity
    df_to_filter = df_to_filter[df_to_filter['avg_prior_rank'] >= param_min_avg_prior_rank_entity]
    logger.info('%s is df_to_filter.shape after avg_prior_rank' % str(df_to_filter.shape))
    # param_min_prior_rank_mention
    df_to_filter = df_to_filter[df_to_filter['prior_rank'] >= param_min_prior_rank_mention]
    logger.info('%s is df_to_filter.shape after prior_rank' % str(df_to_filter.shape))

    df_to_filter = df_to_filter[df_to_filter['prior_rank_men'] >= min_prior_rank_mention_wrt_men]
    logger.info('%s is df_to_filter.shape after prior_rank_men' % str(df_to_filter.shape))

    # param_max_prior_mention
    df_to_filter = df_to_filter[df_to_filter['prior'] <= param_max_prior_mention]
    logger.info('%s is df_to_filter.shape after MAX prior' % str(df_to_filter.shape))

    # param_min_edit_distance_mention_title
    df_to_filter = df_to_filter[
        df_to_filter['edit_distance_mention_title_ent'] >= param_min_edit_distance_mention_title]
    logger.info('%s is df_to_filter.shape after edit_distance_mention_title' % str(df_to_filter.shape))
    #
    # param_min_edit_distance_mention_title_ratio
    df_to_filter = df_to_filter[
        df_to_filter['ratio_edit_distance_mention_title_ent'] >= param_min_edit_distance_mention_title_ratio]
    logger.info('%s is df_to_filter.shape after ratio_edit_distance_mention_title_ent' % str(df_to_filter.shape))

    df_to_filter = df_to_filter[
        df_to_filter['nr_links_mention_per_entity'] >= param_min_same_surface_form_mentions_to_entity]
    #
    # min_len_target_page
    logger.info('%s is df_to_filter.shape after nr_links_mention_per_entity' % str(df_to_filter.shape))

    if not debug:
        # to work locally, need to re-run s02_generate_alias_table_only_target.py
        df_to_filter = df_to_filter[
            df_to_filter['content_length'] >= filter_config['min_len_target_page']]

        logger.info('%s is df_to_filter.shape after min_len_target_page' % str(df_to_filter.shape))

    # BEGIN: filters out all target qids that have redirects in its history, we currently do not deal with redirects
    #  once we deal with redirects, will have to see have to refactor this part
    start_redirects = time.time()
    df_to_filter = df_to_filter[
        df_to_filter.apply(lambda x: x['target_wikidata_qid'] not in wikidata_qids_with_redirects_in_history, axis=1)]
    end_redirects = time.time()
    logger.info('%s is df_to_filter.shape after filtering out target wikidata qids with redirects in history' %
                str(df_to_filter.shape))
    logger.info('%s minutes to filter out target wikidata qids with redirects in history' %
                str((end_redirects - start_redirects) / 60))

    # END: filter of target wikidata qids with redirects
    end = time.time()
    logger.info('%s minutes to apply filters to df_to_filter, the shape now is: %s' %
                ((end - start) / 60, df_to_filter.shape))

    return df_to_filter


def put_mention_in_cluster(subset_clusters_dict: Dict, config: Dict, mention: str):
    # rules:
    #  1- put the mention into the cluster with the lowest edit distance
    #  2- put the mention in the cluster if another mention is contained inside or the other way around
    #     (one_string_in_another returns True).
    #  3- try to maintain the clusters as balanced as possible
    #  4- other??

    subsets_for_which_must = list()
    subsets_for_which_can = dict()
    for curr_subset_name, curr_subset_cluster in subset_clusters_dict.items():
        must_be_this_subset = False
        if mention in curr_subset_cluster:
            subsets_for_which_must.append(curr_subset_name)
            continue
        min_ratio_edit_distance_curr_subset = 1.0
        for curr_subset_mention in curr_subset_cluster:
            try:
                if config['common_filters']['string_in_another_string_when_clustering']:
                    if one_string_in_another(mention, curr_subset_mention) or \
                            one_string_in_another(curr_subset_mention, mention):
                        must_be_this_subset = True
                        continue
                edit_distance, ratio_edit_distance = get_ratio_edit_distance_v2(mention, curr_subset_mention)
                min_ratio_edit_distance_curr_subset = min(min_ratio_edit_distance_curr_subset, ratio_edit_distance)
                if ratio_edit_distance < config['common_filters']['min_edit_distance_ratio_between_subsets']:
                    must_be_this_subset = True
            except:
                logger.error('some sort of error when checking for edit distance between mentions and/or '
                             'one_string_in_another: curr_subset_mention: %s mention: %s curr_subset_cluster: %s'
                             ' subset_clusters_dict: %s' %
                             (curr_subset_mention, mention, curr_subset_cluster, subset_clusters_dict))
                logger.error(traceback.format_exc())
        if must_be_this_subset:
            subsets_for_which_must.append(curr_subset_name)
        if not must_be_this_subset:
            subsets_for_which_can[curr_subset_name] = {'subset_name': curr_subset_name,
                                                       'subset_len': len(curr_subset_cluster)}

    if len(subsets_for_which_must) == 1:
        subset_clusters_dict[subsets_for_which_must[0]].add(mention)
        return subset_clusters_dict, subsets_for_which_must[0]
    elif len(subsets_for_which_must) == 0 and len(subsets_for_which_can) > 0:
        # selects the cluster that has the least nr of distinct mentions and for which ok_for_subset is True
        # and no one_string_in_another in True
        if config['common_filters']['prefer_train_cluster'] and 'train' in subsets_for_which_can:
            subset_clusters_dict['train'].add(mention)
            return subset_clusters_dict, 'train'
        else:
            ok_subset_with_least_elements = sorted(subsets_for_which_can.values(), key=lambda x: x['subset_len'])[0][
                'subset_name']
            subset_clusters_dict[ok_subset_with_least_elements].add(mention)
            return subset_clusters_dict, ok_subset_with_least_elements

    return subset_clusters_dict, None


def process_line_input_file_parallel(config, dict1_target_pages_to_search, dict2_anchor_pages_to_search,
                                     arq: ArticleReadingQueue, v_shutdown_file_processor, start_time,
                                     v_nr_entities, v_nr_entities_found, v_nr_mentions_found):
    # use the queue arq.article_queue
    try:
        bert_tokenizer = BertTokenizer.from_pretrained(
            config['tokenizer']['bert_model'], do_lower_case=config['tokenizer']['lowercase'],
            cache_dir=config['tokenizer']['bert_cache_dir']
        )
        while not (arq.article_queue.empty() and v_shutdown_file_processor.value == 1):
            if v_nr_entities.value % 1000 == 0 and v_nr_entities.value > 0:
                logger.info('nr of loaded entities: %s' % v_nr_entities.value)
                time_lapse_mins = (time.time() - start_time) / 60
                logger.info('Avg. nr of entities per minute: %s' % (v_nr_entities.value / time_lapse_mins))

            json_line_str = arq.article_queue.get(block=True, timeout=30)

            v_nr_entities.value += 1

            json_line = json.loads(json_line_str[0])

            curr_wikidata_qid = json_line['wikidata_qid']

            was_entity_found = False

            curr_wikipedia_title_2022 = json_line['wikipedia_title']
            curr_wikipedia_creation_date = json_line['wikipedia_creation_date']
            curr_wikipedia_page_id = json_line['page_id']
            for curr_time_cut, curr_time_content in json_line['evolution'].items():
                curr_wikipedia_revision_date = curr_time_content['wikipedia_revision_date']
                curr_wikipedia_title = curr_time_content['wikipedia_title'].replace(' ', '_')
                if curr_time_cut in dict1_target_pages_to_search:
                    if curr_wikidata_qid in dict1_target_pages_to_search[curr_time_cut]:
                        if curr_wikidata_qid in dict1_target_pages_to_search[curr_time_cut]:
                            content_to_put_bert = curr_time_content['bert_tokenized']
                            len_text = len(from_bert_to_text(content_to_put_bert))
                            title_to_put_bert = curr_time_content['bert_wikipedia_title']

                            content_to_put = curr_time_content['cleaned_text']
                            title_to_put = curr_time_content['wikipedia_title']
                            curr_target_content = dict1_target_pages_to_search[curr_time_cut][curr_wikidata_qid]
                            arq.process_files_queue.put({
                                'time_cut': curr_time_cut,
                                'is_mention': False,
                                'is_target_page': True,
                                'label_bert_tokenized': content_to_put_bert,
                                'target_len': len_text,
                                # BEGIN: 01/05/2022 new not lowercased
                                'target_text': content_to_put,
                                'target_title': title_to_put,
                                # END: 01/05/2022 new not lowercased
                                'label_title_bert_tokenized': title_to_put_bert,
                                'wikidata_qid': curr_wikidata_qid,
                                'subset': curr_target_content['subset'],
                                'category': curr_target_content['category'],
                                'avg_prior_rank': curr_target_content['avg_prior_rank'],
                                'avg_prior': curr_target_content['avg_prior'],
                                'nr_inlinks': curr_target_content['nr_inlinks'],
                                'target_orig_title_source2': curr_wikipedia_title,
                                'target_title_2022_source2': curr_wikipedia_title_2022,
                                'target_page_id_source2': curr_wikipedia_page_id,
                                'target_wikidata_qid_source2': curr_wikidata_qid,
                                'target_wikipedia_creation_date': curr_wikipedia_creation_date,
                                'target_wikipedia_revision_date': curr_wikipedia_revision_date
                            })

                            was_entity_found = True

                if curr_time_cut in dict2_anchor_pages_to_search:
                    if curr_wikidata_qid in dict2_anchor_pages_to_search[curr_time_cut]:
                        was_entity_found = True
                        for mention_to_extract in dict2_anchor_pages_to_search[curr_time_cut][curr_wikidata_qid]:

                            anchor_mention_text = mention_to_extract['mention']

                            mentions_with_context = \
                                get_random_mentions_with_context(anchor_mention_text,
                                                                 mention_to_extract['target_orig_title'],
                                                                 curr_time_content['wikipedia_source'],
                                                                 curr_time_content['wikipedia_title'],
                                                                 curr_time_cut,
                                                                 config,
                                                                 bert_tokenizer,
                                                                 mention_to_extract['nr_mentions'])
                            if mentions_with_context is None:
                                logger.warning('=================BEGIN WARNING!!=====================')
                                logger.warning('!!!WARNING!!! mentions_with_context in None for some reason: ')
                                logger.warning('\tanchor_mention_text: %s' % anchor_mention_text)
                                logger.warning('\tmention_to_extract[\'target_orig_title\']: %s' %
                                               mention_to_extract['target_orig_title'])
                                logger.warning('\tcurr_time_content[\'wikipedia_source\']: %s' %
                                               curr_time_content['wikipedia_source'])
                                logger.warning('\tcurr_time_content[\'wikipedia_title\']: %s' %
                                               curr_time_content['wikipedia_title'])
                                logger.warning('\tcurr_time_cut: %s' % curr_time_cut)
                                logger.warning('\tnr_mentions to extract: %s' % mention_to_extract['nr_mentions'])
                                logger.warning('=================END WARNING!!=====================')
                                continue
                            elif len(mentions_with_context) != mention_to_extract['nr_mentions']:
                                logger.warning('WARNING! - len(mentions_with_context) %s'
                                               '  different from mention_to_extract[\'nr_mentions\'] %s' %
                                               (len(mentions_with_context), mention_to_extract['nr_mentions']))

                            #
                            mention_to_extract['anchor_orig_title'] = curr_wikipedia_title
                            mention_to_extract['anchor_title_2022'] = curr_wikipedia_title_2022
                            mention_to_extract['anchor_wikipedia_creation_date'] = curr_wikipedia_creation_date
                            mention_to_extract['anchor_wikipedia_revision_date'] = curr_wikipedia_revision_date

                            content_to_put_bert = curr_time_content['bert_tokenized']
                            len_text = len(from_bert_to_text(content_to_put_bert))

                            # added the length of the anchor page in tokens
                            mention_to_extract['anchor_len'] = len_text

                            arq.process_files_queue.put({
                                'time_cut': curr_time_cut,
                                'subset': mention_to_extract['subset'],
                                'is_mention': True,
                                'is_target_page': False,
                                'mention_attributes': mention_to_extract,
                                'orig_context_doc_id': curr_wikidata_qid,
                                'extracted_mention_instances': mentions_with_context
                            })
                            v_nr_mentions_found.value += len(mentions_with_context)
            if was_entity_found:
                v_nr_entities_found.value += 1

    except:
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')
        logger.error('error in process_line_input_file_parallel at %s' % current_time)
        logger.error(traceback.format_exc())
        logger.error('process_line_input_file_parallel, giving a rest of 10 secs')
        time.sleep(10)


def read_from_input_file(input_file_path, dict1_target_pages_to_search, dict2_anchor_pages_to_search,
                         arq: ArticleReadingQueue):
    with io.open(input_file_path, mode='rt', encoding='utf-8') as infile:
        for curr_line in infile:
            try:
                arq.article_queue.put((curr_line,))
            except:
                logger.error('error happened in read_from_input_file when trying to put the line with length of %s'
                             ' I am going to try couple of more times and then abandon:' % len(curr_line))
                logger.error(traceback.format_exc())
                for i in range(5):
                    time.sleep(10)
                    logger.error('RETRY NUMBER %s' % i)
                    arq.article_queue.put((curr_line,))
                    break


def write_to_output_file(arq: ArticleReadingQueue, v_shutdown_writer, config, all_subset_names):
    base_experiment_path = config['base_experiment_path']
    output_dir = config['output_dir']
    output_dir = os.path.join(base_experiment_path, output_dir)

    # removes all the directory and makes it again
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    time_cuts = config['time_cuts']

    type_to_cut_to_subset_to_file = dict()
    type_to_cut_to_subset_to_file['mention'] = dict()
    type_to_cut_to_subset_to_file['entity'] = dict()

    for curr_time_cut in time_cuts:
        type_to_cut_to_subset_to_file['mention'][curr_time_cut] = dict()
        file_entity_path = os.path.join(output_dir, '{}_entities.jsonl'.format(curr_time_cut))
        file_entity = open(file_entity_path, 'a', encoding='utf8')
        type_to_cut_to_subset_to_file['entity'][curr_time_cut] = file_entity
        for curr_subset_name in all_subset_names:
            file_mention_path = os.path.join(output_dir, '{}_{}_mentions.jsonl'.format(curr_time_cut,
                                                                                       curr_subset_name))
            file_mention = open(file_mention_path, 'a', encoding='utf8')
            type_to_cut_to_subset_to_file['mention'][curr_time_cut][curr_subset_name] = file_mention

    while not (arq.process_files_queue.empty() and v_shutdown_writer.value == 1):
        try:
            json_w_content_to_write = arq.process_files_queue.get(block=True, timeout=30)
            time_cut = json_w_content_to_write['time_cut']
            subset = json_w_content_to_write['subset']

            if json_w_content_to_write['is_mention']:
                mention_attributes = json_w_content_to_write['mention_attributes']

                for curr_mention_instance in json_w_content_to_write['extracted_mention_instances']:
                    copied_attrs_to_write = mention_attributes.copy()
                    del copied_attrs_to_write['nr_mentions']
                    copied_attrs_to_write['context_right_bert_tokenized'] = \
                        curr_mention_instance['context_right_bert_tokenized']
                    copied_attrs_to_write['context_left_bert_tokenized'] = \
                        curr_mention_instance['context_left_bert_tokenized']
                    copied_attrs_to_write['mention_bert_tokenized'] = curr_mention_instance[
                        'mention_bert_tokenized']

                    copied_attrs_to_write['context_right'] = curr_mention_instance['context_right']
                    copied_attrs_to_write['context_left'] = curr_mention_instance['context_left']
                    copied_attrs_to_write['mention'] = curr_mention_instance['mention']

                    type_to_cut_to_subset_to_file['mention'][time_cut][subset] \
                        .write(json.dumps(copied_attrs_to_write, ensure_ascii=False) + '\n')
            else:
                # a target entity can only participate in a single subset
                to_write = json_w_content_to_write.copy()
                del to_write['is_mention']
                del to_write['is_target_page']
                type_to_cut_to_subset_to_file['entity'][time_cut] \
                    .write(json.dumps(to_write, ensure_ascii=False) + '\n')

        except:
            now = datetime.now()
            current_time = now.strftime('%H:%M:%S')
            logger.error('error in write_to_output_file at %s' % current_time)
            logger.error(traceback.format_exc())
            logger.error('write_to_output_file, giving a rest of 10 secs')
            time.sleep(10)

    for curr_time_cut in time_cuts:
        type_to_cut_to_subset_to_file['entity'][curr_time_cut].flush()
        type_to_cut_to_subset_to_file['entity'][curr_time_cut].close()
        for curr_subset_name in all_subset_names:
            type_to_cut_to_subset_to_file['mention'][curr_time_cut][curr_subset_name].flush()
            type_to_cut_to_subset_to_file['mention'][curr_time_cut][curr_subset_name].close()


def display(process_write_output_file: List, process_file: List, v_nr_entities, v_nr_entities_found,
            v_nr_mentions_found, arq: ArticleReadingQueue, start_time,
            len_entities_dict1, len_entities_dict2, len_entities_both):
    while True:
        try:
            logger.info('\tDISPLAY Size of queue article_queue: %s' % arq.article_queue.qsize())
            logger.info('\tDISPLAY Size of queue process_files_queue: %s' % arq.process_files_queue.qsize())
            is_alive = 0
            is_dead = 0
            for curr_process_file in process_file:
                if curr_process_file.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1
            logger.info('\tDISPLAY NR file processing processes alive: %s and dead: %s' % (is_alive, is_dead))
            is_alive = 0
            is_dead = 0
            for curr_process_file in process_write_output_file:
                if curr_process_file.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1

            logger.info('\tDISPLAY NR process write output file: %s and dead: %s' % (is_alive, is_dead))

            curr_time = time.time()
            logger.info('\tDISPLAY TOT processed entities: %s Avg. articles per minute: %s' %
                        (v_nr_entities.value, v_nr_entities.value / ((curr_time - start_time) / 60)))
            logger.info('\tDISPLAY TOT entities found: %s' % v_nr_entities_found.value)
            logger.info('\tDISPLAY TOT mentions found (across ALL the time cuts): %s' % v_nr_mentions_found.value)
            logger.info('\tDISPLAY TOT entities dict1_target_pages_to_search: %s' % len_entities_dict1)
            logger.info('\tDISPLAY TOT entities dict2_anchor_pages_to_search: %s' % len_entities_dict2)
            logger.info('\tDISPLAY TOT entities both dicts: %s' % len_entities_both)
            time.sleep(60 * 1)
        except Exception as e:
            logger.error('type error display: ' + str(e))
            logger.error(traceback.format_exc())
            continue
