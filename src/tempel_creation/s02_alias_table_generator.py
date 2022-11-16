import argparse
import json
import logging
import os
import pickle
import re
import time
import traceback

import pandas as pd

from tempel_creation.misc.load_wiki_sql_tables import load_wiki_page_id_to_wikidata_qid
from tempel_creation.misc.utils import get_ratio_edit_distance_v2
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def get_wikidata_qid(wikipedia_page_id_to_wikidata_qid, row):
    wikidata_qid = wikipedia_page_id_to_wikidata_qid.get(row['page_id'])

    if wikidata_qid is None:
        wikidata_qid = '-1'
    return wikidata_qid


def one_string_in_another(string1, string2):
    if string1 in string2:
        l_index = string2.index(string1)
        r_index = l_index + len(string1)
        l_ok = False
        r_ok = False
        # if it is a separate word/phrase inside the title, set AMBIGUOUS_SUBSTRING
        if l_index == 0 or string2[l_index - 1] == ' ':
            l_ok = True

        if r_index == len(string2) or string2[r_index] == ' ' or string2[r_index:].startswith(', '):
            r_ok = True

        if r_ok and l_ok:
            return True
        else:
            return False
    else:
        return False


def get_overlap_mention_entity_title(anchor_mention_text, target_wikipedia_title, target_wikipedia_title_orig_lower):
    edit_distance_mention_title, ratio_edit_distance_mention_title = \
        get_ratio_edit_distance_v2(anchor_mention_text, target_wikipedia_title_orig_lower)

    quotation = {'\'', '"'}
    try:
        while anchor_mention_text[0] in quotation and len(anchor_mention_text) > 2 and \
                anchor_mention_text[-1] == anchor_mention_text[0]:
            anchor_mention_text = anchor_mention_text[1:-1]
    except:
        logger.error('error when doing the quotation loop for the following mention: %s' % anchor_mention_text)
        logger.error(traceback.format_exc())
    disamb_phrase = ''

    if target_wikipedia_title[-1] == ')' and '_(' in target_wikipedia_title:
        rindex_disamb = target_wikipedia_title.rindex('_(')
        disamb_phrase = target_wikipedia_title[rindex_disamb + 2: -1]
        ent_phrase = target_wikipedia_title[:rindex_disamb].lower().replace('_', ' ')
    else:
        ent_phrase = target_wikipedia_title.lower().replace('_', ' ')
    anchor_mention_text = anchor_mention_text.lower()
    anchor_mention_text = anchor_mention_text.strip()
    ent_phrase = ent_phrase.strip()
    disamb_phrase = disamb_phrase.strip()

    edit_distance_mention_title_ent, ratio_edit_distance_mention_title_ent = \
        get_ratio_edit_distance_v2(anchor_mention_text, ent_phrase)

    if len(disamb_phrase) == 0 and anchor_mention_text == ent_phrase:
        overlap_type = 'HIGH_OVERLAP'
    elif len(disamb_phrase) > 0 and anchor_mention_text == ent_phrase:
        overlap_type = 'MULTIPLE_CATEGORIES'
    elif one_string_in_another(anchor_mention_text, ent_phrase) or \
            one_string_in_another(ent_phrase, anchor_mention_text):
        overlap_type = 'AMBIGUOUS_SUBSTRING'
    else:
        overlap_type = 'LOW_OVERLAP'

    return pd.Series([edit_distance_mention_title, edit_distance_mention_title_ent,
                      ratio_edit_distance_mention_title, ratio_edit_distance_mention_title_ent,
                      overlap_type])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/snapshot_extraction/snap_20220515/config/s02_alias_table_generator.json',
                        help='The config file that contains all the parameters')

    args = parser.parse_args()
    logger.info('generating alias table on the following parameters: %s' % args)
    logger.info('getting the top wikidata changes: %s' % args)
    config = json.load(open(args.config_file, 'rt'))

    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(config['output_dir_git_hash'], 'commit_hash_alias_table.txt'), 'wt') as outfile:
        outfile.write(sha)

    input_dir = config['input_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    time_cut_list = config['time_cut_list']
    title_changes_file_name = config['title_changes_file_name']
    prefix_page_link_stats = config['prefix_page_link_stats']
    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    caches_dir = config['caches_dir']
    prefix_page_info = config['prefix_page_info']

    os.makedirs(os.path.dirname(caches_dir), exist_ok=True)

    regex_pat1 = re.compile(r'\]\]$', flags=re.IGNORECASE)
    regex_pat2 = re.compile(r'\]$', flags=re.IGNORECASE)
    input_title_changes_path = os.path.join(input_dir, title_changes_file_name)
    logger.info('start loading the input_title_changes')
    start = time.time()
    df_title_changes = pd.read_csv(open(input_title_changes_path, 'rt'), sep='\t', header=[0])
    end = time.time()
    logger.info('input_title_changes loaded in %s minutes' % ((end - start) / 60))
    logger.info('extracting current titles (2022-01-01T00:00:00Z)')
    start = time.time()
    df_title_changes_2022 = df_title_changes[df_title_changes['filtered_date'] == '2022-01-01T00:00:00Z']
    df_title_changes_2022.rename(columns={'title': 'title_2022'}, inplace=True)
    end = time.time()
    logger.info('filtered titles of 2022 in %s minutes' % ((end - start) / 60))

    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir,
                                                                'wikipedia_page_id_to_wikidata_qid.pickle')
    wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(path_cache_wikipedia_page_id_to_wikidata_qid,
                                                                          path_wikipedia_wikidata_map)

    df_title_changes.rename(columns={'title': 'target_wikipedia_title_orig'}, inplace=True)

    for curr_time_cut in time_cut_list:
        ##### BEGIN adding page info
        curr_page_info_path = '{}{}.csv'.format(prefix_page_info, curr_time_cut)
        curr_page_info_path = os.path.join(input_dir, curr_page_info_path)

        df_page_info = pd.read_csv(open(curr_page_info_path, 'rt'), sep='\t', header=[0])
        df_page_info['wikipedia_creation_time'] = \
            pd.to_datetime(df_page_info['wikipedia_creation_time'], format='%Y-%m-%dT%H:%M:%SZ')

        logger.info('df_page_info.columns: %s' % df_page_info.columns)
        ##### END adding page info

        output_link_stats_file_name = 'prior_link_stats_{}.csv'.format(curr_time_cut)
        output_link_stats_file_path = os.path.join(output_dir, output_link_stats_file_name)

        output_dict_inverse_index = 'dict_inverse_index_{}.pickle'.format(curr_time_cut)
        output_dict_inverse_index = os.path.join(output_dir, output_dict_inverse_index)

        curr_cut_start = time.time()
        logger.info('================================ starting %s ================================= ' % curr_time_cut)
        input_link_stats_file_name = '{}{}.csv'.format(prefix_page_link_stats, curr_time_cut)

        input_link_stats_file_path = os.path.join(input_dir, input_link_stats_file_name)

        # step 1:  load it into pandas
        start = time.time()
        logger.info('starting read_csv for %s' % input_link_stats_file_path)
        curr_df_link_stats = pd.read_csv(open(input_link_stats_file_path, 'rt'), sep='\t', header=[0])
        end = time.time()
        logger.info('%s minutes took read_csv for %s' % (((end - start) / 60), input_link_stats_file_path))

        # step 2: fix the '[' issue  (iterating over all the panda rows?)
        start = time.time()
        if config['debug']:
            df_bugs0 = curr_df_link_stats[curr_df_link_stats['anchor_mention_text'].isnull()]
            logger.debug('some bugs (null).shape: %s' % str(df_bugs0.shape))
            curr_df_link_stats = curr_df_link_stats[~(curr_df_link_stats['anchor_mention_text'].isnull())]
            df_bugs = curr_df_link_stats[
                curr_df_link_stats['anchor_mention_text'].str.contains('^[a-z\-]{2,15}\:', regex=True)]
            logger.debug('some bugs colon ":".shape: %s' % str(df_bugs.shape))
            curr_df_link_stats = curr_df_link_stats[~(curr_df_link_stats['anchor_mention_text'].isnull())]
            df_bugs = curr_df_link_stats[curr_df_link_stats['anchor_mention_text'].str.contains('www\.', regex=True)]
            logger.debug('some bugs .shape: %s' % str(df_bugs.shape))

            df_bugs = curr_df_link_stats[curr_df_link_stats['anchor_mention_text'].str.contains('\]\]$', regex=True)]
            logger.debug('some potential mention bugs (end in ]]).shape: %s' % str(df_bugs.shape))

            df_bugs = curr_df_link_stats[curr_df_link_stats['anchor_mention_text'].str.contains('\]$', regex=True)]
            logger.debug('some potential mention bugs (end in ]).shape: %s' % str(df_bugs.shape))

        logger.info('starting replacing 1 for %s inside curr_df_link_stats with shape %s' %
                    (input_link_stats_file_path, str(curr_df_link_stats.shape)))
        curr_df_link_stats['target_wikipedia_title_orig'] = \
            curr_df_link_stats['target_wikipedia_title_orig'].str.replace(regex_pat1, '')
        curr_df_link_stats['target_wikipedia_title_orig'] = \
            curr_df_link_stats['target_wikipedia_title_orig'].str.replace(regex_pat2, '')
        curr_df_link_stats['target_wikipedia_title_orig'] = \
            curr_df_link_stats['target_wikipedia_title_orig'].str.strip()
        end = time.time()
        logger.info('%s minutes to replacing 1 for %s ' % (((end - start) / 60), input_link_stats_file_path))

        start = time.time()

        logger.info('starting deleting %s language related mentions like "es:hola"' % input_link_stats_file_path)
        curr_df_link_stats = \
            curr_df_link_stats[
                curr_df_link_stats['anchor_mention_text'].str.contains('^[a-z\-]{2,15}\:', regex=True) == False]
        curr_df_link_stats = \
            curr_df_link_stats[
                curr_df_link_stats['anchor_mention_text'].str.contains('www\.', regex=True) == False]
        end = time.time()
        logger.info('%s minutes took deleting %s language related mentions like "es:hola" , '
                    ' the resulting shape of curr_df_link_stats is %s ' %
                    ((end - start) / 60, input_link_stats_file_path, str(curr_df_link_stats.shape)))

        logger.info('also deleting all the cases of target_wikipedia_title_orig List_of_ %s' %
                    input_link_stats_file_path)
        start = time.time()
        curr_df_link_stats = \
            curr_df_link_stats[
                curr_df_link_stats['target_wikipedia_title_orig'].str.contains('^List_of_', regex=True) == False]
        curr_df_link_stats = \
            curr_df_link_stats[
                curr_df_link_stats['anchor_wikipedia_title'].str.contains('^List_of_', regex=True) == False]
        end = time.time()
        logger.info('%s minutes took deleting %s all the cases of target_wikipedia_title_orig List_of_ '
                    'the resulting shape of curr_df_link_stats is %s' %
                    ((end - start) / 60, input_link_stats_file_path, str(curr_df_link_stats.shape)))

        if config['debug']:
            logger.debug('=========== MENTIONS of cut %s' % curr_time_cut)
            for idx, curr_row in curr_df_link_stats.iterrows():
                # if idx >= 1000:
                if idx >= 1:
                    break
                logger.debug('%s - %s' % (curr_row['anchor_wikidata_qid'], curr_row['anchor_mention_text']))

            logger.debug('=========== TITLES of cut %s' % curr_time_cut)
            for idx, curr_row in curr_df_link_stats.iterrows():
                # if idx >= 1000:
                if idx >= 1:
                    break
                logger.debug('%s - %s' % (curr_row['anchor_wikidata_qid'], curr_row['target_wikipedia_title_orig']))
            logger.debug('=========================')

        start = time.time()
        logger.debug('start stripping and lowering')
        curr_df_link_stats['anchor_mention_text'] = curr_df_link_stats['anchor_mention_text'].str.strip()
        curr_df_link_stats['anchor_mention_text'] = curr_df_link_stats['anchor_mention_text'].str.lower()
        end = time.time()
        logger.info('%s minutes took strip and lower' % ((end - start) / 60))

        start = time.time()
        curr_df_dict_index = curr_df_link_stats[['anchor_mention_text',
                                                 'anchor_wikidata_qid',
                                                 'target_wikipedia_title_orig',
                                                 'filtered_date']]
        curr_df_dict_index = curr_df_dict_index.groupby(['anchor_mention_text',
                                                         'anchor_wikidata_qid',
                                                         'target_wikipedia_title_orig',
                                                         'filtered_date']) \
            .size().reset_index(name='count_mentions_entities_anchorqid')

        end = time.time()
        logger.info('%s minutes took group by anchor wikidata qid and target title orig' % ((end - start) / 60))

        logger.info('beginning the merge of curr_df_dict_index of shape %s '
                    ' with df_title_changes of shape %s' % (str(curr_df_dict_index.shape), str(df_title_changes.shape)))
        start = time.time()
        curr_df_dict_index = pd.merge(curr_df_dict_index, df_title_changes,
                                      left_on=('target_wikipedia_title_orig', 'filtered_date'),
                                      right_on=('target_wikipedia_title_orig', 'filtered_date'), how='inner')

        end = time.time()
        logger.info('%s minutes took merging big anchor wikidata qid and target title changes, '
                    'now curr_df_dict_index.shape is %s' % ((end - start) / 60, str(curr_df_dict_index.shape)))

        # only leave the information of target pages, summing over all the mentions
        logger.info('beginning the groupby by target only, initial curr_df_link_stats.shape is %s' %
                    str(curr_df_link_stats.shape))
        curr_df_link_stats = curr_df_link_stats[['anchor_mention_text',
                                                 'target_wikipedia_title_orig',
                                                 'filtered_date']]

        start = time.time()
        logger.info('The curr_df_link_stats.shape before merging with df_title_changes is %s' %
                    str(curr_df_link_stats.shape))
        curr_df_link_stats = pd.merge(curr_df_link_stats, df_title_changes,
                                      left_on=('target_wikipedia_title_orig', 'filtered_date'),
                                      right_on=('target_wikipedia_title_orig', 'filtered_date'), how='inner')

        end = time.time()
        logger.info('%s minutes took merge between curr_df_link_stats and df_title_changes, the '
                    'curr_df_link_stats.shape now is %s ' % ((end - start) / 60, str(curr_df_link_stats.shape)))
        start = time.time()
        curr_df_link_stats = curr_df_link_stats.groupby(
            ['anchor_mention_text',
             'target_wikipedia_title_orig',
             'filtered_date',
             'page_id']).size().reset_index(name='count_mentions_entities')

        end = time.time()
        logger.info('%s minutes took groupby by target only, curr_df_link_stats.shape is %s' %
                    ((end - start) / 60, str(curr_df_link_stats.shape)))

        # gets the page_ids and qids according to the CURRENT (BEGINNING OF 2022!!!)
        # the ones for which the current qid can not be gotten, are just ignored!!!

        start = time.time()

        logger.info('beginning get_wikidata_qid on curr_df_link_stats with shape of %s' % str(curr_df_link_stats.shape))
        curr_df_link_stats['target_wikidata_qid'] = \
            curr_df_link_stats.apply(lambda row: get_wikidata_qid(wikipedia_page_id_to_wikidata_qid, row), axis=1)
        end = time.time()
        logger.info('%s minutes took get_wikidata_qid' % ((end - start) / 60))
        df_not_found_qid = curr_df_link_stats[curr_df_link_stats['target_wikidata_qid'] == '-1']
        logger.info('shape of not found qid dataframe is: %s' % str(df_not_found_qid.shape))
        curr_df_link_stats = curr_df_link_stats[curr_df_link_stats['target_wikidata_qid'] != '-1']

        # now that we have qids, we merge directly by page_id with the index df to get that qids there
        start = time.time()
        logger.info('shape curr_df_dict_index before merge with curr_df_link_stats for wikidata_qid: %s' %
                    str(curr_df_dict_index.shape))
        curr_df_dict_index = pd.merge(curr_df_dict_index,
                                      curr_df_link_stats.drop_duplicates(['page_id'])[
                                          ['page_id', 'target_wikidata_qid']],
                                      on=('page_id',), how='inner')
        end = time.time()
        logger.info('%s minutes took merge with curr_df_link_stats for wikidata_qid' % ((end - start) / 60))
        logger.info('shape curr_df_dict_index after merge with curr_df_link_stats for wikidata_qid: %s' %
                    str(curr_df_dict_index.shape))

        start = time.time()

        dict_index_mentions = dict()

        for row in curr_df_dict_index.itertuples():
            anchor_mentiont_text = row.anchor_mention_text
            target_wikidata_qid = row.target_wikidata_qid
            anchor_wikidata_qid = row.anchor_wikidata_qid
            count_mentions_entities_anchorqid = row.count_mentions_entities_anchorqid
            if anchor_mentiont_text not in dict_index_mentions:
                dict_index_mentions[anchor_mentiont_text] = dict()
            if target_wikidata_qid not in dict_index_mentions[anchor_mentiont_text]:
                dict_index_mentions[anchor_mentiont_text][target_wikidata_qid] = set()
            dict_index_mentions[anchor_mentiont_text][target_wikidata_qid] \
                .add((anchor_wikidata_qid, count_mentions_entities_anchorqid))

        end = time.time()

        logger.info('%s minutes took getting the inverse dictionary' % ((end - start) / 60))
        start = time.time()
        pickle.dump(dict_index_mentions, open(output_dict_inverse_index, 'wb'))
        end = time.time()
        logger.info('%s minutes to pickle the inverse dictionary' % ((end - start) / 60))

        start = time.time()
        logger.info('beginning of curr_df_link_stats.groupby([anchor_mention_text]), curr_df_link_stats.shape: %s' %
                    str(curr_df_link_stats.shape))
        curr_df_link_stats_gr_by_mention = \
            curr_df_link_stats.groupby(['anchor_mention_text'])['count_mentions_entities'].sum().reset_index(
                name='count_mentions')
        end = time.time()
        logger.info('%s minutes took groupby anchor_mention_text' % ((end - start) / 60))

        start = time.time()
        logger.info('beginning merge between curr_df_link_stats_gr_by_mention with shape, %s'
                    ' and curr_df_link_stats with shape %s' %
                    (str(curr_df_link_stats_gr_by_mention.shape), str(curr_df_link_stats.shape)))

        curr_df_link_stats = pd.merge(curr_df_link_stats_gr_by_mention,
                                      curr_df_link_stats,
                                      on=('anchor_mention_text',), how='inner')
        end = time.time()
        logger.info('%s minutes took merge on anchor_mention_text, with the resulting shape of %s' %
                    ((end - start) / 60, str(curr_df_link_stats.shape)))

        start = time.time()
        curr_df_link_stats['prior'] = curr_df_link_stats['count_mentions_entities'] / \
                                      curr_df_link_stats['count_mentions']

        end = time.time()
        logger.info('%s minutes took the prior division' % ((end - start) / 60))

        start = time.time()
        logger.info('curr_df_link_stats.shape before merging with df_title_changes_2022 is %s'
                    % str(curr_df_link_stats.shape))

        curr_df_link_stats = pd.merge(curr_df_link_stats,
                                      df_title_changes_2022[['title_2022', 'page_id']],
                                      on=('page_id',), how='inner')
        end = time.time()

        logger.info('%s minutes took merging with prior AND titles of 2022 happened in and resulting '
                    'shape is %s' % ((end - start) / 60, str(curr_df_link_stats.shape)))

        logger.info('begin renaming')
        start = time.time()
        curr_df_link_stats.rename(columns={'page_id': 'target_page_id',
                                           'title_2022': 'target_title_2022',
                                           'count_mentions_entities': 'nr_links'}, inplace=True)
        end = time.time()
        logger.info('%s minutes took renaming' % ((end - start) / 60))

        # leaving only the necessary rows as per table below:
        #     1. Mention (ex: "Texas")
        #     2. Entity_title (ex: Texas_(novel)) --> The title of the entity in the "cut".
        #     3. Entity_page_id (ex: 123456) --> The page id of the entity page (hope this doesn't change with time).
        #     4. Entity_QID (ex: Q123456) --> The id in wikidata KG.
        #     5. prior (ex: 0.001)
        #     6. prior_rank (ex: 15) --> the rank of the entity according to prior value across all other entities.
        #     7. nr_links --> nr of times this mention is linked to the entity. It is useful to
        #     get inlink stats (with respective average prior and prior_rank).
        #     8. Filtered_date --> the time cut date.
        curr_df_link_stats = curr_df_link_stats[[
            'anchor_mention_text',
            'target_wikidata_qid',
            'target_wikipedia_title_orig',
            'target_page_id',
            'filtered_date',
            'count_mentions',
            'nr_links',
            'prior',
            'target_title_2022']]
        # for each target entity, ranks the priors of mentions pointing to it, later can be used to estimate the
        # difficulty of a particular entity to be disambiguated
        logger.info('starting grouping by dense curr_df_link_stats with shape %s' % str(curr_df_link_stats.shape))
        start = time.time()
        curr_df_link_stats['prior_rank'] = curr_df_link_stats.groupby('target_wikidata_qid')['prior'] \
            .rank('dense', ascending=False)

        # we are here --> this prior_rank was wrongly calculated, should be wrt mention (prior_rank_men in true)!!!
        # priority_rank_men --> introduced 14/06/2022 to also account for overshadowed mentions!!!
        curr_df_link_stats['prior_rank_men'] = curr_df_link_stats.groupby('anchor_mention_text')['prior'] \
            .rank('dense', ascending=False)
        end = time.time()

        logger.info('%s minutes took grouping by dense' % ((end - start) / 60))

        logger.info('BEFORE MASSAGING curr_df_link_stats.shape: %s' % str(curr_df_link_stats.shape))
        start = time.time()
        curr_df_link_stats['target_wikipedia_title_orig_lower'] = \
            curr_df_link_stats['target_wikipedia_title_orig'].str.lower().str.strip().str.replace('_', ' ')
        end = time.time()
        logger.info('%s minutes took the replacing of _ and lowering of target_wikipedia_title_orig' %
                    ((end - start) / 60))

        start = time.time()
        curr_df_link_stats.dropna(subset=['anchor_mention_text'], inplace=True)

        end = time.time()
        logger.info('%s minutes took editdistance with dropna of mentions in nan' % ((end - start) / 60))

        start = time.time()

        curr_df_link_stats[[
            'edit_distance_mention_title',
            'edit_distance_mention_title_ent',
            'ratio_edit_distance_mention_title',
            'ratio_edit_distance_mention_title_ent',
            'overlap_type']] = \
            curr_df_link_stats \
                .loc[:, ['anchor_mention_text', 'target_wikipedia_title_orig', 'target_wikipedia_title_orig_lower']] \
                .apply(lambda x: get_overlap_mention_entity_title(*x), axis=1)

        end = time.time()
        logger.info('%s minutes to calculate the following features: "edit_distance_mention_title", '
                    '"edit_distance_mention_title_ent", '
                    '"ratio_edit_distance_mention_title", '
                    '"ratio_edit_distance_mention_title_ent", '
                    '"overlap_type"' % ((end - start) / 60))

        start = time.time()
        df_avg_prior = curr_df_link_stats[['target_wikidata_qid',
                                           'prior_rank', 'prior']].groupby('target_wikidata_qid').mean()
        end = time.time()
        logger.info('%s minutes to groupby target_wikidata_qid' % ((end - start) / 60))

        df_avg_prior.rename(columns={'prior_rank': 'avg_prior_rank',
                                     'prior': 'avg_prior'}, inplace=True)
        start = time.time()
        df_nr_inlinks = curr_df_link_stats[['target_wikidata_qid', 'nr_links']] \
            .groupby('target_wikidata_qid').sum()
        end = time.time()
        logger.info('%s minutes to groupby target_wikidata_qid' % ((end - start) / 60))
        df_nr_inlinks.rename(columns={'nr_links': 'nr_inlinks'}, inplace=True)
        start = time.time()
        curr_df_link_stats = pd.merge(curr_df_link_stats, df_avg_prior, left_on=('target_wikidata_qid',),
                                      right_on=('target_wikidata_qid',), how='inner')
        end = time.time()
        logger.info('%s minutes to merge on target_wikidata_qid on df_avg_prior' % ((end - start) / 60))
        start = time.time()
        curr_df_link_stats = pd.merge(curr_df_link_stats, df_nr_inlinks, left_on=('target_wikidata_qid',),
                                      right_on=('target_wikidata_qid',), how='inner')
        end = time.time()
        logger.info('%s minutes to merge on target_wikidata_qid on df_nr_inlinks' % ((end - start) / 60))

        nr_all_inlinks = df_nr_inlinks['nr_inlinks'].sum()
        curr_df_link_stats['entity_prior'] = curr_df_link_stats['nr_inlinks'] / nr_all_inlinks

        logger.info('AFTER MASSAGING curr_df_link_stats.shape: %s' % str(curr_df_link_stats.shape))
        start_length = time.time()
        logger.info('BEGIN adding the page lengths of target')

        before_merge_len = curr_df_link_stats.shape[0]
        curr_df_link_stats = pd.merge(left=curr_df_link_stats,
                                      right=df_page_info[
                                          ['wikidata_qid', 'wikipedia_creation_time', 'wikipedia_revision_time',
                                           'content_length']],
                                      left_on='target_wikidata_qid',
                                      right_on='wikidata_qid')

        if curr_df_link_stats.shape[0] != before_merge_len:
            logger.warning('!!!WARNING!!! curr_df_link_stats.shape[0] != before_merge_len '
                           'curr_df_link_stats.shape[0]: %s before_merge_len: %s' %
                           (str(curr_df_link_stats.shape[0]), str(before_merge_len)))

        curr_df_link_stats.rename(columns={'wikipedia_creation_time': 'target_wikipedia_creation_time',
                                           'wikipedia_revision_time': 'target_wikipedia_revision_time'},
                                  inplace=True)
        end_length = time.time()
        logger.info('%s minutes to END adding the page lengths of target' % ((end_length - start_length) / 60))

        # now saves the table
        start = time.time()
        logger.info('saving into final csv')
        curr_df_link_stats.to_csv(output_link_stats_file_path, sep='\t', index=False)
        end = time.time()
        logger.info('%s minutes took end saving into final csv ' % ((end - start) / 60))
        curr_cut_end = time.time()
        logger.info('================================ %s  minutes to end %s cut ===========' %
                    ((curr_cut_end - curr_cut_start) / 60, curr_time_cut))
