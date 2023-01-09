# The main motivation of this module, which is derived from wikipedia_create_dataset.py, is to avoid the creation of
# MANY small files in the storage disk. This is a problem for /project_antwerp/ storage. So the idea is to create a
# SINGLE file and save into it by batches of multiple (like 1000 or so) lines, thus avoiding many I/Os from the disk,
# which can be a network disk.

import argparse
import csv
import json
import logging
import os
import re
import time
import traceback
import xml
from datetime import datetime
from multiprocessing import Process, Value, Lock
from threading import Thread
from typing import List

import py7zr
import requests
from pytorch_transformers import BertTokenizer

from tempel_creation.misc.article_queue import ArticleReadingQueue
from tempel_creation.misc.cleaning import gross_clean, fine_clean, clean_text_from_link_markers
from tempel_creation.misc.compiled_regexes import compiled_regexes
from tempel_creation.misc.load_wiki_sql_tables import load_wiki_page_title_to_wiki_page_id, \
    load_wiki_page_id_to_redirected_page_id, load_wiki_page_id_to_wikidata_qid
from tempel_creation.misc.seven_zip_reader import SevenZipStreamDecompressor
from tempel_creation.misc.wikipedia_history_reader import WikipediaHistoryReader
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def process_xml_parser(shutdown_xml_parser, v_nr_parsed_articles, filter_dates, v_nr_pages_change_title_error, config):
    """

    :return:

        wikipedia_page_id,
        wikipedia_title,
        wikidata_qid,
        wikipedia_creation_time,
        type (redirect, disambiguation, real page with content, any other?),
        content_length

    """

    max_look_back_for_stable_page_version = config['max_look_back_for_stable_page_version']
    min_days_stable_page_version = config['min_days_stable_page_version']

    date_time_obj_map = {fd_str: datetime.strptime(fd_str, '%Y-%m-%dT%H:%M:%SZ') for fd_str in filter_dates}

    dates_filter = {fd_str: (lambda in_date, f_in=fd_str: in_date <= date_time_obj_map[f_in]) for
                    fd_str in date_time_obj_map.keys()}

    filter_namespace = lambda ns: ns != '' and int(ns) == 0
    if len(config['only_process_these_page_ids']) > 0:
        logger.info('setting filter_pages to: %s' % config['only_process_these_page_ids'])
        only_process_these_page_ids = set(config['only_process_these_page_ids'])
        filter_pages = lambda page_id: page_id in only_process_these_page_ids
    else:
        filter_pages = None

    reader = WikipediaHistoryReader(filter_namespace, dates_filter, arq.enqueue_article,
                                    v_nr_pages_change_title_error,
                                    max_look_back_for_stable_page_version,
                                    min_days_stable_page_version, filter_pages=filter_pages)

    while not (shutdown_xml_parser.value == 1 and arq.process_files_queue.empty()):
        try:
            if config['input_format'] == '7zip':
                curr_file, curr_filepath = arq.process_files_queue.get(block=True, timeout=config['queues_timeout'])
                archive = py7zr.SevenZipFile(curr_filepath, mode='r')
                file_to_decompress = archive.files[0]

                decompressor = SevenZipStreamDecompressor(file_to_decompress.folder.coders,
                                                          file_to_decompress.compressed,
                                                          file_to_decompress.folder.unpacksizes, archive.fp,
                                                          file_to_decompress.folder.crc,
                                                          file_to_decompress.folder.password,
                                                          )
                reader.processed_size = 0
                reader.tot_size = file_to_decompress.uncompressed
                logger.info('parsing: %s' % curr_file)
                logger.info('compressed file size (GB): %s, %s' %
                            (curr_file, (os.path.getsize(curr_filepath) / 1024 / 1024 / 1024)))
                logger.info('uncompressed file size (GB): %s, %s ' %
                            (curr_file, (file_to_decompress.uncompressed / 1024 / 1024 / 1024)))
                xml.sax.parse(decompressor, reader)
                with v_nr_parsed_articles.get_lock():
                    v_nr_parsed_articles.value += 1
                logger.info('FINISHED TO PARSING THE FILE NR %s out of %s with time of %s hours' %
                            (v_nr_parsed_articles.value, tot_files_parse, (time.time() - initial_time) / 60 / 60))
            elif config['input_format'] == 'text':
                curr_file, curr_filepath = arq.process_files_queue.get(block=True, timeout=config['queues_timeout'])
                reader.processed_size = 0
                reader.tot_size = 0
                logger.info('parsing: %s' % curr_file)
                xml.sax.parse(curr_filepath, reader)
                with v_nr_parsed_articles.get_lock():
                    v_nr_parsed_articles.value += 1
                logger.info('FINISHED TO PARSING THE FILE NR %s out of %s with time of %s hours' %
                            (v_nr_parsed_articles.value, tot_files_parse, (time.time() - initial_time) / 60 / 60))
            else:
                raise RuntimeError('Unknown format: ' + config['input_format'])

        except Exception as e:
            logger.error('type error process_xml_parser: %s' % str(e))
            logger.error(traceback.format_exc())
            continue


def clean_entity_links_from_text(wikimedia_text, compiled_mention_finder, source_title, filtered_date,
                                 anchor_wikidata_qid, compiled_country_in_link):
    to_ret = ''
    last_span_added_pos = 0
    for curr_found_mention in compiled_mention_finder.finditer(wikimedia_text):
        span_start = curr_found_mention.start()
        span_end = curr_found_mention.end()
        mention_text = curr_found_mention.group()
        mention_data = get_mention(mention_text, source_title, filtered_date, anchor_wikidata_qid,
                                   compiled_country_in_link)
        mention_text = mention_data['mention']
        to_ret += wikimedia_text[last_span_added_pos:span_start]
        to_ret += ' ' + mention_text
        last_span_added_pos = span_end

    to_ret += wikimedia_text[last_span_added_pos:]
    return to_ret


def get_mention(mention_text, source_title, filtered_date, anchor_wikidata_qid, compiled_country_in_link):
    curr_found_mention_pr = mention_text[2:]
    if '|' in curr_found_mention_pr:
        index_pipe = curr_found_mention_pr.index('|')
        link = curr_found_mention_pr[:index_pipe]
        mention = curr_found_mention_pr[index_pipe + 1:]
        if mention.startswith('{{'):
            sm_ignore = True
            if sm_ignore:
                # logger.info('ignoring mention: %s for %s  %s  anchor qid: %s' %
                #             (mention_text, source_title, filtered_date, anchor_wikidata_qid))
                return {'mention': mention, 'link': link, 'has_to_ignore': True, 'has_to_count': True}
        if mention[:mention.index(']]')] == '':
            # pipe trick
            found_parenthesis = re.search(' \(.*\)$', link)
            # first look for parenthesis like "Yours, Mine and Ours (1968 film)"
            if found_parenthesis is not None:
                mention = link[:found_parenthesis.start()].strip() + mention
            else:
                # if there is no parenthesis like in previous case, then look for the first comma like
                #  "Il Buono, il Brutto, il Cattivo"
                found_comma = re.search(', ', link)
                if found_comma is not None:
                    mention = link[:found_comma.start()].strip() + mention

            # if the mention is still empty, something wrong:
            if mention[:mention.index(']]')] == '':
                logger.warning('something wrong, mention empty for: %s  ignoring %s %s anchor qid: %s' %
                               (mention_text, source_title, filtered_date, anchor_wikidata_qid))
                return {'mention': mention, 'link': link, 'has_to_ignore': True, 'has_to_count': True}

    else:
        right_index_link = mention_text.index(']]')
        link = mention_text[2:right_index_link]
        mention = mention_text[2:]

    link = link.strip()
    mention = mention.strip()

    if len(link) > 0:
        if compiled_country_in_link.search(link):
            return {'mention': '', 'link': '', 'has_to_ignore': True, 'has_to_count': False}

    if len(link) == 0:
        logger.warning('something wrong, EMPTY LINK!!!!: "%s" all this inside "%s" "%s" anchor qid: "%s"' %
                       (mention_text, source_title, filtered_date, anchor_wikidata_qid))
        return {'mention': mention, 'link': link, 'has_to_ignore': True, 'has_to_count': True}
    elif len(link) == 1:
        link = link[0].upper()
    else:
        link = link[0].upper() + link[1:]

    mention = mention.replace(']]', '')

    link = link.strip()
    mention = mention.strip()

    if link.startswith('Image:') or link.startswith('File:') or link.startswith('Category:') or \
            link.startswith('Wiktionary:') or \
            link.startswith(':Image:') or link.startswith(':File:') or link.startswith(':Category:') or \
            link.startswith(':Wiktionary:'):
        return {'mention': '', 'link': '', 'has_to_ignore': True, 'has_to_count': False}

    # if it points to anchor, also ignore
    if '#' in link:
        return {'mention': '', 'link': '', 'has_to_ignore': True, 'has_to_count': False}
    # begin DELETE THE QUOTATION
    mention = re.sub(r'(?i)\'+(.*?)\'+', lambda m: m.group(1), mention)
    # end DELETE THE QUOTATION

    return {'mention': mention, 'link': link, 'has_to_ignore': False, 'has_to_count': True}


def parse_mentions_from_source(source, source_title, filtered_date, compiled_mention_finder,
                               anchor_wikidata_qid, compiled_country_in_link, get_span_pos):
    tot_detected_mentions = 0
    tot_link_errors = 0
    to_ret_mention_links = list()
    for curr_found_mention in compiled_mention_finder.finditer(source):
        # strips the [[ and ]]
        found_mention_group = curr_found_mention.group()
        tot_detected_mentions += 1

        returned = get_mention(found_mention_group, source_title, filtered_date, anchor_wikidata_qid,
                               compiled_country_in_link)
        mention = returned['mention']
        link = returned['link']
        has_to_ignore = returned['has_to_ignore']
        has_to_count = returned['has_to_count']
        if not has_to_count:
            tot_detected_mentions -= 1
        if has_to_ignore and has_to_count:
            tot_link_errors += 1
        if has_to_ignore:
            continue

        nr_pipes = mention.count('|')
        if nr_pipes > 0:
            # logger.warning('"%s" "%s": ignoring mention because nr of pipes > 0: "%s" all this inside "%s", '
            #                '"%s" qid: "%s"' %
            #                (source_title, filtered_date, curr_found_mention, source_title,
            #                 filtered_date, anchor_wikidata_qid))
            tot_link_errors += 1
            continue

        if len(link) == 0:
            logger.warning('something wrong, EMPTY LINK!!!!: "%s" all this inside "%s" "%s" qid: "%s"' %
                           (curr_found_mention, source_title, filtered_date, anchor_wikidata_qid))
            tot_link_errors += 1
            continue
        link = link.replace(' ', '_')
        if get_span_pos:
            to_ret_mention_links.append({'anchor_mention_text': mention, 'target_wikipedia_title_orig': link,
                                         'span': curr_found_mention.span()})
        else:
            to_ret_mention_links.append({'anchor_mention_text': mention, 'target_wikipedia_title_orig': link})

    return to_ret_mention_links, tot_detected_mentions, tot_link_errors


def get_mentions_and_links(source, source_length, source_title, anchor_wikidata_qid, filtered_date,
                           compiled_mention_finder, compiled_country_in_link):
    mention_links, tot_detected_mentions, tot_links_errors = parse_mentions_from_source(source, source_title,
                                                                                        filtered_date,
                                                                                        compiled_mention_finder,
                                                                                        anchor_wikidata_qid,
                                                                                        compiled_country_in_link,
                                                                                        get_span_pos=False)
    anchor_content_length = source_length
    for curr_mention_link in mention_links:
        curr_mention_link['target_wikipedia_page_id'] = None
        curr_mention_link['target_wikipedia_title'] = None
        curr_mention_link['target_wikidata_qid'] = None
        curr_mention_link['anchor_content_length'] = anchor_content_length
        curr_mention_link['anchor_wikipedia_title'] = source_title
        curr_mention_link['anchor_wikidata_qid'] = anchor_wikidata_qid
        curr_mention_link['filtered_date'] = filtered_date

    return mention_links, tot_detected_mentions, tot_links_errors


def process_page_link_stats_writer(shutdown_file_writers, time_cut_list, output_dir_stats, config):
    csv_links_writers = dict()
    csv_links_files = dict()

    for curr_time_cut in time_cut_list:
        curr_page_info_file = open(os.path.join(output_dir_stats, 'page_link_stats_{}.csv'.format(curr_time_cut)), 'wt')
        curr_page_info_writer = csv.writer(curr_page_info_file, delimiter='\t')
        csv_links_writers[curr_time_cut] = curr_page_info_writer
        csv_links_files[curr_time_cut] = curr_page_info_file
        # adds title header into csv file
        csv_links_writers[curr_time_cut].writerow(
            ['anchor_wikidata_qid',
             'anchor_wikipedia_title',
             'anchor_wikipedia_page_id',
             'anchor_mention_text',
             'target_wikidata_qid',
             'target_wikipedia_title',
             'target_wikipedia_title_orig',
             'target_page_id',
             'filtered_date'
             ])

    while not (shutdown_file_writers.value == 1 and arq.link_stats_queue.empty()):
        try:
            to_save = arq.link_stats_queue.get(block=True, timeout=config['queues_timeout'])
        except Exception as e:
            continue
        filtered_date = to_save['filtered_date']
        csv_row = to_save['csv_row_link_stats']

        csv_links_writers[filtered_date].writerow(csv_row)

    for curr_date in time_cut_list:
        csv_links_files[curr_date].flush()
        csv_links_files[curr_date].close()

    logger.info('!!(in process_page_link_stats_writer) closed the file from process_page_link_stats_writer(), DONE')


def process_page_info_stats_writer(shutdown_file_writers, time_cut_list, output_dir_stats, config):
    csv_page_info_writers = dict()
    csv_page_info_files = dict()

    for curr_time_cut in time_cut_list:
        curr_page_info_file = open(os.path.join(output_dir_stats, 'page_info_{}.csv'.format(curr_time_cut)), 'wt')
        curr_page_info_writer = csv.writer(curr_page_info_file, delimiter='\t')
        csv_page_info_writers[curr_time_cut] = curr_page_info_writer
        csv_page_info_files[curr_time_cut] = curr_page_info_file
        # adds title header into csv file
        csv_page_info_writers[curr_time_cut].writerow(
            ['page_id', 'wikipedia_title', 'wikidata_qid', 'wikipedia_creation_time', 'wikipedia_revision_time',
             'content_length'])

    while not (shutdown_file_writers.value == 1 and arq.page_info_stats_queue.empty()):
        try:
            to_save = arq.page_info_stats_queue.get(block=True, timeout=config['queues_timeout'])
        except Exception as e:
            continue
        filtered_date = to_save['filtered_date']
        csv_row = to_save['csv_row_page_info']
        csv_page_info_writers[filtered_date].writerow(csv_row)

    for curr_date in time_cut_list:
        csv_page_info_files[curr_date].flush()
        csv_page_info_files[curr_date].close()

    logger.info('!!(in wikipedia_create_dataset) closed the file from process_page_info_writer(), DONE')


def process_page_title_change_writer(shutdown_file_writers, output_dir, config):
    csv_title_changes_file = open(os.path.join(output_dir, 'title_changes.csv'), 'wt')
    csv_title_changes_writer = csv.writer(csv_title_changes_file, delimiter='\t')
    csv_title_changes_writer.writerow(['title', 'filtered_date', 'page_id'])  # title_1, 2001, page_id

    while not (shutdown_file_writers.value == 1 and arq.title_changes_queue.empty()):
        try:
            to_save = arq.title_changes_queue.get(block=True, timeout=config['queues_timeout'])
        except Exception as e:
            continue
        csv_row = to_save['csv_row_title_changes']
        csv_title_changes_writer.writerow(csv_row)

    csv_title_changes_file.flush()
    csv_title_changes_file.close()

    logger.info('!!(in process_page_title_change_writer) closed the file from process_page_info_writer(), DONE')


def process_evolution_file_writer(shutdown_file_writers, str_buffer_size, str_buffer_nr_files, evolution_file_path,
                                  config):
    tot_len = 0
    tot_nr_files = 0
    with open(evolution_file_path, 'wt', encoding='utf8') as outfile:
        accum_json_str = ''
        while not (shutdown_file_writers.value == 1 and arq.evolution_content_queue.empty()):
            try:
                to_save = arq.evolution_content_queue.get(block=True, timeout=config['queues_timeout'])
            except Exception as e:
                continue

            evolution_file_dict = to_save['evolution_file_dict']

            json_str = json.dumps(evolution_file_dict, ensure_ascii=False)
            json_str += '\n'
            len_json_str = len(json_str)
            tot_len += (len_json_str / 1024 / 1024)
            accum_json_str += json_str
            tot_nr_files += 1
            if tot_len > str_buffer_size or tot_nr_files > str_buffer_nr_files:
                logger.info('appending to the evolutionfile with tot_len of %s '
                            'and tot_nr_files of %s' % (tot_len, tot_nr_files))
                outfile.write(accum_json_str)
                outfile.flush()
                accum_json_str = ''
                tot_len = 0
                tot_nr_files = 0
        if len(accum_json_str) > 0:
            outfile.write(accum_json_str)
            outfile.flush()

    logger.info('!!(in process_evolution_file_writer) closed the file from process_page_info_writer(), DONE')


def process_article(wikipedia_page_id_to_wikidata_qid, wikipedia_page_id_to_redirected_page_id, shutdown, v_lock,
                    v_nr_processed_articles, time_cut_list, initial_time, convert_through_api,
                    convert_to_text_dictionary, params):
    request_session1 = requests.Session()
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'],
        do_lower_case=params['lowercase'],
        cache_dir=params['bert_cache_dir']
    )

    while not (shutdown.value == 1 and arq.article_queue.empty()):
        try:
            wikipedia_title, source_per_date, page_id, wikipedia_creation_time, \
            wikipedia_revision_time_per_date, title_of_page_info_per_date = arq.article_queue.get(block=True,
                                                                                                  timeout=config[
                                                                                                      'queues_timeout'])
        except Exception as e:
            logger.error(traceback.format_exc())
            continue
        page_id = int(page_id)
        try:
            if page_id in wikipedia_page_id_to_redirected_page_id:
                continue
            try:
                with v_nr_processed_articles.get_lock():
                    v_nr_processed_articles.value += 1
                    if v_nr_processed_articles.value % 1000 == 0:
                        logger.info('nr of processed articles ' + str(v_nr_processed_articles.value) +
                                    ' in ' + str((time.time() - initial_time) / 60) + ' mins')
                        logger.info('Avg. articles per minute: %s' %
                                    (v_nr_processed_articles.value / ((time.time() - initial_time) / 60)))

            finally:
                pass
            wikidata_qid = wikipedia_page_id_to_wikidata_qid[page_id]

            wikipedia_title = wikipedia_title.replace(' ', '_').strip()
            wiki_temp_evolution_dict = dict()
            wiki_temp_evolution_dict['page_id'] = page_id
            wiki_temp_evolution_dict['wikidata_qid'] = wikidata_qid
            wiki_temp_evolution_dict['wikipedia_creation_date'] = wikipedia_creation_time
            wiki_temp_evolution_dict['wikipedia_title'] = wikipedia_title
            wiki_temp_evolution_dict['evolution'] = dict()

            for curr_filtered_date in time_cut_list:
                if source_per_date[curr_filtered_date] is None or wikipedia_title.strip() == '':
                    continue

                # first preliminary clean based on FEVER's code
                simple_cleaned_text = gross_clean(source_per_date[curr_filtered_date],
                                                  compiled_regexes, convert_through_api, convert_to_text_dictionary,
                                                  request_session1, v_lock)

                # extracts entity links and mentions from text

                simple_cleaned_stripped_code = fine_clean(simple_cleaned_text)
                content_length = len(simple_cleaned_stripped_code.split(' '))

                mention_links, tot_detected_mentions, tot_links_errors = \
                    get_mentions_and_links(simple_cleaned_stripped_code, content_length, wikipedia_title,
                                           wikidata_qid, curr_filtered_date,
                                           compiled_regexes['compiled_mention_finder'],
                                           compiled_regexes['compiled_country_in_link'])

                page_info_csv_row = [page_id, wikipedia_title, wikidata_qid,
                                     wikipedia_creation_time,
                                     wikipedia_revision_time_per_date[curr_filtered_date],
                                     content_length]
                arq.page_info_stats_queue.put(
                    {'filtered_date': curr_filtered_date, 'csv_row_page_info': page_info_csv_row})

                arq.title_changes_queue.put({'csv_row_title_changes':
                                                 [title_of_page_info_per_date[curr_filtered_date]['title'].replace(' ',
                                                                                                                   '_').strip(),
                                                  title_of_page_info_per_date[curr_filtered_date]['filtered_date'],
                                                  title_of_page_info_per_date[curr_filtered_date]['page_id']]})

                for curr_mention_link in mention_links:
                    link_stats_csv_row = [curr_mention_link['anchor_wikidata_qid'],
                                          curr_mention_link['anchor_wikipedia_title'],
                                          page_id,
                                          curr_mention_link['anchor_mention_text'],
                                          curr_mention_link['target_wikidata_qid'],
                                          curr_mention_link['target_wikipedia_title'],
                                          curr_mention_link['target_wikipedia_title_orig'],
                                          curr_mention_link['target_wikipedia_page_id'],
                                          curr_mention_link['filtered_date']]
                    arq.link_stats_queue.put({'filtered_date': curr_filtered_date,
                                              'csv_row_link_stats': link_stats_csv_row})

                curr_temp_version = dict()
                curr_temp_version['wikipedia_revision_date'] = wikipedia_revision_time_per_date[curr_filtered_date]
                curr_temp_version['wikipedia_title'] = title_of_page_info_per_date[curr_filtered_date]['title']
                curr_temp_version['wikipedia_source'] = simple_cleaned_stripped_code

                cleaned_text = clean_text_from_link_markers(simple_cleaned_stripped_code)
                bert_tokenized = tokenizer.tokenize(cleaned_text)
                title_bert_tokenized = tokenizer.tokenize(curr_temp_version['wikipedia_title'])

                curr_temp_version['cleaned_text'] = cleaned_text
                curr_temp_version['bert_wikipedia_title'] = title_bert_tokenized
                curr_temp_version['bert_tokenized'] = bert_tokenized

                wiki_temp_evolution_dict['evolution'][curr_filtered_date] = curr_temp_version

            arq.evolution_content_queue.put({'evolution_file_dict': wiki_temp_evolution_dict})
        except Exception as err:
            logger.error('ERROR TO OBTAIN QID ON ---- enqueued wikipedia_title: %s ---- page_id: %s' %
                         (wikipedia_title, page_id))
            logger.error(traceback.format_exc())
        finally:
            pass

    logger.info('exiting process_article, DONE')


#
def display(process_file_readers: List, processes_article: List, v_tot_detected_mentions, v_tot_processed_mentions,
            v_nr_processed_articles, v_nr_scanned_entities, initial_time, v_nr_hit_dictionary_convert):
    while True:
        try:
            logger.info('\tDISPLAY Size of queue article_queue: %s' % arq.article_queue.qsize())
            logger.info('\tDISPLAY Size of queue process_files_queue: %s' % arq.process_files_queue.qsize())
            logger.info('\tDISPLAY Size of queue page_info_stats_queue: %s' % arq.page_info_stats_queue.qsize())
            logger.info('\tDISPLAY Size of queue link_stats_queue: %s' % arq.link_stats_queue.qsize())
            logger.info('\tDISPLAY Size of queue title_changes_queue: %s' % arq.title_changes_queue.qsize())
            logger.info('\tDISPLAY Size of queue evolution_content_queue: %s' % arq.evolution_content_queue.qsize())
            logger.info('\tDISPLAY Size of v_nr_hit_dictionary_convert: %s' % v_nr_hit_dictionary_convert.value)
            is_alive = 0
            is_dead = 0
            for curr_process_article in processes_article:
                if curr_process_article.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1

            logger.info('\tDISPLAY NR Processing article processes alive: %s and dead %s' % (is_alive, is_dead))
            is_alive = 0
            is_dead = 0
            for curr_process_article in process_file_readers:
                if curr_process_article.is_alive():
                    is_alive += 1
                else:
                    is_dead += 1
            logger.info('\tDISPLAY NR file reading processes alive: %s and dead: %s' % (is_alive, is_dead))

            logger.info('\tDISPLAY TOT detected mentions (if in 0, uncomment the code that sums!!): %s' %
                        v_tot_detected_mentions.value)
            logger.info('\tDISPLAY TOT processed mentions (if in 0, uncomment the code that sums!!): %s' %
                        v_tot_processed_mentions.value)
            curr_time = time.time()
            logger.info('\tDISPLAY TOT processed entities: %s  Avg. articles per minute: %s' %
                        (v_nr_processed_articles.value,
                         v_nr_processed_articles.value / ((curr_time - initial_time) / 60)))
            if v_tot_detected_mentions.value > 0:
                logger.info('\tDISPLAY %% of processed mentions: %s' %
                            (100 * (v_tot_processed_mentions.value / v_tot_detected_mentions.value)))

            logger.info('\tDISPLAY v_nr_scanned_entities: %s Avg. v_nr_scanned_entities per minute: %s' %
                        (v_nr_scanned_entities.value, v_nr_scanned_entities.value / ((curr_time - initial_time) / 60)))

            time.sleep(60 * 1)
        except Exception as e:
            logger.error('type error display: %s' % str(e))
            logger.error(traceback.format_exc())
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/snapshot_extraction/snap_20220515/config/s01_config_content.json',
                        help='The config file that contains all the parameters')
    # ideally, nr_threads_processor + nr_threads_reader = nr of CPUs in the machine the script is being executed on.
    parser.add_argument('--nr_threads_processor', required=False, type=int,
                        default=40,
                        help='Nr of threads that will process the data')
    parser.add_argument('--nr_threads_reader', required=False, type=int,
                        default=20,
                        help='Nr of threads that read each of the history log files')

    args = parser.parse_args()
    logger.info('Running wikipedia_history_reader with the following arguments: %s' % args)

    logger.info('Current working directory: %s' % os.getcwd())

    config = json.load(open(args.config_file, 'rt'))
    config['nr_threads_processor'] = args.nr_threads_processor
    config['nr_threads_reader'] = args.nr_threads_reader
    logger.info('THE CONFIG IS: %s' % config)
    output_dir_data = config['output_dir_data']
    output_dir_stats = config['output_dir_stats']

    caches_dir = config['caches_dir']
    convert_through_api = config['convert_through_api']

    os.makedirs(output_dir_data, exist_ok=True)
    os.makedirs(output_dir_stats, exist_ok=True)
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir_data, 'commit_hash_content.txt'), 'wt') as outfile:
        outfile.write(sha)

    wiki_history_directory = config['wiki_history_directory']
    time_cut_list = config['time_cut_list']

    nr_threads_reader = config['nr_threads_reader']
    nr_threads_processor = config['nr_threads_processor']
    os.makedirs(caches_dir, exist_ok=True)
    ############### BEGIN: for a particular wikipedia snapshot --> data structures that we will need to get statistics
    wikipedia_page_id_to_wikidata_qid = dict()
    wikipedia_page_title_to_wikipedia_page_id = dict()
    wikipedia_page_id_to_wikipedia_page_title = dict()
    wikipedia_page_id_to_redirected_page_id = dict()

    ############### END: for a particular wikipedia snapshot --> data structures that we will need to get statistics

    ################ BEGIN: others independent of time
    # RQ: how far apart in time are wikidata and wikipedia entity creation?
    wikidata_qid_to_wikidata_creation_time = dict()
    wikidata_qid_to_wikipedia_creation_time = dict()

    ################ END: others independent of time

    arq = ArticleReadingQueue(article_queue_size=config['article_queue_size'],
                              link_stats_queue_size=config['link_stats_queue_size'],
                              page_info_stats_queue_size=config['page_info_stats_queue_size'],
                              title_changes_queue_size=config['title_changes_queue_size'],
                              evolution_content_queue_size=config['title_changes_queue_size'])

    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    path_wikipedia_page_info = config['path_wikipedia_page_info']
    path_wikipedia_page_redirects = config['path_wikipedia_page_redirects']

    used_qids = set()
    nr_pages_processed = 0
    idx_lines_to_reprocess = []
    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir, 'wikipedia_page_id_to_wikidata_qid.pickle')
    path_cache_wikipedia_page_title_to_wikipedia_page_id = os.path.join(caches_dir,
                                                                        'wikipedia_page_title_to_wikipedia_page_id.pickle')
    path_cache_wikipedia_page_id_to_wikipedia_page_title = os.path.join(caches_dir,
                                                                        'wikipedia_page_id_to_wikipedia_page_title.pickle')
    path_cache_wikipedia_page_id_to_redirected_page_id = os.path.join(caches_dir,
                                                                      'wikipedia_page_id_to_redirected_page_id.pickle')

    if not os.path.isfile(path_cache_wikipedia_page_id_to_redirected_page_id):
        wikipedia_page_title_to_wikipedia_page_id, _ = load_wiki_page_title_to_wiki_page_id(
            path_cache_wikipedia_page_title_to_wikipedia_page_id,
            path_cache_wikipedia_page_id_to_wikipedia_page_title,
            path_wikipedia_page_info)

    wikipedia_page_id_to_redirected_page_id = \
        load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id,
                                                path_wikipedia_page_redirects)

    wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(path_cache_wikipedia_page_id_to_wikidata_qid,
                                                                          path_wikipedia_wikidata_map)

    ##################

    process_file_readers = []
    v_nr_parsed_articles = Value('i', 0)
    shutdown_xml_parser = Value('i', 0)
    shutdown_file_writers = Value('i', 0)
    v_nr_pages_with_change_of_title = Value('i', 0)
    v_nr_pages_change_title_error = Value('i', 0)
    v_nr_scanned_entities = Value('i', 0)

    tot_files_parse = len(os.listdir(wiki_history_directory))

    v_lock = Lock()

    initial_time = time.time()

    for i in range(nr_threads_reader):
        t = Process(target=process_xml_parser,
                    args=(shutdown_xml_parser, v_nr_parsed_articles, time_cut_list, v_nr_pages_change_title_error,
                          config))
        t.start()
        process_file_readers.append(t)
    for curr_file in os.listdir(wiki_history_directory):
        curr_filepath = os.path.join(wiki_history_directory, curr_file)
        arq.process_files_queue.put((curr_file, curr_filepath))
    #######

    processes_article = []
    logger.info('nr_threads: %s' % nr_threads_processor)
    shutdown = Value('i', 0)
    v_nr_processed_articles = Value('i', 0)
    # total nr of detected mentions with links (i.e., the ones inside [[ ... ]] wikipedia text)
    v_tot_detected_mentions = Value('i', 0)
    # total nr of mentions that were actually processed processed. Ideally should the same as v_tot_detected_mentions
    v_tot_processed_mentions = Value('i', 0)
    convert_to_text_dictionary = arq.manager.dict()
    v_nr_hit_dictionary_convert = Value('i', 0)
    v_nr_api_calls_convert = Value('i', 0)
    #
    logger.info('wikipedia_create_dataset: multi processing activated')
    for i in range(nr_threads_processor):
        logger.info('initializing the process with len(wikipedia_page_id_to_wikidata_qid) %s' %
                    len(wikipedia_page_id_to_wikidata_qid))
        logger.info('initializing the process with len(wikipedia_page_id_to_redirected_page_id) %s' %
                    len(wikipedia_page_id_to_redirected_page_id))

        t = Process(target=process_article, args=(wikipedia_page_id_to_wikidata_qid,
                                                  wikipedia_page_id_to_redirected_page_id, shutdown, v_lock,
                                                  v_nr_processed_articles, time_cut_list, initial_time,
                                                  convert_through_api, convert_to_text_dictionary, config))

        t.start()
        processes_article.append(t)
    #
    processes_file_writers = []
    for _ in range(1):
        t = Process(target=process_page_info_stats_writer,
                    args=(shutdown_file_writers, time_cut_list, output_dir_stats, config))
        t.start()
        processes_file_writers.append(t)

    for _ in range(1):
        t = Process(target=process_page_link_stats_writer,
                    args=(shutdown_file_writers, time_cut_list, output_dir_stats, config))
        t.start()
        processes_file_writers.append(t)

    for _ in range(1):
        t = Process(target=process_page_title_change_writer, args=(shutdown_file_writers, output_dir_stats, config))
        t.start()
        processes_file_writers.append(t)

    for _ in range(1):
        evolution_file_path = os.path.join(output_dir_data, config['evolution_file_path'])
        str_buffer_size = config['str_buffer_size']
        str_buffer_nr_files = config['str_buffer_nr_files']
        t = Process(target=process_evolution_file_writer, args=(shutdown_file_writers, str_buffer_size,
                                                                str_buffer_nr_files, evolution_file_path, config))
        t.start()
        processes_file_writers.append(t)

    #
    logger.info('LAUNCHING THREAD DISPLAY!')
    thread = Thread(target=display, args=(process_file_readers, processes_article, v_tot_detected_mentions,
                                          v_tot_processed_mentions, v_nr_processed_articles,
                                          v_nr_scanned_entities, initial_time, v_nr_hit_dictionary_convert))
    thread.daemon = True
    thread.start()
    #
    logger.info('SOME NUMBERS: NR_THREADS_PROCESSOR: %s NR_THREADS_READER: %s' %
                (nr_threads_processor, nr_threads_reader))

    # makes sure all the processes are finished
    shutdown_xml_parser.value = 1
    for t in process_file_readers:
        t.join()

    logger.info('!!setting shutdown to true!!')
    shutdown.value = 1
    logger.info('!!DONE setting shutdown to true!!')

    for t in processes_article:
        t.join()

    # once we are sure that the processes parsing wiki are finished, send the signal for file writer process
    shutdown_file_writers.value = 1

    # and wait until they are done
    for t in processes_file_writers:
        t.join()

    logger.info('Done')
