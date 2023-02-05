# The initial goal of the module is to create a list of pages that contain redirect in order to
# ignore those in creation of dataset. IN FUTURE, the redirects have to be dealt with by taking the
# content of that point in time of the page a particular redirect points to.

import argparse
import csv
import io
import json
import logging
import os
import re
import time
import traceback
from multiprocessing import Process, Value
from threading import Thread
from typing import List

from src.tempel_creation.misc.article_queue import ArticleReadingQueue
from src.utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def write_to_output_file(arq: ArticleReadingQueue, v_shutdown_writer, config):
    base_experiment_path = config['base_experiment_path']
    output_path = os.path.join(base_experiment_path, config['output_path'])
    with open(output_path, 'wt') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['time_cut', 'wikidata_qid', 'wikipedia_title_redirect_to'])
        while not (arq.process_files_queue.empty() and v_shutdown_writer.value == 1):
            try:
                json_w_content_to_write = arq.process_files_queue.get(block=True, timeout=30)

                writer.writerow([json_w_content_to_write['curr_time_cut'],
                                 json_w_content_to_write['wikidata_qid'],
                                 json_w_content_to_write['redirect_page_to']])
                outfile.flush()
            except Exception as e:
                logger.error('error in write_to_output_file: ' + str(e))
                logger.error(traceback.format_exc())


def process_line_input_file_parallel(config, arq: ArticleReadingQueue, v_shutdown_file_processor):
    redirect_pattern = re.compile(r'(?i)^#REDIRECT \[\[(.*?)\]\]')

    redirect_pattern2 = re.compile(r'(?i){{Soft redirect\|')

    redirect_pattern3 = re.compile(r'(?i){{Wikibooks redirect\|')

    while not (arq.article_queue.empty() and v_shutdown_file_processor.value == 1):
        try:
            json_line_str = arq.article_queue.get(block=True, timeout=30)
            json_line = json.loads(json_line_str[0])
            curr_wikidata_qid = json_line['wikidata_qid']
            for curr_time_cut, curr_time_content in json_line['evolution'].items():
                curr_wikipedia_source = curr_time_content['wikipedia_source']
                found_redirect = False
                redirect_page_to = None
                if config['debug']:
                    curr_wikipedia_source = '#REDIRECT [[Xibornol]] and something else... '
                    # curr_wikipedia_source = """{{Soft redirect|Wikibooks:Transwiki:Geordie dialect words}}
                    #
                    # {{Short pages monitor}}<!-- This long comment was added to the page to prevent it from being listed on Special:Shortpages. It and the accompanying monitoring template were generated via Template:Long comment. Please do not remove the monitor template without removing the comment as well.-->
                    # """

                for curr_found_redirect in redirect_pattern.finditer(curr_wikipedia_source.strip()):
                    found_redirect = True
                    redirect_page_to = curr_found_redirect.group(1)
                    break

                for _ in redirect_pattern2.finditer(curr_wikipedia_source.strip()):
                    found_redirect = True
                    redirect_page_to = '####UNKNOWN####'
                    logger.info('found redirect pattern2: \n%s' % curr_wikipedia_source.strip())
                    break

                for _ in redirect_pattern3.finditer(curr_wikipedia_source.strip()):
                    found_redirect = True
                    redirect_page_to = '####UNKNOWN####'
                    logger.info('found redirect pattern3: \n%s' % curr_wikipedia_source.strip())
                    break

                if found_redirect:  # if redirect pattern found
                    arq.process_files_queue.put({'wikidata_qid': curr_wikidata_qid,
                                                 'redirect_page_to': redirect_page_to,
                                                 'curr_time_cut': curr_time_cut})
        except Exception as e:
            logger.error('error in process_line_input_file_parallel: ' + str(e))
            logger.error(traceback.format_exc())


def display(process_write_output_file: List, process_file: List, v_nr_entities, arq: ArticleReadingQueue, start_time):
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

            time.sleep(60 * 1)
        except Exception as e:
            logger.error('type error display: %s with stack %s' % (str(e), traceback.format_exc()))
            continue


def read_from_input_file(config, arq: ArticleReadingQueue, v_nr_entities):
    base_experiment_path = config['base_experiment_path']
    input_content_path = os.path.join(base_experiment_path, config['input_content_path'])

    with io.open(input_content_path, mode='rt', encoding='utf-8') as infile:
        for curr_line in infile:
            arq.article_queue.put((curr_line,))
            v_nr_entities.value += 1


if __name__ == '__main__':
    # pdb.set_trace()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/snapshot_extraction/snap_20220515/config/s02_redirect_detector.json',
                        help='The config file that contains all the parameters')

    args = parser.parse_args()
    logger.info('getting the final dataset with the following parameters: %s' % args)
    config = json.load(open(args.config_file, 'rt'))

    base_experiment_path = config['base_experiment_path']  # base_experiment_path
    input_content_path = os.path.join(base_experiment_path, config['input_content_path'])
    output_path = os.path.join(base_experiment_path, config['output_path'])

    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(base_experiment_path, 'commit_hash_redirect.txt'), 'wt') as outfile:
        outfile.write(sha)

    arq = ArticleReadingQueue()

    process_read_input_file = list()
    start_time = time.time()
    v_nr_entities = Value('i', 0)

    for i in range(1):
        t = Process(target=read_from_input_file, args=(config, arq, v_nr_entities))
        t.start()
        process_read_input_file.append(t)

    nr_threads = config['nr_threads']
    v_shutdown_file_processor = Value('i', 0)
    v_shutdown_writer = Value('i', 0)

    process_file = list()

    for i in range(nr_threads):
        t = Process(target=process_line_input_file_parallel,
                    args=(config, arq, v_shutdown_file_processor))
        t.start()
        process_file.append(t)

    process_write_output_file = list()

    process_output_file = list()
    for i in range(1):
        t = Process(target=write_to_output_file, args=(arq, v_shutdown_writer, config))
        t.start()
        process_output_file.append(t)

    if nr_threads >= 1:
        logger.info('LAUNCHING THREAD DISPLAY!')
        thread = Thread(target=display, args=(process_write_output_file, process_file, v_nr_entities, arq, start_time))
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    for curr_proc in process_read_input_file:
        curr_proc.join()

    v_shutdown_file_processor.value = 1

    for curr_proc in process_file:
        curr_proc.join()

    v_shutdown_writer.value = 1

    for curr_proc in process_output_file:
        curr_proc.join()
