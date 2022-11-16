import argparse
import json
import logging
import os
import pickle
import shutil
import time
from multiprocessing import Process, Value
from threading import Thread

import git

from models.utils.s05_wikipedia_encode_utils import get_candidate_encodings_parallel, gather_and_write_encodings, \
    gather_candidate_encodings, display
from tempel_creation.misc.article_queue import ArticleReadingQueue
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def main_entry(params, nr_gpus):
    nr_threads_per_gpu = params['nr_threads_per_gpu']
    nr_threads = nr_threads_per_gpu * nr_gpus
    gpu_ids = list(range(nr_gpus))

    already_processed_qids = set()
    logger.info('LOADING already_processed_qids')
    input_file_path = params['output_file']
    base_experiment_path = params['base_experiment_path']
    input_file_path = os.path.join(base_experiment_path, input_file_path)
    start = time.time()
    end = time.time()
    logger.info('%s minutes to read checkpoint: %s' % (((end - start) / 60), input_file_path))

    cand_faiss_path = os.path.join(base_experiment_path, params['cand_faiss_path'])

    os.makedirs(cand_faiss_path, exist_ok=True)
    logger.info('WE DO AVOID SNAPSHOTS!!!! removing everything under cand_faiss_path %s' % cand_faiss_path)
    shutil.rmtree(cand_faiss_path)
    os.makedirs(cand_faiss_path, exist_ok=True)

    wikidata_qid_to_label_id = dict()
    label_id_to_wikidata_qid = dict()
    for curr_date_cut in params['time_cut_list']:
        wikidata_qid_to_label_id[curr_date_cut] = dict()
        label_id_to_wikidata_qid[curr_date_cut] = dict()
        curr_output_label_mappings_path = os.path.join(cand_faiss_path,
                                                       '{}_label_id_to_wikidata_qid_mappings.pt'.format(curr_date_cut))
        # TODO: think if this checkpoint is worth it, ex: what if I start the job with another input???
        if os.path.exists(curr_output_label_mappings_path):
            loaded_d = pickle.load(open(curr_output_label_mappings_path, 'rb'))
            wikidata_qid_to_label_id[curr_date_cut] = loaded_d['wikidata_qid_to_label_id']
            label_id_to_wikidata_qid[curr_date_cut] = loaded_d['label_id_to_wikidata_qid']
    initial_time = time.time()
    # candidate_per_world_encodings, passage_id_to_label_id, label_id_to_passage_ids, label_doc_id_map = \
    process_file_readers = list()
    arq = ArticleReadingQueue()
    v_nr_entities = Value('i', 0)
    v_shutdown_encoders = Value('i', 0)
    v_shutdown_file_writers = Value('i', 0)
    v_shutdown_gather = Value('i', 0)
    logger.info('nr_threads is: %s' % nr_threads)
    logger.info('gpu_ids is: %s' % gpu_ids)
    for i in range(nr_threads):
        t = Process(target=get_candidate_encodings_parallel,
                    args=(params, gpu_ids[i % len(gpu_ids)], initial_time, v_nr_entities, arq, v_shutdown_encoders,
                          already_processed_qids, wikidata_qid_to_label_id))
        t.start()
        process_file_readers.append(t)

    process_gather_and_write = list()
    process_gather_candidate_tensors = list()
    if params['produce_encoded_jsonl']:
        for i in range(1):
            t = Process(target=gather_and_write_encodings,
                        args=(params, arq, v_shutdown_file_writers))
            t.start()
            process_gather_and_write.append(t)

    for i in range(1):
        t = Process(target=gather_candidate_encodings,
                    args=(v_shutdown_gather, arq, params, wikidata_qid_to_label_id, label_id_to_wikidata_qid))
        t.start()
        process_gather_candidate_tensors.append(t)

    if nr_threads >= 1:
        logger.info('LAUNCHING THREAD DISPLAY!')
        thread = Thread(target=display, args=(process_file_readers, process_gather_and_write,
                                              v_nr_entities, arq, initial_time))
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    input_file_path = os.path.join(base_experiment_path, params['input_file'])
    with open(input_file_path, 'rt', encoding='utf8') as infile:
        for curr_line in infile:
            arq.process_files_queue.put(curr_line)

    v_shutdown_encoders.value = 1

    for t in process_file_readers:
        t.join()

    v_shutdown_file_writers.value = 1
    for t in process_gather_and_write:
        t.join()

    v_shutdown_gather.value = 1
    for t in process_gather_candidate_tensors:
        t.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        # TODO
                        default='TODO',
                        help='The config file that contains all the parameters')
    parser.add_argument('--nr_gpus', required=False, type=int,
                        default=4,
                        help='the nr of gpus')

    args = parser.parse_args()
    logger.info('generating the candidate encodings: %s' % str(args))
    config = json.load(open(args.config_file, 'rt'))

    filtered_qids = set()
    base_experiment_path = config['base_experiment_path']
    output_dir = os.path.join(base_experiment_path, config['output_file'])
    output_dir = os.path.dirname(output_dir)
    os.makedirs(output_dir, exist_ok=True)  # 'experiments/eval_biencoder_passages/20211002_local_cpu/output'
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    with open(os.path.join(output_dir, 'commit_hash_last_run_wikipedia_encode.txt'), 'wt') as outfile:
        outfile.write(sha)

    main_entry(config, args.nr_gpus)
