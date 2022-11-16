import logging
import os
# import candidate_ranking.utils as utils
# from bi_encoder.zeshel_utils import WORLDS
# from indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import pickle

import numpy as np

from models.biencoder.faiss_indexer import DenseHNSWFlatIndexer, DenseFlatIndexer
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def faiss_index_passages_single_date_cut(params, date_cut, tensors_shape):
    base_experiment_path = params['base_experiment_path']
    cand_faiss_path = params['cand_faiss_path']
    cand_faiss_path = os.path.join(base_experiment_path, cand_faiss_path)

    os.makedirs(cand_faiss_path, exist_ok=True)
    # output_dir_candidate_encodings = params['output_dir_candidate_encodings']
    # output_dir_candidate_encodings = os.path.join(base_experiment_path, output_dir_candidate_encodings)
    # curr_input_file_numpy_tensors = os.path.join(output_dir_candidate_encodings,
    #                                              '{}_tensors.pt'.format(date_cut))
    curr_input_file_numpy_tensors = os.path.join(cand_faiss_path,
                                                 '{}_tensors.pt'.format(date_cut))

    if params['faiss_hnsw']:
        cand_faiss_path = os.path.join(cand_faiss_path, 'hnsw')
    else:
        cand_faiss_path = os.path.join(cand_faiss_path, 'flat')

    os.makedirs(cand_faiss_path, exist_ok=True)
    curr_date_cut_path = os.path.join(cand_faiss_path, '{}_serialized_faiss.pt'.format(date_cut))

    # vector_dim = date_cut_to_tensor_shape[date_cut][1]
    vector_dim = tensors_shape[1]
    index_buffer = params['faiss_index_buffer']

    if params['faiss_hnsw']:
        logger.info('Using HNSW index in FAISS')
        index = DenseHNSWFlatIndexer(vector_dim, index_buffer)
    else:
        logger.info('Using Flat index in FAISS')
        index = DenseFlatIndexer(vector_dim, index_buffer)

    # BEGIN: commented code that was serializing, this occupies lots of space in disk: do not use serialization for now
    # if os.path.exists(curr_date_cut_path):
    #     index.deserialize_from(curr_date_cut_path)
    # else:
    #     if not params['debug']:
    #         curr_date_cut_candidate_encoding = pickle.load(open(curr_input_file_numpy_tensors, 'rb'))
    #     else:
    #         # just builds one randomly
    #         curr_date_cut_candidate_encoding = dict()
    #         curr_date_cut_candidate_encoding['tensors'] = np.random.rand(params['debug_size'], 768).astype('f')
    #     logger.info('Building index.')
    #     index.index_data(curr_date_cut_candidate_encoding['tensors'])
    #     logger.info('Done indexing data.')
    #
    #     index.serialize(curr_date_cut_path)
    # END: commented code that was serializing, this occupies lots of space in disk: do not use serialization for now

    # BEGIN: new code without serialization (see above the serialization code commented)
    if not params['debug']:
        curr_date_cut_candidate_encoding = pickle.load(open(curr_input_file_numpy_tensors, 'rb'))
    else:
        # just builds one randomly
        curr_date_cut_candidate_encoding = dict()
        curr_date_cut_candidate_encoding['tensors'] = np.random.rand(params['debug_size'], 768).astype('f')
    logger.info('Building index.')
    index.index_data(curr_date_cut_candidate_encoding['tensors'])
    logger.info('Done indexing data.')

    # END: new code without serialization (see above the serialization code commented)
    return index


def faiss_index_passages_old(params, candidate_per_date_cut_encodings, gpu_id):
    """
    Memory-inefficient version which requires numpy representations/embeddings be already pre-loaded inside
    the candidate_per_date_cut_encodings

    :param params:
    :param candidate_per_date_cut_encodings:
    :param gpu_id:
    :return:
    """
    output_path = params['cand_faiss_path']
    if params['faiss_hnsw']:
        output_path = os.path.join(output_path, 'hnsw')
    else:
        output_path = os.path.join(output_path, 'flat')
    os.makedirs(output_path, exist_ok=True)
    # logger.info('Loading candidate encoding from path: %s' % params['candidate_path'])
    # candidate_path = params['candidate_path']
    # candidate_per_world_encodings = torch.load(os.path.join(candidate_path))
    # gpu_id = 0
    faiss_date_cut_to_index = dict()
    for date_cut, curr_date_cut_candidate_encoding in candidate_per_date_cut_encodings.items():
        # print('------------- indexing the world', WORLDS[world_id], '------------------')
        # world_name = WORLDS[world_id]
        curr_date_cut_path = os.path.join(output_path, 'serialized_faiss_{}.pt'.format(date_cut))
        # ex curr_world_path --> 'data/zeshel/cache/faiss/indexed_candidates/pytorch_model_psgs_1_tkns_128_title_True/flat/serialized_faiss_american_football.pt'
        # if the file was already indexed, then ignores it and keeps going
        vector_dim = curr_date_cut_candidate_encoding.shape[1]
        index_buffer = params['faiss_index_buffer']

        if params['faiss_hnsw']:
            logger.info('Using HNSW index in FAISS')
            index = DenseHNSWFlatIndexer(vector_dim, index_buffer)
        else:
            logger.info('Using Flat index in FAISS')
            index = DenseFlatIndexer(vector_dim, index_buffer)

        if os.path.exists(curr_date_cut_path):
            index.deserialize_from(curr_date_cut_path)
        else:
            logger.info('Building index.')
            index.index_data(curr_date_cut_candidate_encoding)
            logger.info('Done indexing data.')
            index.serialize(curr_date_cut_path)
        # if params['faiss_on_gpu']:
        #     res = faiss.StandardGpuResources()
        #     index.index = faiss.index_cpu_to_gpu(res, 0, index.index)

        faiss_date_cut_to_index[date_cut] = index
    return faiss_date_cut_to_index
