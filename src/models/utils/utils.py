import io
import json
import logging
import os
import sys
from gc import get_referents
from types import ModuleType, FunctionType

import numpy as np
import torch
from pytorch_transformers.modeling_utils import CONFIG_NAME

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
from src.utils import tempel_logger

BLACKLIST = type, ModuleType, FunctionType

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def read_dataset(dataset_path, file_name, debug=False, debug_size=10):
    # test_2013-01-01T00:00:00Z_all.jsonl
    # file_name = '{}_{}_all.jsonl'.format(time_cut, dataset_name)
    # file_name = '{}_{}.jsonl'.format(time_cut, dataset_name)
    txt_file_path = os.path.join(dataset_path, file_name)

    samples = []

    with io.open(txt_file_path, mode='r', encoding='utf-8') as file:
        for line in file:
            samples.append(json.loads(line.strip()))
            if debug and len(samples) >= debug_size:
                break

    return samples


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels


def accuracy_and_predicted(out, labels):
    predicted = np.argmax(out, axis=1)
    return np.sum(predicted == labels), predicted == labels, predicted


def save_model(model, tokenizer, output_dir, model_name):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, model_name)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def write_to_file(path, string, mode='w'):
    with open(path, mode) as writer:
        writer.write(string)


def chunk_document_in_sentences(tokenized_document, max_doc_width):
    """

    :param tokenized_document:
    :param max_doc_width:
    :return: the idea is to chunk the document in sentences ; where each of the passages (chunks) has at most
    width of max_doc_width ; if a particular sentence is longer than max_doc_width, then it is splitted.
    """
    # sentences_end_pos = list()
    sentences = list()
    prev_idx_token = 0
    idx_token = 0
    for idx_token, curr_token in enumerate(tokenized_document):
        if curr_token == '.':
            if idx_token < len(tokenized_document) - 1:
                # this part was commented because lowercased can be provided as input
                # if tokenized_document[idx_token + 1][0].isupper():
                # sentences_end_pos.append(idx_token)
                sentences.append(tokenized_document[prev_idx_token:idx_token + 1])
                prev_idx_token = idx_token + 1
            else:
                # sentences_end_pos.append(idx_token)
                sentences.append(tokenized_document[prev_idx_token:idx_token + 1])
                prev_idx_token = idx_token + 1

    if prev_idx_token < idx_token + 1:
        sentences.append(tokenized_document[prev_idx_token:idx_token + 1])

    chunks = list()
    curr_chunk = list()
    for idx_sentence, curr_sentence in enumerate(sentences):
        if len(curr_chunk) > 0 and len(curr_chunk) + len(curr_sentence) > max_doc_width:
            chunks.append(curr_chunk)
            curr_chunk = list()

        if len(curr_chunk) + len(curr_sentence) <= max_doc_width:
            curr_chunk = curr_chunk + curr_sentence
        elif len(curr_chunk) == 0:  # and len(curr_sentence) > max_doc_width:
            curr_chunks = [curr_sentence[i:i + max_doc_width] for i in range(0, len(curr_sentence), max_doc_width)]
            for curr_chunk in curr_chunks:
                if len(curr_chunk) == max_doc_width:
                    chunks.append(curr_chunk)
            if len(curr_chunk) == max_doc_width:
                curr_chunk = list()
        else:
            logger.warning('!!WARNING: CHUNKS SUB-ROUTINE: strange else here!!')
    if len(curr_chunk) > 0:
        chunks.append(curr_chunk)

    # controls that the sum of the tokens of chunks is equal to the total number of tokens in the document
    assert sum(len(chk) for chk in chunks) == len(tokenized_document)

    # also asserts that the content is exactly the same
    flattened_chunks = [item for sublist in chunks for item in sublist]
    assert all(y == x for y, x in zip(flattened_chunks, tokenized_document))

    return chunks


def getsize(obj):
    """sum size of object & members. Function to get the size of a particular object. """
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
