#
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers.modeling_utils import WEIGHTS_NAME
from pytorch_transformers.optimization import WarmupLinearSchedule
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from models.biencoder.biencoder_v2_parallelizable import BiEncoderRankerV2
from models.utils.optimizer import get_bert_optimizer
from models.utils.utils import accuracy, write_to_file, save_model
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(reranker: BiEncoderRankerV2, eval_dataloader, params, device, end_epoch=False, nr_epoch=-1):
    reranker.eval()
    if params['silent']:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc='Evaluation')

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input = batch
        with torch.no_grad():
            # eval_loss, logits = biencoder_ranker(context_input, candidate_input)
            logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
            torch.arange(params['eval_batch_size'])
        ).numpy()
        tmp_eval_accuracy, _ = accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    if not end_epoch:
        logger.info('Eval accuracy: %.5f' % normalized_eval_accuracy)
    else:
        logger.info('Epoch %d Eval accuracy: %.5f' % (nr_epoch, normalized_eval_accuracy))

    results['normalized_accuracy'] = normalized_eval_accuracy
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params['type_optimization'],
        params['learning_rate'],
        fp16=params.get('fp16'),
    )


def get_scheduler(params, optimizer, len_train_data):
    batch_size = params['train_batch_size']
    grad_acc = params['gradient_accumulation_steps']
    epochs = params['num_train_epochs']

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params['warmup_proportion'])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(' Num optimization steps = %d' % num_train_steps)
    logger.info(' Num warmup steps = %d', num_warmup_steps)
    return scheduler


def main_train_biencoder(gpu_id, params, world_size, train_tensor_data, valid_tensor_data, base_experiment_path,
                         tokenizer):
    if params['data_parallel']:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://',
                                             rank=gpu_id, world_size=world_size)

    output_path = os.path.join(base_experiment_path, params['output_path'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_dir_logger = os.path.dirname(output_path)

    # Init model
    # device = 'cpu'
    # biencoder = BiEncoder(params)

    if not params['no_cuda']:
        device = torch.device('cuda:{}'.format(gpu_id))
        # reranker.to(device)
        # reranker.to(gpu_id)
    else:
        device = 'cpu'
    reranker = BiEncoderRankerV2(params).to(device)

    if params['data_parallel']:
        reranker = DistributedDataParallel(reranker,
                                           device_ids=[gpu_id],
                                           output_device=gpu_id,
                                           find_unused_parameters=True,
                                           )

    if params['gradient_accumulation_steps'] < 1:
        raise ValueError(
            'Invalid gradient_accumulation_steps parameter: {}, should be >= 1'.format(
                params['gradient_accumulation_steps']
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params['train_batch_size'] = (
            params['train_batch_size'] // params['gradient_accumulation_steps']
    )
    train_batch_size = params['train_batch_size']
    eval_batch_size = params['eval_batch_size']
    grad_acc_steps = params['gradient_accumulation_steps']

    # Fix the random seeds
    seed = params['seed']
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if reranker.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    # Load train data
    # train_samples = utils.read_dataset('train', data_path)

    if not params['data_parallel']:
        if params['shuffle']:
            train_sampler = RandomSampler(train_tensor_data)
        else:
            train_sampler = SequentialSampler(train_tensor_data)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_tensor_data,
            num_replicas=world_size,
            rank=gpu_id
        )

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    if not params['data_parallel']:
        valid_sampler = SequentialSampler(valid_tensor_data)
    else:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_tensor_data,
            num_replicas=world_size,
            rank=gpu_id
        )

    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # number_of_samples_per_dataset = {}

    time_start = time.time()

    write_to_file(
        os.path.join(output_dir_logger, 'training_params.txt'), str(params)
    )

    logger.info('Starting training')
    logger.info(
        'device: {} n_gpu: {}, distributed training: {}'.format(device, gpu_id, False)
    )

    optimizer = get_optimizer(reranker, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data))

    reranker.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params['num_train_epochs']
    for epoch_idx in trange(int(num_train_epochs), desc='Epoch'):
        tr_loss = 0

        if params['silent']:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc='Batch')

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input = batch
            scores = reranker(context_input, candidate_input)
            bs = scores.size(0)
            target = torch.arange(bs, device=device).long()
            loss = F.cross_entropy(scores, target, reduction='mean')

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params['print_interval'] * grad_acc_steps) == 0:
                logger.info(
                    'Step {} - epoch {} average loss: {}\n'.format(
                        step,
                        epoch_idx,
                        tr_loss / (params['print_interval'] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    reranker.parameters(), params['max_grad_norm']
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params['eval_interval'] * grad_acc_steps) == 0:
                evaluate(
                    reranker, valid_dataloader, params, device=device, end_epoch=False,
                )
                reranker.train()

                logger.info('\n')

        logger.info('***** Saving fine - tuned model *****')
        epoch_output_folder_path = os.path.join(output_dir_logger, 'epoch_{}'.format(epoch_idx))
        save_model(reranker, tokenizer, epoch_output_folder_path, params['output_model_name'])

        results = evaluate(
            reranker, valid_dataloader, params, device=device, end_epoch=True, nr_epoch=epoch_idx
        )
        reranker.train()

        ls = [best_score, results['normalized_accuracy']]
        li = [best_epoch_idx, epoch_idx]

        argmax_ls = np.argmax(ls)
        best_score = ls[argmax_ls]
        best_epoch_idx = li[argmax_ls]
        logger.info('\n')

    execution_time = (time.time() - time_start) / 60
    write_to_file(
        os.path.join(output_dir_logger, 'training_time.txt'),
        'The training took {} minutes\n'.format(execution_time),
    )
    logger.info('The training took {} minutes\n'.format(execution_time))

    # save the best model in the parent_dir
    logger.info('Best performance in epoch: {}'.format(best_epoch_idx))
    params['path_to_model'] = os.path.join(
        output_dir_logger,
        'epoch_{}'.format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    save_model(reranker, tokenizer, output_dir_logger, params['output_model_name'])
