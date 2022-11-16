import logging
import os
import time
from typing import Dict

import faiss
import psutil
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def get_topk_faiss_predictions_passages(faiss_date_cut_to_index,
                                        test_dataloader, params,
                                        label_id_to_candidate_token_ids,
                                        date_cut, stats: Dict,
                                        subset: str,
                                        save_predictions=False):
    top_k = params['top_k']
    silent = params['silent']
    logger.info('Getting top %d predictions.' % top_k)
    if silent:
        iter_ = test_dataloader
    else:
        iter_ = tqdm(test_dataloader)

    lst_context = []
    lst_candidate_toks = []
    lst_context_toks = []
    lst_pred_correct_passage_ids = []
    lst_pred_passage_ids = []
    lst_gold_passage_ids = []
    nn_labels_pred_scores_lst = []

    oid = 0
    process = psutil.Process(os.getpid())
    logger.info('step 7a rss memory (MB): %s' % (process.memory_info().rss / 1024 / 1024))
    indexes_sort = list()

    # TODO (kzaporoj 10/03/2022) - BEGIN new one
    if params['faiss_on_gpu']:
        # unloads the current index from gpu to cpu
        # faiss_date_cut_to_index[date_cut].index = faiss.index_gpu_to_cpu(faiss_date_cut_to_index[date_cut].index)
        # loads the new index from cpu to gpu
        load_time_begin = time.time()
        res = faiss.StandardGpuResources()
        faiss_date_cut_to_index[date_cut['entities_cut']].index = \
            faiss.index_cpu_to_gpu(res, 0, faiss_date_cut_to_index[date_cut['entities_cut']].index)
        load_time_end = time.time()
        logger.info('time to load faiss from cpu to gpu: %s minutes' % ((load_time_end - load_time_begin) / 60))

    # TODO (kzaporoj 10/03/2022) - END new one

    tn_candidates_toks_ids = label_id_to_candidate_token_ids[date_cut['entities_cut']]

    for step, batch in enumerate(iter_):
        context_input, gold_passage_ids_field, orig_index, context_tok_ids = batch
        # context_input.shape --> [2, 1024]
        # context_input -->
        # tensor([[ 0.1274,  0.2016, -0.5249,  ..., -0.3330,  0.4219,  0.3734],
        #         [ 0.2248, -0.0471, -0.7646,  ..., -0.2577,  0.5958,  0.3833]])
        # ---
        # gold_passage_ids_field.shape --> [2]
        # gold_passage_ids_field (ex from 2014) --> tensor([651110, 651110]) -->
        #    - example from 2013 --> [744633, 744633]   --> weird that both 2013 as 2014 present equal values.
        # ---
        # orig_index.shape --> [2]
        # orig_index --> tensor([98, 99])

        values, indicies = faiss_date_cut_to_index[date_cut['entities_cut']].search_knn(context_input.numpy(), top_k)
        values = torch.from_numpy(values)
        indicies = torch.from_numpy(indicies)
        # values.shape --> torch.Size([2, 64])
        # values --> tensor([[101.0603,  96.4545,  95.9813,  93.3255,  93.2835,  93.1024,  91.7401,
        #           ....
        #           84.4571],
        #         [100.4285,  99.8317,  99.3000,  98.0757,  97.0289,  96.5212,  95.9009,
        #           ....
        #           89.5940]])
        # ---
        # indicies.shape --> torch.Size([2, 64])
        # indicies -->
        # tensor([[ 651110,  956097, 3280061, 2520103,  171854,   86800,  334205, 2758315,
        #           ...
        #            56711,  524740, 3188151,  614150, 1321727, 3247476,  133713, 4077890],
        #         [ 956097,  171854, 4171701,  651110, 3280061, 1069643, 2821077, 2572832,
        #           ...
        #          1155212, 2561371, 3888932, 1758954, 1358141, 2774806, 3182491,    7785]])
        # ---
        for instance_in_batch in range(context_input.shape[0]):
            oid += 1
            indexes_sort.append(orig_index[instance_in_batch].item())

            inds = indicies[instance_in_batch]
            # indicies --> the faiss-ranked indicies of candidates
            nn_labels_pred_scores_lst.append(values[instance_in_batch].cpu())
            # gold_passage_ids_field
            # tensor([651110, 651110])
            # TODO --> 24/03/2022 --> BEGIN - see if version with [instance_in_batch] is correct,
            # previous version; think it is wrong because it assumes multiple correct passage_ids probably
            gold_passage_ids_s = {gold_passage_ids_field[instance_in_batch].item()}
            # TODO --> 24/03/2022 --> END - see if version with [instance_in_batch] is correct,

            # gold_passage_ids_s --> {651110}
            added_pointer = False
            pointers_to_correctly_predicted = list()
            for curr_pointer in range(top_k):
                pred_label_id = inds[curr_pointer].item()
                if pred_label_id in gold_passage_ids_s:
                    pointers_to_correctly_predicted.append(curr_pointer)
                    if not added_pointer:
                        stats[date_cut['mentions_cut']][subset].add(curr_pointer)
                        added_pointer = True
            if not added_pointer:
                stats[date_cut['mentions_cut']][subset].add(-1)
            if not save_predictions:
                continue

            lst_pred_passage_ids.append(inds.cpu())
            lst_context.append(context_input[instance_in_batch].cpu())

            curr_candidate_toks = tn_candidates_toks_ids[indicies[instance_in_batch], :]
            lst_candidate_toks.append(curr_candidate_toks)

            lst_context_toks.append(context_tok_ids[instance_in_batch, :])
            #
            # lst context in the end is of the size of dataset where each tensor is of dimension 1024
            if len(pointers_to_correctly_predicted) == 0:
                # (kzaporoj) not sure if this is necessary, check what happens if it is empty
                pointers_to_correctly_predicted.append(-1)
            lst_pred_correct_passage_ids.append(pointers_to_correctly_predicted)
            # lst_pred_correct_passage_ids
            # [[3], [1], [0], [0], [5], [4], [0], [0], [0], [0], [0], [6], [-1], [6], [2], [7], [5], [10], [7], [0],
            # [1], [5], [50], [4], [12], [0], [0], [4], [0], [-1], [1], [2], [-1], [-1], [-1], [-1], [37], [28], [-1],
            # [3], [5], [1], [63], [-1], [4], [42], [49], [38], [44], [-1], [13], [4], [3], [1], [1], [4], [0], [0],
            # [0], [0], [0], [0], [0], [0], [0], [9], [3], [25], [4], [23], [5], [0], [32], [2], [0], [0], [0], [0],
            # [0], [1], [0], [0], [0], [0], [-1], [0], [47], [12], [24], [26], [9], [0], [0], [0], [0], [3], [1], [-1],
            # [-1], [-1]
            #
            curr_gold_passage_id_field = gold_passage_ids_field[instance_in_batch].cpu()
            lst_gold_passage_ids.append(curr_gold_passage_id_field.unsqueeze(0))
            # lst_gold_passage_ids
            # [tensor([386428]), tensor([386428]), tensor([386428]), tensor([386428]), tensor([386428]),
            # tensor([2309211]), tensor([309326]), tensor([309326]), tensor([309326]), tensor([309326]),
            # ...
            # tensor([2963953]), tensor([744633]), tensor([744633]), tensor([744633])]

    if params['faiss_on_gpu']:
        # unloads the current index from gpu to cpu
        unload_time_begin = time.time()
        faiss_date_cut_to_index[date_cut['entities_cut']].index = \
            faiss.index_gpu_to_cpu(faiss_date_cut_to_index[date_cut['entities_cut']].index)
        unload_time_end = time.time()
        logger.info('time to unload faiss from gpu to cpu: %s seconds' % (unload_time_end - unload_time_begin))

    logger.info('in evaluate_faiss_passages')
    logger.info('step 7b rss memory (MB): %s' % (process.memory_info().rss / 1024 / 1024))

    logger.info('step 7c rss memory (MB): %s' % (process.memory_info().rss / 1024 / 1024))

    nn_context = torch.stack(lst_context).float()  # --> shape --> torch.Size([5, 128])
    nn_candidate_toks = torch.stack(lst_candidate_toks).int()
    # nn_candidate_toks.shape --> [10, 64, 128]
    # db 23/03/2022 nn_context.shape --> [100, 1024]
    nn_context_toks = torch.stack(lst_context_toks).int()
    nn_pred_correct_passage_ids = [torch.LongTensor(nl) for nl in lst_pred_correct_passage_ids]
    nn_pred_correct_passage_ids = pad_sequence(nn_pred_correct_passage_ids, batch_first=True, padding_value=-1).long()
    # --> shape --> torch.Size([5, 2])
    # db 23/03/2022 nn_pred_correct_passage_ids.shape --> [100, 1]
    # db 23/03/2022 nn_pred_correct_passage_ids -->  [25],
    #         [ 4],
    #         [23],
    #         [ 5],
    #         [ 0],
    #         [32],
    #         [ 2],
    #         [ 0],
    #         [ 0],
    #         [ 0],
    #         [ 0],
    #         [ 0],
    #         [ 1],
    #         [ 0],
    #         [ 0],
    #         [ 0],
    #         [ 0],
    #         [-1],
    #         [ 0],
    #         [47],
    #         [12],
    #         [24],
    #         [26],
    #         [ 9],
    #         [ 0],
    #         [ 0],
    #         [ 0],
    #         [ 0],
    #         [ 3],
    #         [ 1],
    #         [-1],
    #         [-1],
    #         [-1]])

    nn_gold_passage_ids = pad_sequence(lst_gold_passage_ids, batch_first=True, padding_value=-1).long()
    # no need for this pad since the elements in lst_gold_passage_ids are of shape [1] each.
    #  as a result, nn_gold_passage_ids.shape --> [100, 1]
    # --> shape --> torch.Size([5, 2])
    # db 23/03/2022 nn_gold_passage_ids.shape --> [100, 1]
    # db 23/03/2022 nn_gold_passage_ids -->
    #         [3228282],
    #         [3228282],
    #         [3228282],
    #         [3228282],
    #         [3228282],
    #           ...
    #         [3129626],
    #         [3129626],
    #         [ 744633],
    #         [ 744633]])
    # db 23/03/2022 nn_gold_passage_ids  MAX VALUE vs --> nn_pred_correct_passage_ids

    nn_pred_passage_ids = torch.stack(lst_pred_passage_ids).long()  # --> shape --> torch.Size([5, 2])

    # nn_pred_passage_ids.shape --> [100, 64]
    # nn_pred_passage_ids
    # tensor([[2413781, 1186900, 2938598,  ...,  427657, 3523622, 2667945],
    #         [1186900,  386428, 2938598,  ...,  372563, 3473419,  427657],
    #         [ 386428, 1186900, 2938598,  ..., 3590088, 1594566, 3512887],
    #         ...,
    #         [1950801, 3284709, 3269377,  ..., 3248220, 3222482, 3503673],
    #         [1757768, 3292302, 1502382,  ..., 1196483, 2451309, 3725781],
    #         [1950801, 2003186, 3284709,  ..., 3224139, 3269739, 3504034]])

    # pads with -1s in case nested candidates of different length
    # max_nested_len = max([len(llist) for llist in nn_labels_all_scores_lst])
    # nn_labels_all_scores_lst = [llist + [-1] * (max_nested_len - len(llist)) for llist in nn_labels_all_scores_lst]
    # kzaporoj - in case of problems of problems with this padding, check this:
    #  - https://discuss.pytorch.org/t/nested-list-of-variable-length-to-a-tensor/38699/4
    nn_pred_passage_scores = torch.stack(nn_labels_pred_scores_lst).float()  # --> shape --> torch.Size([5, 2])
    # nn_pred_passage_scores.shape --> [100, 64]

    nn_context_orig_order = torch.zeros_like(nn_context, dtype=nn_context.dtype, device=nn_context.device)
    nn_context_orig_order[indexes_sort, :] = nn_context

    nn_candidate_toks_orig_order = torch.zeros_like(nn_candidate_toks, dtype=nn_candidate_toks.dtype,
                                                    device=nn_candidate_toks.device)
    nn_candidate_toks_orig_order[indexes_sort, :] = nn_candidate_toks
    # nn_candidate_toks_orig_order.shape --> torch.Size([10, 64, 128])

    nn_context_toks_orig_order = torch.zeros_like(nn_context_toks, dtype=nn_context_toks.dtype,
                                                  device=nn_context_toks.device)
    nn_context_toks_orig_order[indexes_sort, :] = nn_context_toks
    # nn_context_toks_orig_order.shape --> torch.Size([10, 128])
    #
    # nn_context_orig_order.shape
    #   torch.Size([100, 1024])
    # nn_context_orig_order
    # tensor([[-0.0229, -0.4406,  0.0656,  ..., -0.6035,  0.4161,  0.5474],
    #         [ 0.0974, -0.4917,  0.0056,  ..., -0.3998,  0.4687,  0.4267],
    #         [-0.1207, -0.1656,  0.0243,  ..., -0.5417,  0.4821,  0.2415],
    #         ...,
    #         [-0.2167, -0.1382, -0.1857,  ...,  0.3580, -0.4347,  0.0993],
    #         [-0.1917, -0.1956, -0.2644,  ..., -0.1247, -0.3258,  0.2089],
    #         [ 0.0190, -0.0378, -0.2039,  ...,  0.0265, -0.6223, -0.0705]])

    nn_pred_passage_ids_orig_order = torch.zeros_like(nn_pred_passage_ids, dtype=nn_pred_passage_ids.dtype,
                                                      device=nn_pred_passage_ids.device)
    nn_pred_passage_ids_orig_order[indexes_sort, :] = nn_pred_passage_ids
    # nn_pred_passage_ids_orig_order.shape
    #   torch.Size([100, 64])

    # tensor([[2413781, 1186900, 2938598,  ...,  427657, 3523622, 2667945],
    #         [1186900,  386428, 2938598,  ...,  372563, 3473419,  427657],
    #         [ 386428, 1186900, 2938598,  ..., 3590088, 1594566, 3512887],
    #         ...,
    #         [1950801, 3284709, 3269377,  ..., 3248220, 3222482, 3503673],
    #         [1757768, 3292302, 1502382,  ..., 1196483, 2451309, 3725781],
    #         [1950801, 2003186, 3284709,  ..., 3224139, 3269739, 3504034]])

    nn_pred_correct_passage_ids_orig_order = torch.zeros_like(nn_pred_correct_passage_ids,
                                                              dtype=nn_pred_correct_passage_ids.dtype,
                                                              device=nn_pred_correct_passage_ids.device)
    # indexes_sort
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    nn_pred_correct_passage_ids_orig_order[indexes_sort, :] = nn_pred_correct_passage_ids
    # nn_pred_correct_passage_ids_orig_order.shape
    # torch.Size([100, 1])
    #
    #
    # nn_pred_correct_passage_ids_orig_order
    # tensor([[ 3],
    #         [ 1],
    #         [ 4],
    #         [ 0],
    #          ...
    #         [-1],
    #         [ 1],
    #         [ 2],
    #         [-1]])
    nn_gold_passage_ids_orig_order = torch.zeros_like(nn_gold_passage_ids, dtype=nn_gold_passage_ids.dtype,
                                                      device=nn_gold_passage_ids.device)
    nn_gold_passage_ids_orig_order[indexes_sort, :] = nn_gold_passage_ids
    # nn_gold_passage_ids_orig_order.shape
    #   torch.Size([100, 1])
    # nn_gold_passage_ids_orig_order
    # tensor([[ 386428],
    #         [ 386428],
    #         [ 386428],
    #         [ 386428],
    #         [ 744633],
    #         [ 744633]])
    #
    # I think that
    #  nn_gold_passage_ids_orig_order and nn_gold_passage_ids ARE THE SAME!
    nn_pred_passage_scores_orig_order = torch.zeros_like(nn_pred_passage_scores, dtype=nn_pred_passage_scores.dtype,
                                                         device=nn_pred_passage_scores.device)
    nn_pred_passage_scores_orig_order[indexes_sort, :] = nn_pred_passage_scores
    # nn_pred_passage_scores_orig_order.shape
    # torch.Size([100, 64])
    # tensor([[122.0473, 120.3862, 120.0884,  ..., 108.4440, 108.4073, 108.3958],
    #         [124.7854, 123.5683, 123.1532,  ..., 109.7947, 109.6969, 109.6883],
    #         [129.6593, 125.3910, 123.8640,  ..., 111.8109, 111.7014, 111.6869],
    #         ...,
    #         [112.1647, 110.4191, 109.4874,  ..., 103.4169, 103.3659, 103.3498],
    #         [ 89.3968,  89.3915,  87.8958,  ...,  83.0173,  82.9635,  82.9003],
    #         [107.0604, 103.9812, 103.3505,  ...,  97.8657,  97.8420,  97.8322]])
    #
    nn_data = {
        'context_vecs': nn_context_toks_orig_order,
        'candidate_vecs': nn_candidate_toks_orig_order,
        'token_ids_context': None,
        'labels': nn_pred_correct_passage_ids_orig_order.squeeze(-1),
        'pred_passage_ids': nn_pred_passage_ids_orig_order,  # .shape for debug_size 100: torch.Size([100, 64])
        'gold_passage_ids': nn_gold_passage_ids_orig_order,  # .shape for debug_size 100: torch.Size([100, 1])
        'pred_passage_scores': nn_pred_passage_scores_orig_order  # .shape for debug_size 100: torch.Size([100, 64])
    }

    logger.info('step 7d rss memory (MB): %s' % (process.memory_info().rss / 1024 / 1024))
    return nn_data

# def get_topk_predictions_passages(
#         reranker: BiEncoderRanker,
#         test_dataloader,
#         cand_encode_list,
#         params,
#         logger,
#         # top_k=10,
#         # is_zeshel=False,
#         save_predictions=False):
#     """
#
#     :param reranker: model instance of blink.biencoder.biencoder.BiEncoderRanker
#     :param test_dataloader: torch.utils.data.dataloader.DataLoader
#     :param cand_encode_list: dictionary with following keys: dict_keys([0, 1, 2, 3, 4, 5, 6, 7])
#         During debugging, each entry contains a matrix with shape: torch.Size([201, 1024])
#     :param silent: False
#     :param logger:
#     :param top_k: 64
#     :param is_zeshel: True
#     :param save_predictions: True
#     :return:
#     """
#     # import pdb
#     #
#     # pdb.set_trace()
#
#     # reranker.eval()
#     device = reranker.device
#     top_k = params['top_k']
#     silent = params['silent']
#     logger.info('Getting top %d predictions.' % top_k)
#     if silent:
#         iter_ = test_dataloader
#     else:
#         iter_ = tqdm(test_dataloader)
#
#     lst_context = []
#     lst_pred_correct_passage_ids = []
#     lst_pred_passage_ids = []
#     lst_gold_passage_ids = []
#     nn_labels_pred_scores_lst = []
#     nn_worlds = []
#     stats = {}
#
#     world_size = len(WORLDS)
#
#     logger.info('World size : %d' % world_size)
#
#     for instance_in_batch in range(world_size):
#         stats[instance_in_batch] = Stats(top_k)
#
#     oid = 0
#     process = psutil.Process(os.getpid())
#     print('step 7a rss memory (MB): ', process.memory_info().rss / 1024 / 1024)
#     src = -1
#     cand_encs = None
#     indexes_sort = list()
#     for step, batch in enumerate(iter_):
#         batch = tuple(t.to(device) for t in batch)
#         # context_vecs, cand_vecs, src_vecs, passage_ids, tn_label_lengths
#         context_input, world_ids, gold_passage_ids, gold_passage_lengths, orig_index = batch
#         # context_input.shape --> [1,128]; contains the bert tok ids of the context surrounding the mention together with the mention
#         # world_ids.shape --> [1]; contains the world ids from which the context was extracted
#         # passage_ids.shape --> [1, 3]; contains the gold passage ids of the world in the world_ids[0]
#         # passage_lengths.shape --> [1]; contains the respective gold passage lengths
#         if src != world_ids[0]:
#             # make sure .to is invoked as little as possible, only if the current cand_encs does not have
#             # the candidates for the current world in GPU already: the .to operation of loading tensors from
#             # cpu to gpu is expensive
#             print('new world detected: ', world_ids[0], ' loading to device')
#             src = world_ids[0].item()
#             cand_encs = cand_encode_list[src].to(device)
#         # context_input.shape --> torch.Size([1, 128])
#         _, scores = reranker(
#             context_input=None,  # context_input.shape --> [1, 128]
#             cand_input=None,
#             biencoder=None,
#             label_input=None,
#             cand_encs=cand_encs,
#             context_encs=context_input
#             # cand_encode_list[src].shape --> torch.Size([70, 768]) with the debug_size of 70
#         )
#         scores = scores.detach()
#         # scores.shape --> torch.Size([2, 5])
#         # scores example --> tensor([[175.8596, 158.3336, 150.9940, 187.7227, 168.5059],
#         #         [135.8151, 140.2693, 154.0614, 157.0635, 136.5018]])
#
#         # this min is used during debugging where the size of loaded tensors can be set very low (ex: 10).
#         top_k = min(scores.shape[-1], top_k)
#
#         values, indicies = scores.topk(top_k)
#         # top_k --> 2
#         # values.shape --> torch.Size([2, 2])
#         # indicies.shape --> torch.Size([2, 2])
#         # values example --> tensor([[187.7227, 175.8596],
#         #         [157.0635, 154.0614]])
#         # indicies example --> tensor([[3, 0],
#         #         [3, 2]])
#         old_src = src
#         for instance_in_batch in range(context_input.size(0)):
#             oid += 1
#             inds = indicies[instance_in_batch]
#             indexes_sort.append(orig_index[instance_in_batch].item())
#             if world_ids[instance_in_batch] != old_src:
#                 # not the same domain, need to re-do
#                 if src != world_ids[instance_in_batch]:
#                     # make sure .to is invoked as little as possible, only if the current cand_encs does not have
#                     # the candidates for the current world in GPU already: the .to operation of loading tensors from
#                     # cpu to gpu is expensive
#                     print('new world detected: ', world_ids[instance_in_batch], ' loading to device')
#                     src = world_ids[instance_in_batch].item()
#                     cand_encs = cand_encode_list[src].to(device)
#
#                 _, new_scores = reranker(
#                     context_input=None,  # context_input[[instance_in_batch]],  # context_input.shape --> [1, 128]
#                     cand_input=None,
#                     label_input=None,
#                     cand_encs=cand_encs,
#                     context_encs=context_input[[instance_in_batch]]
#                     # cand_encode_list[src].shape --> torch.Size([70, 768]) with the debug_size of 70
#                 )
#                 new_scores = new_scores.detach()
#                 values, inds = new_scores.topk(top_k)
#                 inds = inds[0]
#                 nn_labels_pred_scores_lst.append(values.squeeze(0).cpu())
#             else:
#                 nn_labels_pred_scores_lst.append(values[instance_in_batch].cpu())
#
#             gold_passage_ids_s = gold_passage_ids[instance_in_batch][0:gold_passage_lengths[instance_in_batch]].tolist()
#             gold_passage_ids_s = set(gold_passage_ids_s)
#             added_pointer = False
#             pointers_to_correctly_predicted = list()
#             for curr_pointer in range(top_k):
#                 pred_label_id = inds[curr_pointer].item()
#                 if pred_label_id in gold_passage_ids_s:
#                     pointers_to_correctly_predicted.append(curr_pointer)
#                     if not added_pointer:
#                         stats[src].add(curr_pointer)
#                         added_pointer = True
#             if not added_pointer:
#                 stats[src].add(-1)
#             if not save_predictions:
#                 continue
#
#             lst_pred_passage_ids.append(inds.cpu())
#             lst_context.append(context_input[instance_in_batch].cpu())
#             if len(pointers_to_correctly_predicted) == 0:
#                 # (kzaporoj) TODO: not sure if this is necessary, check what happens if it is empty
#                 pointers_to_correctly_predicted.append(-1)
#             lst_pred_correct_passage_ids.append(pointers_to_correctly_predicted)
#             lst_gold_passage_ids.append(
#                 gold_passage_ids[instance_in_batch][0:gold_passage_lengths[instance_in_batch]].cpu())
#             nn_worlds.append(src)
#
#     print('step 7b rss memory (MB): ', process.memory_info().rss / 1024 / 1024)
#
#     res = Stats(top_k)
#     for src in range(world_size):
#         if stats[src].cnt == 0:
#             continue
#         logger.info('In world ' + WORLDS[src])
#         output = stats[src].output()
#         logger.info(output)
#         res.extend(stats[src])
#
#     print('step 7c rss memory (MB): ', process.memory_info().rss / 1024 / 1024)
#
#     logger.info(res.output())
#
#     nn_context = torch.stack(lst_context).float()  # --> shape --> torch.Size([5, 128])
#
#     nn_pred_correct_passage_ids = [torch.LongTensor(nl) for nl in lst_pred_correct_passage_ids]
#     nn_pred_correct_passage_ids = pad_sequence(nn_pred_correct_passage_ids, batch_first=True, padding_value=-1).long()
#     # --> shape --> torch.Size([5, 2])
#     # nn_pred_correct_passage_ids = torch.LongTensor(nn_pred_correct_passage_ids)
#
#     # nn_gold_passage_ids = [torch.LongTensor(nlg) for nlg in lst_gold_passage_ids]
#     nn_gold_passage_ids = pad_sequence(lst_gold_passage_ids, batch_first=True, padding_value=-1).long()
#     # --> shape --> torch.Size([5, 2])
#     # nn_gold_passage_ids = torch.LongTensor(nn_gold_passage_ids)
#
#     # nn_pred_passage_ids = torch.LongTensor(lst_pred_passage_ids)
#     nn_pred_passage_ids = torch.stack(lst_pred_passage_ids).long()  # --> shape --> torch.Size([5, 2])
#     # pads with -1s in case nested candidates of different length
#     # max_nested_len = max([len(llist) for llist in nn_labels_all_scores_lst])
#     # nn_labels_all_scores_lst = [llist + [-1] * (max_nested_len - len(llist)) for llist in nn_labels_all_scores_lst]
#     # kzaporoj - in case of problems of problems with this padding, check this:
#     #  - https://discuss.pytorch.org/t/nested-list-of-variable-length-to-a-tensor/38699/4
#     # nn_all_passage_scores = torch.FloatTensor(nn_labels_all_scores_lst)
#     # nn_pred_passage_scores = torch.FloatTensor(torch.stack(nn_labels_pred_scores_lst))
#     nn_pred_passage_scores = torch.stack(nn_labels_pred_scores_lst).float()  # --> shape --> torch.Size([5, 2])
#     nn_context_orig_order = torch.zeros_like(nn_context, dtype=nn_context.dtype, device=nn_context.device)
#     nn_context_orig_order[indexes_sort, :] = nn_context
#
#     nn_pred_passage_ids_orig_order = torch.zeros_like(nn_pred_passage_ids, dtype=nn_pred_passage_ids.dtype,
#                                                       device=nn_pred_passage_ids.device)
#     nn_pred_passage_ids_orig_order[indexes_sort, :] = nn_pred_passage_ids
#
#     nn_pred_correct_passage_ids_orig_order = torch.zeros_like(nn_pred_correct_passage_ids,
#                                                               dtype=nn_pred_correct_passage_ids.dtype,
#                                                               device=nn_pred_correct_passage_ids.device)
#     nn_pred_correct_passage_ids_orig_order[indexes_sort, :] = nn_pred_correct_passage_ids
#
#     nn_gold_passage_ids_orig_order = torch.zeros_like(nn_gold_passage_ids, dtype=nn_gold_passage_ids.dtype,
#                                                       device=nn_gold_passage_ids.device)
#     nn_gold_passage_ids_orig_order[indexes_sort, :] = nn_gold_passage_ids
#
#     nn_pred_passage_scores_orig_order = torch.zeros_like(nn_pred_passage_scores, dtype=nn_pred_passage_scores.dtype,
#                                                          device=nn_pred_passage_scores.device)
#     nn_pred_passage_scores_orig_order[indexes_sort, :] = nn_pred_passage_scores
#
#     nn_data = {
#         'context_vecs': nn_context_orig_order,  # .shape for debug_size 100: torch.Size([100, 128])
#         'pred_passage_ids': nn_pred_passage_ids_orig_order,  # .shape for debug_size 100: torch.Size([100, 64])
#         'pred_correct_passage_ids': nn_pred_correct_passage_ids_orig_order,
#         # .shape for debug_size 100: torch.Size([100, 5])
#         'gold_passage_ids': nn_gold_passage_ids_orig_order,  # .shape for debug_size 100: torch.Size([100, 5])
#         'pred_passage_scores': nn_pred_passage_scores_orig_order  # .shape for debug_size 100: torch.Size([100, 64])
#     }
#     # nn_data = {
#     #     'context_vecs': nn_context,  # .shape for debug_size 100: torch.Size([100, 128])
#     #     # 'pred_passage_vecs': nn_pred_passages,  # .shape for debug_size 100: torch.Size([100, 64, 128])
#     #     # kzaporoj, introduce 'labels_pred'
#     #     'pred_passage_ids': nn_pred_passage_ids,  # .shape for debug_size 100: torch.Size([100, 64])
#     #     'pred_correct_passage_ids': nn_pred_correct_passage_ids,  # .shape for debug_size 100: torch.Size([100, 5])
#     #     'gold_passage_ids': nn_gold_passage_ids,  # .shape for debug_size 100: torch.Size([100, 5])
#     #     # 'all_passage_scores': nn_all_passage_scores,  # .shape for debug_size 100: torch.Size([100, 100])
#     #     'pred_passage_scores': nn_pred_passage_scores  # .shape for debug_size 100: torch.Size([100, 64])
#     # }
#
#     nn_data['worlds'] = torch.LongTensor(nn_worlds)
#     print('step 7d rss memory (MB): ', process.memory_info().rss / 1024 / 1024)
#     return nn_data
