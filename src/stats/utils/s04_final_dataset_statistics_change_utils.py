import os
import pickle
from typing import Set, Dict, List

import torch
from tqdm import tqdm


def get_mappings(config, sorted_cuts, base_input_path, input_encodings_dir, shared_target_wikidata_qids: Set,
                 device):
    cut_to_label_id_wikidata_qid_map = dict()
    cut_to_shared_label_ids = dict()
    cut_to_filtered_tensors = dict()
    # for curr_cut in config['time_cuts']:
    # shared_label_ids =
    cut_to_shared_label_ids_to_filtered_ids = dict()

    for curr_cut in sorted_cuts:
        print('starting 1 curr_cut: ', curr_cut)
        curr_cut_config = config['time_cuts'][curr_cut]
        label_id_to_wikidata_qid_path = os.path.join(base_input_path, input_encodings_dir,
                                                     curr_cut_config['label_id_to_wikidata_qid_path'])
        print('starting 2 pickle.load: ', curr_cut, '  ', label_id_to_wikidata_qid_path)
        loaded_label_id_wikidata_qid_map = pickle.load(open(label_id_to_wikidata_qid_path, 'rb'))
        #
        #
        if config['debug']:
            # adjust labels so the maximum label is of debug_size
            loaded_label_id_wikidata_qid_map['wikidata_qid_to_label_id'] = \
                {wqid: min(lbl_id, config['debug_size'] - 1) for wqid, lbl_id in
                 loaded_label_id_wikidata_qid_map['wikidata_qid_to_label_id'].items()}

        # print('label_id_to_wikidata_qid_path: ', label_id_to_wikidata_qid_path)
        cut_to_label_id_wikidata_qid_map[curr_cut] = loaded_label_id_wikidata_qid_map
        # print('curr_cut is of', curr_cut)

        cut_to_shared_label_ids[curr_cut] = list()
        for cw in shared_target_wikidata_qids:
            if cw in loaded_label_id_wikidata_qid_map['wikidata_qid_to_label_id']:
                cut_to_shared_label_ids[curr_cut].append(
                    loaded_label_id_wikidata_qid_map['wikidata_qid_to_label_id'][cw])
            else:
                print('!!!WARNING!!! wikidata_qid not in loaded_label_id_wikidata_qid_map : ', cw)
        # cut_to_shared_label_ids[curr_cut] = [loaded_label_id_wikidata_qid_map['wikidata_qid_to_label_id'][cw]
        #                                      for cw in shared_target_wikidata_qids]

        # from a particular label_id to an id after the tensor (see loaded_tensor below) has been filtered by
        # label_ids only present in cut_to_shared_label_ids
        print('starting creating label_ids_to_filtered_ids')
        for curr_cut, curr_label_ids in cut_to_shared_label_ids.items():
            label_ids_to_filtered_ids = {lbl_id: idx for idx, lbl_id in enumerate(curr_label_ids)}
            cut_to_shared_label_ids_to_filtered_ids[curr_cut] = label_ids_to_filtered_ids
        print('end starting creating label_ids_to_filtered_ids')
        # print('here we are')
        encoded_tensors_path = os.path.join(base_input_path, input_encodings_dir,
                                            curr_cut_config['encoded_tensors_path'])
        if not config['debug']:
            print('starting 3 pickle.load: ', curr_cut, '  ', encoded_tensors_path)
            loaded_tensor = pickle.load(open(encoded_tensors_path, 'rb'))
            loaded_tensor = loaded_tensor['tensors']
            loaded_tensor = torch.from_numpy(loaded_tensor)
            print('ended 3 pickle.load: ', curr_cut, '  ', encoded_tensors_path)
        else:
            # creates a random tensor of debug_size and makes sure that label_ids are lower of the nr of rows
            # of the tensor
            # and makes sure that filtered_ids are lower than the nr of rows of the tensor
            loaded_tensor = torch.rand((config['debug_size'], 1024))

        if len(loaded_label_id_wikidata_qid_map['wikidata_qid_to_label_id']) != loaded_tensor.shape[0]:
            print('!!WARNING!!, SHAPE OF SAVED TENSORS DOES NOT MATCH THE '
                  'SIZE OF THE MAP BETWEEN LABEL_IDS AND WIKIDATA_QIDS: len(loaded_label_id_wikidata_qid_map): ',
                  len(loaded_label_id_wikidata_qid_map['wikidata_qid_to_label_id']),
                  ' --- loaded_tensor.shape[0]: ', loaded_tensor.shape[0])

        # lst_shared_label_ids = list(cut_to_shared_label_ids_to_filtered_ids[curr_cut].keys())
        lst_shared_label_ids = cut_to_shared_label_ids[curr_cut]
        # from the tensor we only filter the shared label ids:
        filtered_shared_tensors = loaded_tensor[lst_shared_label_ids, :]
        cut_to_filtered_tensors[curr_cut] = filtered_shared_tensors.to(device)
        # print('encoded_tensors_path: ', encoded_tensors_path)
        # ploaded['tensors'] --> shape--> (4086473, 1024)

    return {'cut_to_label_id_wikidata_qid_map': cut_to_label_id_wikidata_qid_map,
            'cut_to_shared_label_ids_to_filtered_ids': cut_to_shared_label_ids_to_filtered_ids,
            'cut_to_filtered_tensors': cut_to_filtered_tensors}


def get_similarities_dataframe(config, device, shared_target_wikidata_qids: Set,
                               cut_to_label_id_wikidata_qid_map: Dict,
                               sorted_cuts: List,
                               cut_to_shared_label_ids_to_filtered_ids: Dict,
                               cut_to_filtered_tensors: Dict):
    cut_wikipedia_changes = list()
    reranker = BiEncoderRanker(config)
    if not config['no_cuda']:
        # device = torch.device('cuda:{}'.format(gpu_id))
        reranker.to(device)
    print('getting the dot products')
    for curr_shared_wikidata_qid in tqdm(shared_target_wikidata_qids):
        if curr_shared_wikidata_qid not in \
                cut_to_label_id_wikidata_qid_map[sorted_cuts[0]]['wikidata_qid_to_label_id']:
            print('WARNING, curr_shared_wikidata_qid not found: ', curr_shared_wikidata_qid)
            continue

        tot_nr_cuts = len(sorted_cuts)
        ####
        orig_label_id_from = cut_to_label_id_wikidata_qid_map[sorted_cuts[0]]['wikidata_qid_to_label_id'] \
            [curr_shared_wikidata_qid]
        filtered_label_id_from = cut_to_shared_label_ids_to_filtered_ids[sorted_cuts[0]][orig_label_id_from]
        representation_entity_from = cut_to_filtered_tensors[sorted_cuts[0]][filtered_label_id_from, :]
        ####
        curr_repr_from = None  # todo
        for idx_cut, curr_cut in enumerate(sorted_cuts):
            ####
            if curr_shared_wikidata_qid not in \
                    cut_to_label_id_wikidata_qid_map[curr_cut]['wikidata_qid_to_label_id']:
                print('WARNING2, curr_shared_wikidata_qid not found: ', curr_shared_wikidata_qid)
                continue

            orig_label_id_to = cut_to_label_id_wikidata_qid_map[curr_cut]['wikidata_qid_to_label_id'] \
                [curr_shared_wikidata_qid]
            if orig_label_id_from not in cut_to_shared_label_ids_to_filtered_ids[curr_cut]:
                print('WARNING3!!!')
                continue
            filtered_label_id_to = cut_to_shared_label_ids_to_filtered_ids[curr_cut][orig_label_id_from]
            representation_entity_to = cut_to_filtered_tensors[curr_cut][filtered_label_id_to, :]
            ####
            reranker.eval()
            _, scores = reranker(
                context_input=None,  # context_input.shape --> [1, 128]
                cand_input=None,
                biencoder=None,
                label_input=None,
                cand_encs=representation_entity_from.unsqueeze(0),
                context_encs=representation_entity_to.unsqueeze(0)
                # cand_encode_list[src].shape --> torch.Size([70, 768]) with the debug_size of 70
            )
            # print('shape of scores: ', scores.shape)
            # print('value of scores: ', scores.item())
            curr_dot_product = scores.item()

            cut_wikipedia_changes.append({
                'wikidata_qid': curr_shared_wikidata_qid,
                'cut_from': sorted_cuts[0],
                'cut_to': curr_cut,
                'dot_product': curr_dot_product
            })

    return cut_wikipedia_changes
