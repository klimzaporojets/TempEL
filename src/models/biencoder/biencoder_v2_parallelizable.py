# The objective of this module is to build biencoder ranker that can be executed in parallel using DistributedDataParallel

import torch

from models.biencoder.biencoder import to_bert_input, NULL_IDX
from models.utils.custom_bert import BertModel
from models.utils.optimizer import get_bert_optimizer
from models.utils.ranker_base import BertEncoder


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRankerV2(params)
    return biencoder


class BiEncoderRankerV2(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderRankerV2, self).__init__()
        self.params = params
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not params['no_cuda'] else 'cpu'
        )
        self.n_gpu = torch.cuda.device_count()
        self.START_TOKEN = '[CLS]'
        self.END_TOKEN = '[SEP]'

        #### BEGIN: previously in BiEncoder:
        ctxt_bert = BertModel.from_pretrained(params['bert_model'], cache_dir=params['bert_cache_dir'])
        cand_bert = BertModel.from_pretrained(params['bert_model'], cache_dir=params['bert_cache_dir'])
        self.config = ctxt_bert.config

        self.context_encoder = BertEncoder(
            ctxt_bert,
            params['out_dim'],
            layer_pulled=params['pull_from_layer'],
            add_linear=params['add_linear'],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            # ctxt_bert,
            params['out_dim'],
            layer_pulled=params['pull_from_layer'],
            add_linear=params['add_linear'],
        )
        #### END: previously in BiEncoder:

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params['type_optimization'],
            self.params['learning_rate'],
            fp16=self.params.get('fp16'),
        )

    def run_biencoder(self, token_idx_ctxt,
                      segment_idx_ctxt,
                      mask_ctxt,
                      token_idx_cands,
                      segment_idx_cands,
                      mask_cands, ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            context_encs=None,
            cand_encs=None
    ):
        if context_encs is None:
            # Encode contexts first
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
                text_vecs, NULL_IDX
            )
            embedding_ctxt, _ = self.run_biencoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
            )
        else:
            embedding_ctxt = context_encs
        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, NULL_IDX
        )
        _, embedding_cands = self.run_biencoder(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None, cand_encs=None, context_encs=None):
        """

        :param context_input:
        :param cand_input:
        :param label_input:
        :param cand_encs: (kzaporoj) --> will only come as input during evaluation from already pre-loaded encodings
                                        This should evoid calculating BERT encoding when predicting (time consuming).
                                        During training, cand_input is passed and cand_encs will be None.
        :return:
        """
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag, cand_encs=cand_encs, context_encs=context_encs)
        return scores
