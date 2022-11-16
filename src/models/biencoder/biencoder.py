import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.custom_bert import BertModel
from models.utils.optimizer import get_bert_optimizer
from models.utils.ranker_base import BertEncoder

NULL_IDX = 0


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoder(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoder, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not params['no_cuda'] else 'cpu'
        )
        ctxt_bert = BertModel.from_pretrained(params['bert_model'], cache_dir=params['bert_cache_dir'])
        cand_bert = BertModel.from_pretrained(params['bert_model'], cache_dir=params['bert_cache_dir'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params['out_dim'],
            layer_pulled=params['pull_from_layer'],
            add_linear=params['add_linear'],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params['out_dim'],
            layer_pulled=params['pull_from_layer'],
            add_linear=params['add_linear'],
        )
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
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


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not params['no_cuda'] else 'cpu'
        )
        self.n_gpu = torch.cuda.device_count()
        self.START_TOKEN = '[CLS]'
        self.END_TOKEN = '[SEP]'

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params['type_optimization'],
            self.params['learning_rate'],
            fp16=self.params.get('fp16'),
        )

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            context_encs=None,
            cand_encs=None,  # pre-computed candidate encoding.
            biencoder: BiEncoder = None
    ):
        if context_encs is None:
            # Encode contexts first
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
                text_vecs, NULL_IDX
            )
            embedding_ctxt, _ = biencoder(
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
        _, embedding_cands = biencoder(
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
    def forward(self, context_input, cand_input, biencoder: BiEncoder = None,
                label_input=None, cand_encs=None, context_encs=None):
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
        scores = self.score_candidate(context_input, cand_input, flag, cand_encs=cand_encs, context_encs=context_encs,
                                      biencoder=biencoder)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction='mean')
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            loss = loss_fct(scores, label_input)
        return loss, scores


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
