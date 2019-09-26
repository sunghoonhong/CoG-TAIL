import torch
import torch.nn.functional as F
from pytools import memoize_method
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer
import modeling_util


class CustomBertModel(BertModel):
    """
    Based on BertModel, but also outputs un-contextualized embeddings.
    """
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Based on BertModel
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=True)
        return [embedding_output] + encoded_layers

class BertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1 # from bert-base-uncased
        self.BERT_SIZE = 768 # from bert-base-uncased
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks # vocab index

    def encode_bert(self, query_tok, doc_tok):
        # query_tok includes CLS token
        # doc_tok includes SEP token
        BATCH = query_tok.shape[0]
        QLEN = 20
        DLEN = 800
        DIFF = 3
        maxlen = self.bert.config.max_position_embeddings # 512
        MAX_DLEN = maxlen - QLEN - DIFF # 489
        # 1(CLS) + 20(Q) + 1(SEP) + 489(D) + 1(SEP) = 512

        query_mask = torch.where(query_tok > 0, torch.ones_like(query_tok), torch.zeros_like(query_tok))
        doc_mask = torch.where(doc_tok > 0, torch.ones_like(doc_tok), torch.zeros_like(doc_tok))

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DLEN)
        doc_masks, _ = modeling_util.subbatch(doc_mask, MAX_DLEN)
        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_masks = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_masks[:, :1])
        NILS = torch.zeros_like(query_masks[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        segs = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        masks = torch.cat([ONES, query_masks, ONES, doc_masks, ONES], dim=1)

        # execute BERT 
        result = self.bert(toks, segs.long(), masks)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN+1] for r in result]   # (N, QLEN)
        doc_results = [r[:, QLEN+2:-1] for r in result] # (N, MAX_DLEN)
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DLEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)
            
        return query_tok, doc_tok, cls_results, query_results, doc_results


class VanillaBertRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(self.BERT_SIZE, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        self.out = torch.nn.Linear(10, 1)

    def forward(self, query_tok, doc_tok):
        _, _, cls_reps, _, _ = self.encode_bert(query_tok, doc_tok)
        x = self.dropout(cls_reps[-1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class CedrPacrrRanker(BertRanker):
    def __init__(self):
        super().__init__()
        QLEN = 20
        KMAX = 2
        NFILTERS = 32
        MINGRAM = 1
        MAXGRAM = 3
        self.simmat = modeling_util.SimmatModule()
        self.ngrams = torch.nn.ModuleList()
        self.rbf_bank = None
        for ng in range(MINGRAM, MAXGRAM+1):
            ng = modeling_util.PACRRConvMax2dModule(ng, NFILTERS, k=KMAX, channels=self.CHANNELS)
            self.ngrams.append(ng)
        qvalue_size = len(self.ngrams) * KMAX
        self.linear1 = torch.nn.Linear(self.BERT_SIZE + QLEN * qvalue_size, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, query_tok, doc_tok):
        query_tok, doc_tok, cls_reps, query_reps, doc_reps \
            = self.encode_bert(query_tok, doc_tok)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        scores = [ng(simmat) for ng in self.ngrams]
        scores = torch.cat(scores, dim=2)
        scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2])
        scores = torch.cat([scores, cls_reps[-1]], dim=1)
        rel = F.relu(self.linear1(scores))
        rel = F.relu(self.linear2(rel))
        rel = self.linear3(rel)
        return rel


class CedrKnrmRanker(BertRanker):
    def __init__(self):
        super().__init__()
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.kernels = modeling_util.KNRMRbfKernelBank(MUS, SIGMAS)
        self.fc1 = torch.nn.Linear(self.kernels.count() * self.CHANNELS + self.BERT_SIZE, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        self.out = torch.nn.Linear(10, 1)

    def forward(self, query_tok, doc_tok):
        query_tok, doc_tok, cls_reps, query_reps, doc_reps \
            = self.encode_bert(query_tok, doc_tok)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        result = torch.cat([result, cls_reps[-1]], dim=1)
        # scores = self.combine(result) # linear combination over kernels
        x = F.relu(self.fc1(result))
        x = F.relu(self.fc2(x))
        return self.out(x)


class CedrDrmmRanker(BertRanker):
    def __init__(self):
        super().__init__()
        NBINS = 11
        HIDDEN = 5
        self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.histogram = modeling_util.DRMMLogCountHistogram(NBINS)
        self.hidden_1 = torch.nn.Linear(NBINS * self.CHANNELS + self.BERT_SIZE, HIDDEN)
        self.hidden_2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, query_tok, doc_tok):
        query_tok, doc_tok, cls_reps, query_reps, doc_reps \
            = self.encode_bert(query_tok, doc_tok)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        histogram = self.histogram(simmat, doc_tok, query_tok)
        BATCH, CHANNELS, QLEN, BINS = histogram.shape
        histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
        # repeat cls representation for each query token
        cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
        output = torch.cat([output, cls_rep], dim=1)
        term_scores = self.hidden_2(F.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        return term_scores.sum(dim=1).unsqueeze(1)


RANKER = {
    'bert': VanillaBertRanker,
    'cedr_knrm': CedrKnrmRanker,
    'cedr_pacrr': CedrPacrrRanker,
    'cedr_drmm': CedrDrmmRanker
}