# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import torch
import torch.nn as nn
from model.conditional_random_field import CRF
from utils.vocab import Vocab
from typing import List, Dict, Tuple
from transformers import BertModel, BertTokenizerFast
from config.Constants import PAD_STATEGY, TRUNCATION_STATEGY


class NERModel(nn.Module):
    def __init__(self, vocab: Vocab, bert_path: str, device: torch.device):
        super(NERModel, self).__init__()

        self.BERT = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.bert_proj = nn.Linear(768, len(vocab))
        self.crf = CRF(len(vocab), 0)
        self.vocab = vocab
        self.device = device

    def forward(self, insts: Dict[str, List[List[str]]]):
        """forward func of the model.
        Args:
            insts: input insts, including 'snts', 'golds'
        Returns:
            loss or res
        """
        snts = insts['snts']
        ids, masks, batch_size, seq_len = self.__tokenize(snts)
        bert_embeddings = self.BERT(ids, attention_mask=masks)[0][:, 1:-1, :]  # [batch_size, seq_len, dim]
        assert (batch_size, seq_len) == bert_embeddings.shape[:-1]

        labels = insts.get('golds', None)
        if labels is not None:
            labels = self.vocab.encode(labels)
            labels = torch.tensor([
                label + [self.vocab.encode('O')]*(seq_len-len(label))
                for label in labels
            ]).to(self.device)
        crf_dict = self.crf(self.bert_proj(bert_embeddings), masks[:, 2:], labels)

        return crf_dict['loss'], crf_dict['predicted_tags']

    def __tokenize(self, snts: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        seq_len_list = [len(snt) for snt in snts]
        batch_size, seq_len = len(snts), max(seq_len_list)

        # tokenize sentences
        tokens = self.tokenizer(
            [' '.join(snt) for snt in snts], padding=PAD_STATEGY, max_length=512, truncation=TRUNCATION_STATEGY,
            return_attention_mask=True
        )
        ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        for mask, snt_len in zip(attention_mask, seq_len_list):
            assert sum(mask) == snt_len+2

        return (
            torch.tensor(ids, dtype=torch.long).to(self.device),
            torch.tensor(attention_mask, dtype=torch.long).to(self.device),
            batch_size, seq_len
        )

    def pack_state_dict(self):
        """generate state_dict when save the model.
        Returns:
            state dict
        """
        pass
