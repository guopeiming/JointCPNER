# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import torch
import os
import numpy as np
import torch.nn as nn
from utils.pretrain_dataset import Vocab
from typing import List, Dict, Tuple, Union
from transformers import BertModel, BertTokenizerFast
from utils.transliterate import BERT_UNREADABLE_CODES_MAPPING, TRANSLITERATIONS, BERT_TOKEN_MAPPING
from config.Constants import LANGS_NEED_SEG, PAD_STATEGY, TRUNCATION_STATEGY, CHARACTER_BASED


class PretrainBERT(nn.Module):

    def __init__(
        self,
        subtree_vocab: Vocab,
        head_vocab: Vocab,
        token_vocab: Vocab,
        subword: str,
        bert_path: str,
        transliterate: str,
        bert_emb_dropout: float,
        language: str,
        device: torch.device
    ):
        super(PretrainBERT, self).__init__()

        if language not in LANGS_NEED_SEG:
            assert subword != CHARACTER_BASED

        self.bert_encoder = BERTEncoder(subword, transliterate, bert_path, device)
        self.bert_dropout = nn.Dropout(bert_emb_dropout)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.subtree_proj = nn.Linear(self.bert_encoder.BERT.config.hidden_size, len(subtree_vocab))
        self.head_proj = nn.Linear(self.bert_encoder.BERT.config.hidden_size, len(head_vocab))
        self.token_proj = nn.Linear(self.bert_encoder.BERT.config.hidden_size, len(token_vocab))

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.subtree_vocab = subtree_vocab
        self.head_vocab = head_vocab
        self.token_vocab = token_vocab
        self.device = device

    def forward(self, insts: Dict[str, List[List[Union[str, List[int]]]]]):
        subtree_spans, snts, children_spans = insts['subtree_spans'], insts['snts'], insts['children_spans']
        mask_lm_list = self.snts_mask(snts, subtree_spans)
        words_repr = self.bert_dropout(self.bert_encoder(snts)[:, 1:-1, :])  # [batch_size, seq_len, dim]

        snt_lens = [len(snt) for snt in snts]
        batch_size, seq_len = len(snt_lens), max(snt_lens)
        assert (batch_size, seq_len) == words_repr.shape[:-1]

        # subtree spans
        subtree_repr_list = []
        subtree_label_list = []

        children_repr_list = []
        children_head_label_list = []

        mask_token_repr_list = []
        mask_lm_label_list = []
        assert len(subtree_spans) == len(children_spans) == len(mask_lm_list)
        for i, (subtree_span, children_span, mask_lm) in enumerate(zip(subtree_spans, children_spans, mask_lm_list)):
            # subtree span
            for start, end, label in subtree_span:
                subtree_repr = words_repr[i, start:end, :]
                # [1, dim]
                subtree_repr = self.pooling(subtree_repr.unsqueeze(0).permute(0, 2, 1)).squeeze(0).permute(1, 0)
                subtree_repr_list.append(subtree_repr)
                subtree_label_list.append(label)

            # children span -> head loss
            for start, end, label in children_span:
                children_repr = words_repr[i, start:end, :]
                # [1, dim]
                children_repr = self.pooling(children_repr.unsqueeze(0).permute(0, 2, 1)).squeeze(0).permute(1, 0)
                children_repr_list.append(children_repr)
                children_head_label_list.append(label)

            # mask lm
            mask_idx, mask_token_label = mask_lm[0], mask_lm[1]
            mask_token_repr_list.append(words_repr[i, mask_idx, :].unsqueeze(0))
            mask_lm_label_list.append(mask_token_label)

        assert len(subtree_repr_list) == len(subtree_label_list)
        subtrees_repr = torch.cat(subtree_repr_list, dim=0)
        subtrees_label_repr = torch.tensor(subtree_label_list, dtype=torch.long, device=self.device)
        subtree_logits = self.subtree_proj(subtrees_repr)
        subtree_loss = self.criterion(subtree_logits, subtrees_label_repr)

        assert len(children_repr_list) == len(children_head_label_list)
        childrens_repr = torch.cat(children_repr_list, dim=0)
        childrens_head_label_repr = torch.tensor(children_head_label_list, dtype=torch.long, device=self.device)
        children_head_logits = self.head_proj(childrens_repr)
        children_head_loss = self.criterion(children_head_logits, childrens_head_label_repr)

        assert len(mask_token_repr_list) == len(mask_lm_label_list)
        mask_tokens_repr = torch.cat(mask_token_repr_list, dim=0)
        mask_lm_label_repr = torch.tensor(mask_lm_label_list, dtype=torch.long, device=self.device)
        mask_lm_logits = self.token_proj(mask_tokens_repr)
        mask_lm_loss = self.criterion(mask_lm_logits, mask_lm_label_repr)

        total_loss = subtree_loss+children_head_loss+mask_lm_loss
        if self.training:
            return total_loss
        else:
            return (
                *self.get_pred(subtree_logits, children_head_logits, mask_lm_logits),
                subtree_label_list, children_head_label_list, mask_lm_label_list
            )

    def get_pred(
        self, subtree_logits: torch.Tensor,
        children_head_logits: torch.Tensor,
        mask_lm_logits: torch.Tensor
    ):
        _, subtree_label = torch.max(subtree_logits, dim=1)  # [len]
        subtree_label = subtree_label.cpu().tolist()

        _, children_head_label = torch.max(children_head_logits, dim=1)
        children_head_label = children_head_label.cpu().tolist()

        _, mask_lm_label = torch.max(mask_lm_logits, dim=1)
        mask_lm_label = mask_lm_label.cpu().tolist()
        return subtree_label, children_head_label, mask_lm_label

    def save_models(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.bert_encoder.BERT.save_pretrained(path)
        self.bert_encoder.tokenizer.save_pretrained(path)
        torch.save(
            {
                'subtree': self.subtree_proj.state_dict(),
                'head': self.head_proj.state_dict(),
                'mask_lm': self.token_proj.state_dict()
            }, os.path.join(path, 'other.pt')
        )

    def snts_mask(self, snts: List[List[str]], subtree_spans: List[List[List[int]]]) -> List[Tuple[int, int]]:
        assert len(snts) == len(subtree_spans)
        mask_list = []
        for snt, subtree_span in zip(snts, subtree_spans):
            mask_token_id = np.random.randint(0, len(snt))
            mask_token = snt[mask_token_id]
            # subtree mask
            p = np.random.rand()
            if p < 0.8:
                for idx in range(subtree_span[0][0], subtree_span[0][1]):
                    snt[idx] = self.bert_encoder.tokenizer.mask_token
            elif 0.8 <= p < 0.9:
                for idx in range(subtree_span[0][0], subtree_span[0][1], 4):
                    snt[idx] = self.token_vocab.decode(np.random.randint(0, len(self.token_vocab)))
            else:
                pass

            # language model
            p = np.random.rand()
            if p < 0.8:
                snt[mask_token_id] = self.bert_encoder.tokenizer.mask_token
            elif 0.8 <= p < 0.9:
                snt[mask_token_id] = self.token_vocab.decode(np.random.randint(0, len(self.token_vocab)))
            else:
                snt[mask_token_id] = mask_token
            mask_list.append((mask_token_id, self.token_vocab.encode(mask_token)))
        return mask_list


class BERTEncoder(nn.Module):

    def __init__(self, subword: str, transliterate: str, bert: str, device: torch.device):
        super(BERTEncoder, self).__init__()

        self.BERT = BertModel.from_pretrained(bert)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert)

        self.subword = subword
        self.device = device
        self.tranliterate = transliterate
        self.bert_hidden_size = self.BERT.config.hidden_size

        if subword == 'max_pool':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif subword == 'avg_pool':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = None

        if transliterate == '':
            self.bert_transliterate = None
        else:
            assert transliterate in TRANSLITERATIONS
            self.bert_transliterate = TRANSLITERATIONS[transliterate]

    def forward(self, snts: List[List[str]]) -> torch.Tensor:

        # BERT tokenize
        ids, bert_mask, offsets, batch_size, seq_len = self.__tokenize(snts)
        bert_embeddings = self.BERT(ids, attention_mask=bert_mask)[0]  # [batch_size, seq_len, dim]

        if self.subword != CHARACTER_BASED:
            bert_embeddings = self.__process_subword_repr(bert_embeddings, offsets, batch_size, seq_len, snts)

        return bert_embeddings  # [batch_size, seq_len+2, dim]

    def __tokenize(self, snts: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], int, int]:
        """tokenize the sentences and generate offset list.
        Args:
            snts:
        Returns:
            offsets: [batch_size, seq_len], e.g., [0, 1, 1, 1, 2, 2, 3, 4] for one sentence.
        """
        # generate batch_size, seq_len
        batch_size, seq_len = len(snts), max(len(snt) for snt in snts)+2

        # clean sentences
        snts_cleaned, offsets = [], []
        # TODO: process english and chinese, character-based and word-based
        for snt in snts:
            if self.bert_transliterate is None:
                cleaned_words = []
                for word in snt:
                    word = BERT_TOKEN_MAPPING.get(word, word)
                    word = BERT_UNREADABLE_CODES_MAPPING.get(word, word)
                    # This un-escaping for / and * was not yet added for the
                    # parser version in https://arxiv.org/abs/1812.11760v1
                    # and related model releases (e.g. benepar_en2)
                    word = word.replace('\\/', '/').replace('\\*', '*')
                    # Mid-token punctuation occurs in biomedical text
                    word = word.replace('-LSB-', '[').replace('-RSB-', ']')
                    word = word.replace('-LRB-', '(').replace('-RRB-', ')')
                    if word == "n't" and cleaned_words:
                        cleaned_words[-1] = cleaned_words[-1] + "n"
                        word = "'t"
                    cleaned_words.append(word)
            else:
                # When transliterating, assume that the token mapping is
                # taken care of elsewhere
                cleaned_words = [self.bert_transliterate(word) for word in snts]
            snts_cleaned.append(' '.join(cleaned_words))

        # tokenize sentences
        tokens = self.tokenizer(snts_cleaned, padding=PAD_STATEGY, max_length=512, truncation=TRUNCATION_STATEGY,
                                return_attention_mask=True, return_offsets_mapping=True)
        ids, attention_mask, offsets_mapping = tokens['input_ids'], tokens['attention_mask'], tokens['offset_mapping']
        if self.subword == CHARACTER_BASED:
            assert len(ids[0]) == seq_len

        # generate offsets list
        output_len = len(ids[0])
        if self.subword != CHARACTER_BASED:
            for i, offset_mapping in enumerate(offsets_mapping):
                snt = snts_cleaned[i] + ' '
                offset, word_tail_idx, word_idx = [0], snt.find(' '), 1
                for subword_head_idx, subword_tail_idx in offset_mapping[1:]:
                    if subword_tail_idx == 0:
                        offset.append(word_idx+1)
                        offset.extend([word_idx+2]*(output_len-len(offset)))
                        break
                    if subword_head_idx > word_tail_idx:
                        word_tail_idx = snt.find(' ', word_tail_idx+1)
                        assert word_tail_idx > subword_head_idx and subword_tail_idx <= word_tail_idx
                        word_idx += 1
                    offset.append(word_idx)
                assert offset[-1]+1 == len(snts[i]) + (2 if attention_mask[i][-1] != 0 else 3),\
                    'error sentence % s' % snts[i]
                offsets.append(offset)

        return (
            torch.tensor(ids, dtype=torch.long, device=self.device),
            torch.tensor(attention_mask, dtype=torch.int, device=self.device),
            offsets,
            batch_size,
            seq_len
        )

    def __process_subword_repr(self, bert_embeddings: torch.Tensor, offsets: List[List[int]], batch_size: int,
                               seq_len: int, snts: List[List[str]]) -> torch.Tensor:
        # generate start/end point idx
        offsets_tensor = torch.tensor(offsets, dtype=torch.int)
        start_expand = -torch.ones((batch_size, 1), dtype=torch.int)
        start_expand = torch.cat([start_expand, offsets_tensor], dim=1)[:, :-1]
        startpoint_idx = torch.ne((offsets_tensor - start_expand), 0)
        end_expand = torch.tensor([[offset[-1]+1] for offset in offsets], dtype=torch.int)
        end_expand = torch.cat([offsets_tensor, end_expand], dim=1)[:, 1:]
        endpoint_idx = torch.ne((offsets_tensor - end_expand), 0)

        if self.subword == 'startpoint' or self.subword == 'endpoint':
            point_idx = startpoint_idx if self.subword == 'startpoint' else endpoint_idx
            bert_embeddings = torch.masked_select(bert_embeddings, point_idx.unsqueeze(2).to(self.device))
            # [batch_size*seq_len, self.bert_hidden_size]
            bert_embeddings = bert_embeddings.reshape(-1, self.bert_hidden_size)
            assert bert_embeddings.shape[0] == torch.sum(point_idx)
            batch_length = torch.sum(point_idx, dim=1).tolist()
            assert sum(batch_length) == bert_embeddings.shape[0]
            snts_repr_list = torch.split(bert_embeddings, batch_length, dim=0)
            # In general, the sentence which generates seq_len and the sentence
            # which generates max(batch_length) are identicial. In that
            # situation, max(batch_length) == seq_len. If two different
            # sentences generate the values, max(batch_length) == seq_len + 1.
            # And we must drop the last word representation.
            assert max(batch_length) == seq_len or max(batch_length) == seq_len + 1
            if max(batch_length) == seq_len + 1:
                snts_repr_list = [snt_repr[:-1, :] for snt_repr in snts_repr_list]
            snts_repr_list = [
                torch.cat([
                    snt_repr.unsqueeze(0),
                    torch.zeros((1, seq_len-snt_repr.shape[0], snt_repr.shape[1]), device=self.device)], dim=1)
                for snt_repr in snts_repr_list
            ]
            bert_embeddings = torch.cat(snts_repr_list, dim=0)  # [batch_size, seq_len, dim]
        elif self.subword == 'max_pool' or self.subword == 'avg_pool':
            bert_embeddings_list = []
            for i in range(batch_size):
                startpoint = torch.nonzero(startpoint_idx[i], as_tuple=False).squeeze(1)
                endpoint = torch.nonzero(endpoint_idx[i], as_tuple=False).squeeze(1)
                words_length = torch.add(endpoint - startpoint, 1).tolist()
                assert len(words_length) == len(snts[i])+2 or (len(words_length) == len(snts[i])+3)
                # if do not drop the last word representation, we can not deal the situation
                # that is showed in if sentence above.
                # (if self.subword == 'startpoint' or self.subword == 'endpoint':)
                # in details.
                # the sentence is the longest raw sentence, but it is not the longes one
                # after tokenization.
                if len(words_length) == len(snts[i]) + 3:
                    last_word_len = words_length[-1]
                    words_length = words_length[:-1]
                    snt_bert_embeddings = bert_embeddings[i, :-last_word_len, :]
                else:
                    snt_bert_embeddings = bert_embeddings[i, :, :]
                assert sum(words_length) == snt_bert_embeddings.shape[0]
                words_repr_list = torch.split(snt_bert_embeddings, words_length, dim=0)
                words_repr = torch.cat(
                    [self.pool(word_repr.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1)
                     for word_repr in words_repr_list] +
                    [torch.zeros(1, seq_len-len(words_repr_list), self.bert_hidden_size, device=self.device)], dim=1
                )
                bert_embeddings_list.append(words_repr)
            bert_embeddings = torch.cat(bert_embeddings_list, dim=0)
        else:
            print('subword method (%s) is illegal' % self.subword)
        return bert_embeddings
