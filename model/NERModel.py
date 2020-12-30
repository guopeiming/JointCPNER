# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import torch
import numpy as np
import torch.nn as nn
from model.conditional_random_field import CRF
from utils.vocab import Vocab
from typing import List, Dict, Tuple, Set
from transformers import BertModel, BertTokenizerFast
from utils.transliterate import TRANSLITERATIONS, BERT_TOKEN_MAPPING
from config.Constants import NER_LABELS, PAD_STATEGY, TRUNCATION_STATEGY, CHARACTER_BASED
from model.transformer import Transformer, LearnedPositionalEmbedding
from model.partition_transformer import PartitionTransformer


class NERSLModel(nn.Module):
    def __init__(self, vocab: Vocab, bert_path: str, device: torch.device):
        super(NERSLModel, self).__init__()

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
            ], device=self.device)
        crf_dict = self.crf(self.bert_proj(bert_embeddings), masks[:, 2:], labels)

        if not self.training:
            return self.vocab.decode(crf_dict['predicted_tags'])
        else:
            return crf_dict['loss']

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
            torch.tensor(ids, dtype=torch.long, device=self.device),
            torch.tensor(attention_mask, dtype=torch.long, device=self.device),
            batch_size, seq_len
        )

    def pack_state_dict(self):
        """generate state_dict when save the model.
        Returns:
            state dict
        """
        pass


class NERSpanModel(nn.Module):
    def __init__(self, subword, transliterate, bert_path: str, position_emb_dropout: float,
                 bert_emb_dropout: float, emb_dropout: float, d_model: int, partition: bool, layer_num: int,
                 hidden_dropout: float, attention_dropout: float, dim_ff: int, nhead: int, kqv_dim: int,
                 label_hidden: int, device: torch.device):
        super(NERSpanModel, self).__init__()

        self.encoder = NERSpanEncoder(
            subword, transliterate, bert_path, d_model, partition, position_emb_dropout, bert_emb_dropout, emb_dropout,
            layer_num, hidden_dropout, attention_dropout, dim_ff, nhead, kqv_dim, device
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(d_model, label_hidden),
            nn.LayerNorm(label_hidden),
            nn.ReLU(),
            nn.Linear(label_hidden, len(NER_LABELS))
        )
        self.vocab = NER_LABELS
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device

    def forward(self, insts: Dict[str, List[List[str]]]):
        """forward func of the model.
        Args:
            insts: input insts, including 'snts', 'golds'
        Returns:
            loss or res
        """
        snts = insts['snts']
        snt_lens = [len(snt) for snt in snts]
        batch_size, seq_len = len(snt_lens), max(snt_lens)
        words_repr = self.encoder(snts)
        assert (batch_size, seq_len) == words_repr.shape[:-1]

        # [batch_size, seq_len, seq_len, dim]
        spans_repr = words_repr.unsqueeze(1) - words_repr.unsqueeze(2)
        label_score = self.label_classifier(spans_repr)
        empty_label = torch.full((batch_size, seq_len, seq_len, 1), 0., device=self.device)
        label_score = torch.cat([label_score, empty_label], dim=-1)

        # during test, return tags list
        if not self.training:
            label_score = nn.functional.softmax(label_score, dim=-1)
            res_tags = []
            for i in range(batch_size):
                res_tag = self.generate_tags(label_score[i], snt_lens[i])
                res_tags.append(res_tag)
            return res_tags

        # during train, return loss tensor
        gold_tags = insts['golds']
        assert len(gold_tags) == batch_size
        # batch_ids, g_is, g_js, g_labels = [], [], [], []
        # for idx, gold_tag in enumerate(gold_tags):
        #     spans = self.generate_spans(gold_tag)
        #     for label_idx, (start_i, end_j) in spans:
        #         batch_ids.append(idx)
        #         g_is.append(start_i)
        #         g_js.append(end_j-1)
        #         g_labels.append(label_idx)

        # target = torch.tensor(g_labels, dtype=torch.long, device=self.device)
        # loss = self.criterion(label_score[batch_ids, g_is, g_js, :], target)

        spans_mask = [
            [[0]*i + [1]*(snt_len-i) + [0]*(seq_len-snt_len) if i < snt_len else [0]*seq_len for i in range(seq_len)]
            for snt_len in snt_lens
        ]
        spans_mask_tensor = torch.tensor(spans_mask, dtype=torch.bool, device=self.device).unsqueeze(3)
        spans_label_idx = []
        for idx, gold_tag in enumerate(gold_tags):
            label_idx_np = np.full((snt_lens[idx], snt_lens[idx]), len(self.vocab), dtype=np.int)
            spans = self.generate_spans(gold_tag)
            for label_idx, (start_i, end_j) in spans:
                label_idx_np[start_i, end_j-1] = label_idx
            for i in range(snt_lens[idx]):
                spans_label_idx.extend(label_idx_np[i, i:].tolist())
        assert np.sum(np.array(spans_mask)) == len(spans_label_idx)

        target = torch.tensor(spans_label_idx, dtype=torch.long, device=self.device)
        loss = self.criterion(torch.masked_select(label_score, spans_mask_tensor).view(-1, len(self.vocab)+1), target)
        return loss

    def generate_res_spans(self, label_score: torch.Tensor, snt_len: int) -> Set[Tuple[str, Tuple[int, int]]]:
        res_spans = set()
        _, label_tensor = torch.max(label_score, dim=-1)
        label_list = label_tensor.cpu().numpy()
        for i in range(snt_len):
            for j in range(i, snt_len):
                if label_list[i, j] != len(self.vocab):
                    res_spans.add((self.vocab[label_list[i, j]], (i, j+1)))
        return res_spans

    def generate_tags(self, label_score: torch.Tensor, snt_len: int) -> List[str]:
        # [i, j]
        res_tag = ['O' for _ in range(snt_len)]
        _, label_tensor = torch.max(label_score, dim=-1)
        label_list = label_tensor.cpu().numpy()
        for m in range(snt_len-1, -1, -1):
            for i in range(0, snt_len-m):
                if label_list[i, i+m] != len(self.vocab):
                    res_tag[i] = 'B-'+self.vocab[label_list[i, i+m]]
                    for k in range(i+1, i+m+1):
                        res_tag[k] = 'I-'+self.vocab[label_list[i, i+m]]
        return res_tag

    def generate_spans(self, tags: List[str], ignore_labels=None) -> Set[Tuple[str, Tuple[int, int]]]:
        """给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O'].
            返回[('singer', (1, 4))] (左闭右开区间)
        Args:
            tags: List[str],
            ignore_labels: List[str], 在该list中的label将被忽略

        Returns:
            Set[Tuple[str, Tuple[int, int]]]. {(label，(start, end))}
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bio_tag = None
        for idx, tag in enumerate(tags):
            bio_tag, label = tag[:1], tag[2:]
            if bio_tag == 'B':
                spans.append((self.vocab.index(label), [idx, idx]))
            elif bio_tag == 'I':
                if prev_bio_tag in ('B', 'I') and len(spans) > 0 and self.vocab.index(label) == spans[-1][0]:
                    spans[-1][1][1] = idx
            elif bio_tag == 'O':  # o tag does not count
                pass
            else:
                print('error %s tag in BIO tags' % bio_tag)
            prev_bio_tag = bio_tag
        return {(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels}

    def pack_state_dict(self):
        """generate state_dict when save the model.
        Returns:
            state dict
        """
        pass


class NERSpanEncoder(nn.Module):
    def __init__(self, subword: str, transliterate: str, bert_path: str, d_model: int, partition: bool,
                 position_emb_dropout: float, bert_emb_dropout: float, emb_dropout: float, layer_num: int,
                 hidden_dropout: float, attention_dropout: float, dim_ff: int, nhead: int, kqv_dim: int,
                 device: torch.device):
        super(NERSpanEncoder, self).__init__()

        self.BERT = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.bert_hidden_size = self.BERT.config.hidden_size

        self.partition = partition
        self.d_model = d_model
        self.d_content = d_model // 2 if partition else d_model
        self.d_position = d_model - d_model//2 if partition else d_model
        self.subword = subword
        self.device = device

        if subword == 'max_pool':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif subword == 'avg_pool':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = None

        self.position_emb_dropout = nn.Dropout(position_emb_dropout)
        self.bert_emb_dropout = nn.Dropout(bert_emb_dropout)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)

        self.bert_proj = nn.Linear(self.bert_hidden_size, self.d_content)

        self.position_embeddings = LearnedPositionalEmbedding(self.d_position, max_len=512)

        if transliterate == '':
            self.bert_transliterate = None
        else:
            assert transliterate in TRANSLITERATIONS
            self.bert_transliterate = TRANSLITERATIONS[transliterate]

        if self.partition:
            self.transf = PartitionTransformer(d_model, layer_num, nhead, dim_ff, hidden_dropout, attention_dropout,
                                               'relu', kqv_dim=kqv_dim)
        else:
            self.transf = Transformer(d_model, layer_num, nhead, dim_ff, hidden_dropout, attention_dropout, 'relu',
                                      kqv_dim=kqv_dim)

    def forward(self, snts: List[List[str]]) -> torch.Tensor:

        # BERT tokenize
        ids, bert_mask, snts_mask, offsets = self.__tokenize(snts)
        batch_size, seq_len = snts_mask.shape
        bert_embeddings = self.BERT(ids, attention_mask=bert_mask)[0]  # [batch_size, seq_len, dim]
        if self.subword != CHARACTER_BASED:
            bert_embeddings = self.__process_subword_repr(bert_embeddings, offsets, batch_size, seq_len, snts)
        content_embeddings = self.bert_proj(self.bert_emb_dropout(bert_embeddings))
        position_embeddings = self.position_emb_dropout(self.position_embeddings(batch_size, seq_len))
        if self.partition:
            embeddings = torch.cat([content_embeddings, position_embeddings.expand(batch_size, -1, -1)], dim=2)
        else:
            embeddings = torch.add(content_embeddings+position_embeddings)
        assert embeddings.shape[2] == self.d_model
        embeddings = self.emb_dropout(self.layer_norm(embeddings))

        # [batch_size, seq_len+2, dim]
        hidden_states = self.transf(hidden_states=embeddings, attention_mask=snts_mask)[0]
        if self.partition:
            # Rearrange the annotations to ensure that the transition to
            # fenceposts captures an even split between position and content.
            # TODO(nikita): try alternatives, such as omitting position entirely
            hidden_states = torch.cat([
                hidden_states[:, :, 0::2],
                hidden_states[:, :, 1::2],
            ], 2)
        # [batch_size, seq_len, dim]
        words_repr = torch.cat([
                hidden_states[:, :-2, :self.d_model//2], hidden_states[:, 1:-1, self.d_model//2:]
            ], dim=2)
        return words_repr

    def __tokenize(self, snts: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
        """tokenize the sentences and generate offset list.
        Args:
            snts:
        Returns:
            offsets: [batch_size, seq_len], e.g., [0, 1, 1, 1, 2, 2, 3, 4] for one sentence.
        """
        # generate sents mask
        max_len = max(len(snt) for snt in snts)
        snts_mask = [[1]*(len(snt)+2)+[0]*(max_len-len(snt)) for snt in snts]

        # clean sentences
        snts_cleaned, offsets = [], []
        # TODO: process english and chinese, character-based and word-based
        for snt in snts:
            if self.bert_transliterate is None:
                cleaned_words = []
                for word in snt:
                    word = BERT_TOKEN_MAPPING.get(word, word)
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
            assert len(ids[0]) == max_len+2

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
                offsets.append(offset)

        return (
            torch.tensor(ids, dtype=torch.long, device=self.device),
            torch.tensor(attention_mask, dtype=torch.int, device=self.device),
            torch.tensor(snts_mask, dtype=torch.int, device=self.device),
            offsets
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
                    torch.zeros((seq_len-snt_repr.shape[0], snt_repr.shape[1])).unsqueeze(0).to(self.device)], dim=1)
                for snt_repr in snts_repr_list]
            bert_embeddings = torch.cat(snts_repr_list, dim=0)  # [batch_size, seq_len, dim]
        elif self.subword == 'max_pool' or self.subword == 'avg_pool':
            bert_embeddings_list = []
            for i in range(batch_size):
                startpoint = torch.nonzero(startpoint_idx[i]).squeeze(1)
                endpoint = torch.nonzero(endpoint_idx[i]).squeeze(1)
                words_length = torch.add(endpoint - startpoint, 1).tolist()
                assert len(words_length) == len(snts[i]+2) or (len(words_length) == len(snts[i])+3)
                # if do not drop the last word representation, we can not deal the situation
                # that is showed in if sentence above.
                # (if self.subword == 'startpoint' or self.subword == 'endpoint':)
                # in details.
                # the sentence is the longest raw sentence, but it is not the longes one
                # after tokenization.
                if len(words_length) == snts[i] + 3:
                    last_word_len = words_length[-1]
                    words_length = words_length[:-1]
                    snt_bert_embeddings = bert_embeddings[i, :-last_word_len, :]
                else:
                    snt_bert_embeddings = bert_embeddings[i, :, :]
                assert sum(words_length) == snt_bert_embeddings.shape[0]
                words_repr_list = torch.split(snt_bert_embeddings, words_length, dim=0)
                words_repr = torch.cat([
                    self.pool(word_repr.permute(1, 0)).permute(1, 0) for word_repr in words_repr_list] +
                    [torch.zeros(seq_len-len(words_repr_list), self.bert_hidden_size, device=self.device)], dim=0)
                bert_embeddings_list.append(words_repr.unsqueeze(0))
            bert_embeddings = torch.cat(bert_embeddings_list, dim=0)
        else:
            print('subword method (%s) is illegal' % self.subword)
        return bert_embeddings
