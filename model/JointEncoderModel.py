# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import torch
from queue import Queue
import numpy as np
import torch.nn as nn
from utils import chart_helper
from utils import trees
from utils.vocabulary import Vocabulary
from utils.transliterate import TRANSLITERATIONS, BERT_TOKEN_MAPPING
from typing import List, Dict, Union, Tuple
from utils.trees import InternalParseNode, LeafParseNode
from transformers import BertModel, BertTokenizerFast
from config.Constants import NER_LABELS, STOP, START, TAG_UNK, PAD_STATEGY, TRUNCATION_STATEGY, CHARACTER_BASED
from model.transformer import LearnedPositionalEmbedding, Transformer
from model.partition_transformer import PartitionTransformer


class JointEncoderModel(nn.Module):
    def __init__(self, joint_vocabs: Dict[str, Vocabulary], parsing_vocabs: Dict[str, Vocabulary],
                 #  cross_labels_idx: Dict[str, Tuple[int]],
                 subword: str, use_pos_tag: bool, bert_path: str,
                 transliterate: str, d_model: int, partition: bool, pos_tag_emb_dropout: float,
                 position_emb_dropout: float, bert_emb_dropout: float, emb_dropout: float, layer_num: int,
                 hidden_dropout: float, attention_dropout: float, dim_ff: int, nhead: int, kqv_dim: int,
                 label_hidden: int, lambda_scaler: float, alpha_scaler: float, language: str, device: torch.device):
        super(JointEncoderModel, self).__init__()

        self.embeddings = EmbeddingLayer(
            joint_vocabs, subword, use_pos_tag, bert_path, transliterate, d_model, partition, pos_tag_emb_dropout,
            position_emb_dropout, bert_emb_dropout, emb_dropout, language, device
        )
        self.encoder = Encoder(
            d_model, partition, layer_num, hidden_dropout, attention_dropout, dim_ff, nhead, kqv_dim, device
        )
        self.joint_label_classifier = nn.Sequential(
            nn.Linear(d_model, label_hidden),
            nn.LayerNorm(label_hidden),
            nn.ReLU(),
            nn.Linear(label_hidden, joint_vocabs['labels'].size-1)
        )

        self.parsing_label_classifier = nn.Sequential(
            nn.Linear(d_model, label_hidden),
            nn.LayerNorm(label_hidden),
            nn.ReLU(),
            nn.Linear(label_hidden, parsing_vocabs['labels'].size-1)
        )

        self.ner_label_classifier = nn.Sequential(
            nn.Linear(d_model, label_hidden),
            nn.LayerNorm(label_hidden),
            nn.ReLU(),
            nn.Linear(label_hidden, len(NER_LABELS))
        )
        self.criterion_ner = nn.CrossEntropyLoss(reduction='sum')

        self.softmax = nn.Softmax(dim=-1)
        self.pos_tags_vocab = joint_vocabs['pos_tags']
        self.joint_labels_vocab = joint_vocabs['labels']
        self.parsing_labels_vocab = parsing_vocabs['labels']
        # self.cross_label_idx = cross_labels_idx
        self.lambda_scaler = lambda_scaler
        self.alpha_scaler = alpha_scaler
        self.device = device

    def forward(self, insts: Dict[str, List[Union[List[str], InternalParseNode]]], return_charts: bool = False)\
            -> Union[torch.Tensor, Tuple[List[InternalParseNode], List[np.ndarray]], List[np.ndarray]]:
        """forward func of the model.
        Args:
            insts: input insts, including 'pos_tags', 'snts', 'gold_trees'
        Returns:
            pred tensor outputed by model
        """
        pos_tags, snts = insts['pos_tags'], insts['snts']
        snts_len = [len(pos_tag) for pos_tag in pos_tags]
        batch_size, seq_len = len(snts_len), max(snts_len)
        embeddings, mask = self.embeddings(pos_tags, snts)
        assert (batch_size, seq_len + 2) == embeddings.shape[0:2] == mask.shape[0:2]
        words_repr = self.encoder(embeddings, mask)
        assert (batch_size, seq_len+1) == words_repr.shape[0:2]
        spans_repr = words_repr.unsqueeze(1) - words_repr.unsqueeze(2)  # [batch_size, seq_len+1, seq_len+1, dim]
        assert (batch_size, seq_len+1, seq_len+1) == spans_repr.shape[0:3]

        joint_labels_score = self.joint_label_classifier(spans_repr)
        parsing_labels_score = self.parsing_label_classifier(spans_repr)
        ner_labels_score = self.ner_label_classifier(spans_repr)

        empty_label_score = torch.zeros((batch_size, seq_len+1, seq_len+1, 1), device=self.device)
        joint_charts = torch.cat([empty_label_score, joint_labels_score], dim=3)
        parsing_charts = torch.cat([empty_label_score, parsing_labels_score], dim=3)
        empty_label_score = torch.full((batch_size, seq_len+1, seq_len+1, 1), 0., device=self.device)
        ner_labels_score = torch.cat([ner_labels_score, empty_label_score], dim=3)
        joint_charts_np = joint_charts.cpu().detach().numpy()
        parsing_charts_np = parsing_charts.cpu().detach().numpy()

        # compute loss and generate tree

        # Just return the charts, for ensembling
        if return_charts:
            ret_charts = []
            for i, snt_len in enumerate(snts_len):
                ret_charts.append(joint_charts[i, :snt_len+1, :snt_len+1, :].cpu().numpy())
            return ret_charts

        # when model test, just return trees and scores
        if not self.training:
            trees = []
            scores = []
            for i, snt_len in enumerate(snts_len):
                chart_np = joint_charts_np[i, :snt_len+1, :snt_len+1, :]
                score, p_i, p_j, p_label, _ = self.parse_from_chart(snt_len, chart_np, self.joint_labels_vocab)
                pos_tag, snt = pos_tags[i], snts[i]
                tree = self.generate_tree(p_i, p_j, p_label, pos_tag, snt)
                trees.append(tree)
                scores.append(score)
            return trees, scores

        # when model train, return loss
        # During training time, the forward pass needs to be computed for every
        # cell of the chart, but the backward pass only needs to be computed for
        # cells in either the predicted or the gold parse tree. It's slightly
        # faster to duplicate the forward pass for a subset of the chart than it
        # is to perform a backward pass that doesn't take advantage of sparsity.
        # Since this code is not undergoing algorithmic changes, it makes sense
        # to include the optimization even though it may only be a 10% speedup.
        # Note that no dropout occurs in the label portion of the network
        # cross_loss = torch.tensor(0., device=self.device)
        joint_golds = insts['joint_gold_trees']
        parsing_golds = insts['parsing_gold_trees']
        p_is, p_js, g_is, g_js, p_labels, g_labels, batch_ids, paugment_total_joint = [], [], [], [], [], [], [], 0.0
        p_is_parsing, p_js_parsing, g_is_parsing, g_js_parsing, p_labels_parsing, g_labels_parsing, batch_ids_parsing,\
            paugment_total_parsing = [], [], [], [], [], [], [], 0.0
        ner_is, ner_js, ner_labels, ner_batch_ids = [], [], [], []
        for i, snt_len in enumerate(snts_len):

            # joint parser
            chart_np = joint_charts_np[i, :snt_len+1, :snt_len+1, :]
            p_i, p_j, p_label, p_augment, g_i, g_j, g_label =\
                self.parse_from_chart(snt_len, chart_np, self.joint_labels_vocab, joint_golds[i])
            paugment_total_joint += p_augment
            p_is.extend(p_i.tolist())
            p_js.extend(p_j.tolist())
            p_labels.extend(p_label.tolist())
            g_is.extend(g_i.tolist())
            g_js.extend(g_j.tolist())
            g_labels.extend(g_label.tolist())
            batch_ids.extend([i for _ in range(len(p_i))])

            # parsing parser
            chart_np = parsing_charts_np[i, :snt_len+1, :snt_len+1, :]
            p_i, p_j, p_label, p_augment, g_i, g_j, g_label =\
                self.parse_from_chart(snt_len, chart_np, self.parsing_labels_vocab, parsing_golds[i])
            paugment_total_parsing += p_augment
            p_is_parsing.extend(p_i.tolist())
            p_js_parsing.extend(p_j.tolist())
            p_labels_parsing.extend(p_label.tolist())
            g_is_parsing.extend(g_i.tolist())
            g_js_parsing.extend(g_j.tolist())
            g_labels_parsing.extend(g_label.tolist())
            batch_ids_parsing.extend([i for _ in range(len(p_i))])

            # cross loss
            # cross_spans = self.generate_cross_label_spans(golds[i])
            # for constit, constit_gold, ner, ner_gold, span_start, span_end in cross_spans:
            #     constit_idx = self.cross_label_idx[constit]
            #     ner_idx = self.cross_label_idx[ner]
            #     cross_constit_loss = self.log_softmax(charts[i, span_start, span_end, constit_idx])[constit_gold]
            #     cross_ner_loss = self.log_softmax(charts[i, span_start, span_end, ner_idx])[ner_gold]
            #     cross_loss = cross_loss - cross_constit_loss - cross_ner_loss

            # ner idx
            ner_i, ner_j, ner_label = self.generate_ner_spans(joint_golds[i])
            ner_is.extend(ner_i)
            ner_js.extend(ner_j)
            ner_labels.extend(ner_label)
            ner_batch_ids.extend([i for _ in range(len(ner_i))])

        p_scores_joint = torch.sum(joint_charts[batch_ids, p_is, p_js, p_labels])
        g_scores_joint = torch.sum(joint_charts[batch_ids, g_is, g_js, g_labels])
        loss_joint = p_scores_joint - g_scores_joint + paugment_total_joint

        p_scores_parsing = torch.sum(parsing_charts[batch_ids_parsing, p_is_parsing, p_js_parsing, p_labels_parsing])
        g_scores_parsing = torch.sum(parsing_charts[batch_ids_parsing, g_is_parsing, g_js_parsing, g_labels_parsing])
        loss_parsing = p_scores_parsing - g_scores_parsing + paugment_total_parsing

        ner_score: torch.Tensor = ner_labels_score[ner_batch_ids, ner_is, ner_js, :]
        assert ner_score.shape[0] == len(ner_labels)
        # ner_loss = torch.sum(torch.log2(self.softmax(ner_score)[[i for i in range(len(ner_labels))], ner_labels]))
        ner_loss = self.criterion_ner(ner_score, torch.tensor(ner_labels, dtype=torch.long, device=self.device))

        loss = loss_joint + self.lambda_scaler*(self.alpha_scaler*loss_parsing + (1.-self.alpha_scaler)*ner_loss)
        return loss

    # def generate_cross_label_spans(self, tree: InternalParseNode) -> List[Tuple[str, int, str, int, int, int]]:
    #     ner_spans = []
    #     q = Queue()
    #     q.put(tree)
    #     while not q.empty():
    #         tree = q.get()

    #         # generate ner spans
    #         for label in tree.label:
    #             if label.endswith('*'):
    #                 label_splited_list = label[:-1].split('-')
    #                 constitent, ner = '-'.join(label_splited_list[:-1]), label_splited_list[-1]
    #                 assert constitent in self.cross_label_idx and ner in self.cross_label_idx

    #                 tree_label_idx = self.labels_vocab.index(tree.label)
    #                 assert tree_label_idx in self.cross_label_idx[constitent]
    #                 assert tree_label_idx in self.cross_label_idx[ner]

    #                 constitent_gold = self.cross_label_idx[constitent].index(tree_label_idx)
    #                 ner_gold = self.cross_label_idx[ner].index(tree_label_idx)
    #                 ner_spans.append((constitent, constitent_gold, ner, ner_gold, tree.left, tree.right))

    #         for child in tree.children:
    #             assert isinstance(child, InternalParseNode) or isinstance(child, LeafParseNode)
    #             if isinstance(child, InternalParseNode):
    #                 q.put(child)

    #     return ner_spans

    def generate_ner_spans(self, tree: InternalParseNode):
        ner_i, ner_j, ner_label = [], [], []
        q = Queue()
        q.put(tree)
        while not q.empty():
            tree = q.get()

            # generate ner spans
            for label in tree.label:
                if label.endswith('*'):
                    ner = label[:-1].split('-')[-1]
                    assert ner in NER_LABELS
                    ner_i.append(tree.left)
                    ner_j.append(tree.right)
                    ner_label.append(NER_LABELS.index(ner))

            for child in tree.children:
                assert isinstance(child, InternalParseNode) or isinstance(child, LeafParseNode)
                if isinstance(child, InternalParseNode):
                    q.put(child)
        return ner_i, ner_j, ner_label

    def parse_from_chart(self, snt_len: int, chart_np: np.ndarray, vocab, gold=None):
        decoder_args = dict(
            sentence_len=snt_len,
            label_scores_chart=chart_np,
            gold=gold,
            label_vocab=vocab,
            is_train=self.training
        )

        p_score, p_i, p_j, p_label, p_augment = chart_helper.decode(False, **decoder_args)
        if self.training:
            g_score, g_i, g_j, g_label, g_augment = chart_helper.decode(True, **decoder_args)
            return p_i, p_j, p_label, p_augment, g_i, g_j, g_label
        else:
            return p_score, p_i, p_j, p_label, p_augment

    def generate_tree(self, p_i, p_j, p_label, pos_tag, snt):
        idx = -1

        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.joint_labels_vocab.value(label_idx)
            if (i + 1) >= j:
                tag, word = pos_tag[i], snt[i]
                tree = trees.LeafParseNode(int(i), tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label:
                    return [trees.InternalParseNode(label, children)]
                else:
                    return children

        tree = make_tree()[0]
        return tree

    def pack_state_dict(self):
        """generate state_dict when save the model.
        Returns:
            state dict
        """
        pass


class EmbeddingLayer(nn.Module):

    def __init__(self, vocabs: Dict[str, Vocabulary], subword: str, use_pos_tag: bool, bert_path: str,
                 transliterate: str, d_model: int, partition: bool, pos_tag_emb_dropout: float,
                 position_emb_dropout: float, bert_emb_dropout: float, emb_dropout: float, language: str,
                 device: torch.device):
        super(EmbeddingLayer, self).__init__()

        self.BERT = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.bert_hidden_size = self.BERT.config.hidden_size
        if subword == 'max_pool':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif subword == 'avg_pool':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = None

        self.position_emb_dropout = nn.Dropout(position_emb_dropout)
        self.bert_emb_dropout = nn.Dropout(bert_emb_dropout)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)

        self.partition = partition
        self.d_model = d_model
        self.d_content = d_model // 2 if partition else d_model
        self.d_position = d_model - d_model//2 if partition else d_model
        self.bert_proj = nn.Linear(self.bert_hidden_size, self.d_content)
        self.pos_tag_embeddings = None
        self.pos_tag_emb_dropout = None
        self.use_pos_tag = use_pos_tag
        if use_pos_tag:
            self.pos_tag_emb_dropout = nn.Dropout(pos_tag_emb_dropout)
            self.pos_tag_embeddings = nn.Embedding(vocabs['pos_tags'].size+1, self.d_content,
                                                   padding_idx=vocabs['pos_tags'].size)

        self.position_embeddings = LearnedPositionalEmbedding(self.d_position, max_len=512)

        if transliterate == '':
            self.bert_transliterate = None
        else:
            assert transliterate in TRANSLITERATIONS
            self.bert_transliterate = TRANSLITERATIONS[transliterate]
        self.subword = subword
        self.language = language
        if self.subword == CHARACTER_BASED and self.language == 'chinese':
            assert not self.use_pos_tag

        self.pos_tags_vocab = vocabs['pos_tags']
        self.labels_vocab = vocabs['labels']
        self.device = device

    def forward(self, pos_tags: List[List[str]], snts: List[List[str]]):
        # BERT tokenize
        ids, pos_tags_ids, bert_mask, snts_mask, offsets = self.__tokenize(pos_tags, snts)
        batch_size, seq_len = snts_mask.shape
        bert_embeddings = self.BERT(ids, attention_mask=bert_mask)[0]  # [batch_size, seq_len, dim]
        if self.subword != CHARACTER_BASED:
            bert_embeddings = self.__process_subword_repr(bert_embeddings, offsets, batch_size, seq_len, snts)
        content_embeddings = self.bert_proj(self.bert_emb_dropout(bert_embeddings))
        if self.use_pos_tag:
            pos_tags_embeddings = self.pos_tag_emb_dropout(self.pos_tag_embeddings(pos_tags_ids))
            content_embeddings = torch.add(content_embeddings, pos_tags_embeddings)
        position_embeddings = self.position_emb_dropout(self.position_embeddings(batch_size, seq_len))
        if self.partition:
            embeddings = torch.cat([content_embeddings, position_embeddings.expand(batch_size, -1, -1)], dim=2)
        else:
            embeddings = torch.add(content_embeddings+position_embeddings)
        assert embeddings.shape[2] == self.d_model
        return self.emb_dropout(self.layer_norm(embeddings)), snts_mask

    def __tokenize(self, pos_tags: List[List[str]], snts: List[List[str]])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
        """tokenize the sentences and generate offset list and pos tags id.
        Args:
            pos_tags:
            snts:
        Returns:
            offsets: [batch_size, seq_len], e.g., [0, 1, 1, 1, 2, 2, 3, 4] for one sentence.
        """
        # generate pos tag ids
        pos_tags_ids = None
        if self.use_pos_tag:
            max_len = max(len(pos_tag) for pos_tag in pos_tags)
            pos_tags_ids = [[self.pos_tags_vocab.index_or_unk(tag, TAG_UNK)
                            for tag in [START]+pos_tag+[STOP]] + [self.pos_tags_vocab.size]*(max_len-len(pos_tag))
                            for pos_tag in pos_tags]

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
            torch.tensor(pos_tags_ids, dtype=torch.long, device=self.device) if self.use_pos_tag else None,
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


class Encoder(nn.Module):
    def __init__(self, d_model: int, partition: bool, layer_num: int, hidden_dropout: float, attention_dropout: float,
                 dim_ff: int, nhead: int, kqv_dim: int, device: torch.device):
        super(Encoder, self).__init__()

        self.partition = partition
        self.d_model = d_model
        if self.partition:
            self.transf = PartitionTransformer(d_model, layer_num, nhead, dim_ff, hidden_dropout, attention_dropout,
                                               'relu', kqv_dim=kqv_dim)
        else:
            self.transf = Transformer(d_model, layer_num, nhead, dim_ff, hidden_dropout, attention_dropout, 'relu',
                                      kqv_dim=kqv_dim)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transf(hidden_states=embeddings, attention_mask=mask)[0]  # [batch_size, seq_len, dim]
        if self.partition:
            # Rearrange the annotations to ensure that the transition to
            # fenceposts captures an even split between position and content.
            # TODO(nikita): try alternatives, such as omitting position entirely
            hidden_states = torch.cat([
                hidden_states[:, :, 0::2],
                hidden_states[:, :, 1::2],
            ], 2)
        hidden_states = torch.cat([
                hidden_states[:, :-1, :self.d_model//2], hidden_states[:, 1:, self.d_model//2:]
            ], dim=2)
        return hidden_states
