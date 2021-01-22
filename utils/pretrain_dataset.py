# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Tuple
import collections


class Tree(object):
    """Tree data structure.

    Attributes:
        label:
        word:
        is_leaf:
        left and right: span idx in sentence, left included, right excluded. -> [left, right)
    """

    def __init__(self, label: str, children_or_word: Union[List['Tree'], str], span_start_idx: int, span_end_idx: int):
        super(Tree, self).__init__()

        self.label: str = label
        self.word = None
        self.children = None
        self.is_leaf = isinstance(children_or_word, str)

        if self.is_leaf:
            self.word: str = children_or_word
        else:
            assert isinstance(children_or_word, collections.abc.Sequence)
            assert all(isinstance(child, Tree) for child in children_or_word)
            self.children: List[Tree] = children_or_word

        self.left = span_start_idx
        self.right = span_end_idx
        assert self.left < self.right
        if not self.is_leaf:
            assert all(left.right == right.left for left, right in zip(self.children, self.children[1:]))
            assert self.left == self.children[0].left and self.right == self.children[-1].right

    def linearize(self):
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])
        return '(%s %s)' % (self.label, text)

    def leaves(self):
        if self.is_leaf:
            yield (int(self.label), self.word)
        else:
            for child in self.children:
                yield from child.leaves()

    def __str__(self) -> str:
        return self.linearize()

    def generate_data_list(self, data_list: List, head_list: List[int]):
        height = 0
        if self.is_leaf:
            self.head = self.left
            self.height = height
            return 0
        else:
            for child in self.children:
                height_temp = child.generate_data_list(data_list, head_list)
                if height < height_temp+1:
                    height = height_temp + 1
                    self.height = height

        if height == 1:
            self.head = self.left
        else:
            # head
            if not self.label.startswith('NamedEntity'):
                num = 0
                for i, head_idx in enumerate(head_list[self.left: self.right]):
                    if not (self.left <= head_idx < self.right):
                        num += 1
                        head = i+self.left
                assert num == 1
            else:
                head = self.right-1
            self.head = head

        if height >= 2:
            # subtree & child span
            subtree_span = []
            children_span = []
            subtree_str = ''
            subtree_leaf_str = ''
            subtree_str += self.label + '->/'
            subtree_leaf_str += self.label + '->/'
            for child in self.children:
                assert child.left <= child.head < child.right
                children_span.append([child.left, child.right, child.head])
                if child.is_leaf:
                    subtree_str += child.word + '/'
                    subtree_leaf_str += 'LEAF' + '/'
                else:
                    subtree_str += child.label + '/'
                    subtree_leaf_str += child.label + '/'
            subtree_span.append([self.left, self.right, subtree_str])
            if subtree_str != subtree_leaf_str:
                subtree_span.append([self.left, self.right, subtree_leaf_str])
            assert len(children_span) >= 1

            if len(data_list) > 1 and self.children[-1].height > 1:
                assert data_list[-1][0][0][:-1] == children_span[-1][:-1]
            data_list.append((subtree_span, children_span))
        return height


def generate_tree_from_str(text: str) -> Tree:
    assert text.count('(') == text.count(')')
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    idx = 0
    return build_tree(tokens, idx, 0)[0]


def build_tree(tokens: List[str], idx: int, span_startpoint_idx: int):
    """generate a tree from tokens list.
    and the tree to be generated is bracketed.
    Args:
        tokens[idx] must be '('
        span_startpoint_idx: the start point idx in the sentence of the span

    Returns:
        tree and idx to be processed.
    """
    idx += 1
    label = tokens[idx]
    idx += 1
    assert idx < len(tokens)

    if tokens[idx] == '(':
        children = []
        span_endpoint_idx = span_startpoint_idx
        while idx < len(tokens) and tokens[idx] == '(':
            child, idx, span_endpoint_idx = build_tree(tokens, idx, span_endpoint_idx)
            children.append(child)
        # generate internal node
        tree = Tree(label, children, span_startpoint_idx, span_endpoint_idx)
        assert not tree.is_leaf
    elif tokens[idx] == ')':
        print('No word!!!')
        exit(-1)
    else:
        word = tokens[idx]
        idx += 1
        # generate leaf node
        span_endpoint_idx = span_startpoint_idx + 1
        tree = Tree(label, word, span_startpoint_idx, span_endpoint_idx)
        assert tree.is_leaf

    assert tokens[idx] == ')'
    return tree, idx+1, span_endpoint_idx


class PretrainData(object):

    def __init__(self, content: str) -> None:
        super(PretrainData, self).__init__()
        self.tree = content
        self.snt: List[str] = None
        self.head: List[int] = None
        # [(subtree, children_span)] -> self.data_list
        # subtree_span -> List[[int, int, str], ] (left, right, subtree)
        # children_span -> List[[int, int, int],] (left, right, head)
        self.data_list: List[Tuple[List[List[int, int, str]], List[List[int, int, int]]]] = []

    def get_input_data(self) -> Tuple[List[List[Union[int, str]]], List[List[int]], List[str]]:
        if not self.snt:
            tree = generate_tree_from_str(self.tree)
            leaves = list(tree.leaves())
            self.snt = [item[1] for item in leaves]
            self.head = [item[0] for item in leaves]
            tree.generate_data_list(self.data_list, self.head)
            random.shuffle(self.data_list)
        rand_num = np.random.randint(0, len(self.data_list))

        subtree_span, children_span = self.data_list[rand_num]

        return [item.copy() for item in subtree_span], [item.copy() for item in children_span], self.snt.copy()


class Vocab(object):

    def __init__(self, path: str) -> None:
        super(Vocab, self).__init__()
        self.id2token = ['<UNK>']
        self.token2id = {'<UNK>': 0}
        self.unkid = 0
        self.unklabel = '<UNK>'
        self.vocab_init(path)

    def __len__(self):
        return len(self.id2token)

    def vocab_init(self, path: str):
        with open(path, 'r', encoding='utf-8') as reader:
            for line in reader:
                word = line.strip().split('\t')[0]
                self.token2id[word] = len(self.id2token)
                self.id2token.append(word)

    def encode(self, tokens: Union[str, List[str]]):
        if isinstance(tokens, list):
            return [self.encode(token) for token in tokens]
        else:
            return self.token2id.get(tokens, self.unkid)

    def decode(self, ids: Union[int, List[int]]):
        if isinstance(ids, list):
            return [self.decode(idx) for idx in ids]
        else:
            return self.id2token[ids]


class PretrainDataset(Dataset):
    def __init__(self, data: List[PretrainData], subtree_vocab: Vocab, head_vocab: Vocab, token_vocab: Vocab):
        super(PretrainDataset, self).__init__()
        self.dataset = data
        self.subtree_vocab = subtree_vocab
        self.head_vocab = head_vocab
        self.token_vocab = token_vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        subtree_span, children_span, snt = self.dataset[idx].get_input_data()

        # children_span
        for span in children_span:
            span[2] = self.head_vocab.encode(snt[span[2]])

        # subtree_span
        for span in subtree_span:
            span[2] = self.subtree_vocab.encode(span[2])

        return {'subtree_span': subtree_span, 'children_span': children_span, 'snt': snt}


def aggregate_collate_fn(insts) -> Dict[str, List[List[Union[str, List[int]]]]]:
    """aggregate the instance to the max seq length in batch
    Args:
        insts: list of sample
    Returns:

    """
    subtree_spans, children_spans, snts = [], [], []
    for inst in insts:
        subtree_spans.append(inst['subtree_span'])
        snts.append(inst['snt'])
        children_spans.append(inst['children_span'])
    assert len(subtree_spans) == len(snts) == len(children_spans)
    return {'subtree_spans': subtree_spans, 'snts': snts, 'children_spans': children_spans}


def batch_filter(
    insts: Dict[str, List[List[Union[str, List[int]]]]], DATASET_MAX_SNT_LENGTH: int
) -> Tuple[Dict[str, List[List[Union[str, List[int]]]]], int, int]:
    subtree_spans, snts, children_spans = insts['subtree_spans'], insts['snts'], insts['children_spans']
    res_subtree_spans, res_snts, res_children_spans = [], [], []
    max_len = 0
    assert len(subtree_spans) == len(snts) == len(children_spans)
    for subtree_span, snt, children_span in zip(subtree_spans, snts, children_spans):
        snt_len = len(snt)
        if snt_len <= DATASET_MAX_SNT_LENGTH:
            res_subtree_spans.append(subtree_span)
            res_snts.append(snt)
            res_children_spans.append(children_span)
            if max_len < snt_len:
                max_len = snt_len
    if len(res_snts) == 0:
        res_subtree_spans, res_snts, res_children_spans = subtree_spans[0], snts[0], children_spans[0]
    return (
        {'subtree_spans': res_subtree_spans, 'snts': res_snts, 'children_spans': res_children_spans},
        len(res_snts), max_len
    )


def batch_spliter(
    insts: Dict[str, List[Union[str, List[List[int]]]]], max_len: int, BATCH_MAX_SNT_LENGTH: int
) -> List[Dict[str, List[List[Union[str, List[int]]]]]]:
    sub_batch_times = (max_len // BATCH_MAX_SNT_LENGTH) + 1
    res = []
    for i in range(sub_batch_times):
        res.append({key: insts[key][i::sub_batch_times] for key in insts.keys()})
    return res


def load_dataset(path: str):
    data_list = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data_list.append(PretrainData(line.strip()))
    return data_list


def load_data(path: str, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool)\
        -> Tuple[DataLoader, DataLoader, Vocab, Vocab, Vocab]:
    """load the datasets.
    Args:
        path: path of input data.
        batch_size: the number of insts in a batch.
        accum_steps: accumulation steps.
        shuffle: whether to shuffle the dataset.
        num_workers: the number of process to load data.
        drop_last: whether to drop the last data.
    Returns:
    """
    print('data loading starts...', flush=True)
    train_data_list = load_dataset(os.path.join(path, 'train.corpus'))
    print('len(train_data): %d' % len(train_data_list), flush=True)
    dev_data_list = load_dataset(os.path.join(path, 'dev.corpus'))
    print('len(dev_data): %d' % len(dev_data_list), flush=True)

    subtree_vocab = Vocab(os.path.join(path, 'subtree.vocab'))
    head_vocab = Vocab(os.path.join(path, 'head.vocab'))
    token_vocab = Vocab(os.path.join(path, 'token.vocab'))
    print('len(subtree_vocab): %d' % len(subtree_vocab))
    print('len(head_vocab): %d' % len(head_vocab))
    print('len(toekn_vocab): %d' % len(token_vocab), flush=True)

    train_dataloader = DataLoader(
        PretrainDataset(train_data_list, subtree_vocab, head_vocab, token_vocab),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=aggregate_collate_fn,
        drop_last=drop_last
    )
    dev_dataloader = DataLoader(
        PretrainDataset(dev_data_list, subtree_vocab, head_vocab, token_vocab),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=aggregate_collate_fn,
        drop_last=drop_last
    )
    return train_dataloader, dev_dataloader, subtree_vocab, head_vocab, token_vocab
