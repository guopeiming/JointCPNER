# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import os
from utils.ner_dataset import write_ners
from torch.utils.data import Dataset, DataLoader
from utils.vocabulary import Vocabulary
from typing import List, Dict, Union, Tuple, Set
from utils.trees import InternalParseNode, InternalTreebankNode, load_trees, write_trees
from config.Constants import LANGS_NEED_SEG, DATASET_LIST, START, STOP, TAG_UNK, EMPTY_LABEL


class JointDataset(Dataset):
    def __init__(self, data: List[InternalParseNode]):
        super(JointDataset, self).__init__()
        self.parse_trees = data

    def __len__(self):
        return len(self.parse_trees)

    def __getitem__(self, idx):
        leaves = list(self.parse_trees[idx].leaves())
        pos_tags = [leaf.tag for leaf in leaves]
        words = [leaf.word for leaf in leaves]
        return {'pos_tags': pos_tags, 'words': words, 'gold_tree': self.parse_trees[idx]}


def aggregate_collate_fn(insts) -> Dict[str, List[Union[List[str], InternalParseNode]]]:
    """aggregate the instance to the max seq length in batch
    Args:
        insts: list of sample
    Returns:

    """
    pos_tags, snts, gold_trees = [], [], []
    for inst in insts:
        pos_tags.append(inst['pos_tags'])
        snts.append(inst['words'])
        gold_trees.append(inst['gold_tree'])
    assert len(pos_tags) == len(snts) == len(gold_trees)
    for pos_tag, snt in zip(pos_tags, snts):
        assert len(pos_tag) == len(snt)
    return {'pos_tags': pos_tags, 'snts': snts, 'gold_trees': gold_trees}


def batch_filter(insts: Dict[str, List[Union[List[str], InternalParseNode]]], language: str,
                 DATASET_MAX_SNT_LENGTH: int) -> Tuple[Dict[str, List[Union[List[str], InternalParseNode]]], int, int]:
    pos_tags, snts, gold_trees = insts['pos_tags'], insts['snts'], insts['gold_trees']
    res_pos_tags, res_snts, res_gold_trees = [], [], []
    max_len = 0
    assert len(pos_tags) == len(snts) == len(gold_trees)
    for pos_tag, snt, gold_tree in zip(pos_tags, snts, gold_trees):
        if language in LANGS_NEED_SEG:
            snt_len = sum([len(word) for word in snt])
        else:
            snt_len = len(snt)
        if snt_len <= DATASET_MAX_SNT_LENGTH:
            res_pos_tags.append(pos_tag)
            res_snts.append(snt)
            res_gold_trees.append(gold_tree)
            if max_len < snt_len:
                max_len = snt_len
    if len(res_snts) == 0:
        res_pos_tags, res_snts, res_gold_trees = pos_tags[0], snts[0], gold_trees[0]
    return {'pos_tags': res_pos_tags, 'snts': res_snts, 'gold_trees': res_gold_trees}, len(res_snts), max_len


def batch_spliter(insts: Dict[str, List[Union[List[str], InternalParseNode]]], max_len: int, BATCH_MAX_SNT_LENGTH: int)\
        -> List[Dict[str, List[Union[List[str], InternalParseNode]]]]:
    sub_batch_times = (max_len // BATCH_MAX_SNT_LENGTH) + 1
    res = []
    for i in range(sub_batch_times):
        res.append({key: insts[key][i::sub_batch_times] for key in insts.keys()})
    return res


def load_data(path: str, batch_size: int, accum_steps: int, shuffle: bool, num_workers: int, drop_last: bool)\
        -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Vocabulary]]:
    """load the datasets.
    Args:
        path: path of input data.
        batch_size: the number of insts in a batch.
        accum_steps: accumulation steps.
        shuffle: whether to shuffle the dataset.
        num_workers: the number of process to load data.
        drop_last: whether to drop the last data.
    Returns:
        treebank and Dataloader of train, dev, test data. batch_filter. batch_spliter. vocabs
    """
    print('data loading starts...', flush=True)
    res = tuple()
    vocabs = None
    assert DATASET_LIST[0] == 'train'
    for item in DATASET_LIST:
        treebank = load_trees(os.path.join(path, item+'.corpus'), strip_top=True)
        parse_trees = [tree.convert() for tree in treebank]
        print('len(%s_data): %d' % (item, len(parse_trees)))

        if item == DATASET_LIST[0]:
            vocabs = vocabs_init(parse_trees)

        data_loader = DataLoader(
                        JointDataset(parse_trees),
                        batch_size=batch_size if item == DATASET_LIST[0] else batch_size*accum_steps,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=aggregate_collate_fn,
                        drop_last=drop_last)
        res = res + (data_loader, )
    return res + (vocabs, )


def vocabs_init(train_data: List[InternalParseNode]) -> Dict[str, Vocabulary]:
    print("Constructing vocabularies...", flush=True)

    pos_tags_vocab = Vocabulary()
    pos_tags_vocab.index(START)
    pos_tags_vocab.index(STOP)
    pos_tags_vocab.index(TAG_UNK)

    # words_vocab = Vocabulary()
    # words_vocab.index(START)
    # words_vocab.index(STOP)
    # words_vocab.index(UNK)

    labels_vocab = Vocabulary()
    labels_vocab.index(EMPTY_LABEL)

    for tree in train_data:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, InternalParseNode):
                labels_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                pos_tags_vocab.index(node.tag)
                # words_vocab.index(node.word)

    pos_tags_vocab.freeze()
    # words_vocab.freeze()
    labels_vocab.freeze()

    print('len(pos_tags_vocab): %d\nlen(labels_vocab): %d' % (pos_tags_vocab.size, labels_vocab.size))

    return {'pos_tags': pos_tags_vocab, 'labels': labels_vocab}


def write_joint_data(
        path: str,
        data_dict: Dict[str, List[Union[InternalTreebankNode, Set[Tuple[str, Tuple[int, int]]]]]],
        type_: str
):
    assert len(data_dict['pred_trees']) == len(data_dict['gold_trees'])
    assert len(data_dict['pred_ners']) == len(data_dict['gold_ners'])
    assert len(data_dict['pred_ners']) == len(data_dict['pred_trees'])
    write_trees(os.path.join(path, type_+'.pred.best.trees'), data_dict['pred_trees'],
                os.path.join(path, type_+'.gold.trees'), data_dict['gold_trees'])

    snts, ner_tags_pred, ner_tags_gold = [], [], []
    for tree, ner_span_pred, ner_span_gold in \
            zip(data_dict['gold_trees'], data_dict['pred_ners'], data_dict['gold_ners']):

        leaves = list(tree.leaves())
        snt = [leaf.word for leaf in leaves]
        snts.append(snt)

        ner_tag_pred = ['O' for _ in range(len(snt))]
        for span in ner_span_pred:
            span_label = span[0]
            start, end = span[1]
            ner_tag_pred[start] = 'B-' + span_label
            for idx in range(start+1, end):
                ner_tag_pred[idx] = 'I-' + span_label
        ner_tags_pred.append(ner_tag_pred)

        ner_tag_gold = ['O' for _ in range(len(snt))]
        for span in ner_span_gold:
            span_label = span[0]
            start, end = span[1]
            ner_tag_gold[start] = 'B-' + span_label
            for idx in range(start+1, end):
                ner_tag_gold[idx] = 'I-' + span_label
        ner_tags_gold.append(ner_tag_gold)

    write_ners(
        os.path.join(path, type_+'.pred.best.ners'),
        os.path.join(path, type_+'.gold.ners'),
        {'snts': snts, 'pred_tags': ner_tags_pred, 'gold_tags': ner_tags_gold})
