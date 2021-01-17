# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import os
from utils.ner_dataset import write_ners
from torch.utils.data import Dataset, DataLoader
from utils.vocabulary import Vocabulary
from typing import List, Dict, Union, Tuple, Set
from utils.trees import InternalParseNode, InternalTreebankNode, load_trees, write_trees
from config.Constants import LANGS_NEED_SEG, DATASET_LIST, NER_LABELS, START, STOP, TAG_UNK, EMPTY_LABEL


class JointDataset(Dataset):
    def __init__(self, joint_data: List[InternalParseNode], parsing_data: List[InternalParseNode]):
        super(JointDataset, self).__init__()
        self.joint_trees = joint_data
        self.parsing_trees = parsing_data
        assert len(self.joint_trees) == len(self.parsing_trees)

    def __len__(self):
        return len(self.joint_trees)

    def __getitem__(self, idx):
        leaves = list(self.joint_trees[idx].leaves())
        assert len(leaves) == len(list(self.parsing_trees[idx].leaves()))
        pos_tags = [leaf.tag for leaf in leaves]
        words = [leaf.word for leaf in leaves]
        return {
            'pos_tags': pos_tags, 'words': words, 'joint_gold_tree': self.joint_trees[idx],
            'parsing_gold_tree': self.parsing_trees[idx]
        }


def aggregate_collate_fn(insts) -> Dict[str, List[Union[List[str], InternalParseNode]]]:
    """aggregate the instance to the max seq length in batch
    Args:
        insts: list of sample
    Returns:

    """
    pos_tags, snts, joint_gold_trees, parsing_gold_trees = [], [], [], []
    for inst in insts:
        pos_tags.append(inst['pos_tags'])
        snts.append(inst['words'])
        joint_gold_trees.append(inst['joint_gold_tree'])
        parsing_gold_trees.append(inst['parsing_gold_tree'])
    assert len(pos_tags) == len(snts) == len(joint_gold_trees) == len(parsing_gold_trees)
    for pos_tag, snt in zip(pos_tags, snts):
        assert len(pos_tag) == len(snt)
    return {
        'pos_tags': pos_tags, 'snts': snts, 'joint_gold_trees': joint_gold_trees,
        'parsing_gold_trees': parsing_gold_trees
    }


def batch_filter(insts: Dict[str, List[Union[List[str], InternalParseNode]]], language: str,
                 DATASET_MAX_SNT_LENGTH: int) -> Tuple[Dict[str, List[Union[List[str], InternalParseNode]]], int, int]:
    pos_tags, snts, joint_gold_trees, parsing_gold_trees = \
        insts['pos_tags'], insts['snts'], insts['joint_gold_trees'], insts['parsing_gold_trees']
    res_pos_tags, res_snts, res_joint_gold_trees, res_parsing_gold_trees = [], [], [], []
    max_len = 0
    assert len(pos_tags) == len(snts) == len(joint_gold_trees) == len(parsing_gold_trees)
    for pos_tag, snt, joint_gold_tree, parsing_gold_tree in zip(pos_tags, snts, joint_gold_trees, parsing_gold_trees):
        if language in LANGS_NEED_SEG:
            snt_len = sum([len(word) for word in snt])
        else:
            snt_len = len(snt)
        if snt_len <= DATASET_MAX_SNT_LENGTH:
            res_pos_tags.append(pos_tag)
            res_snts.append(snt)
            res_joint_gold_trees.append(joint_gold_tree)
            res_parsing_gold_trees.append(parsing_gold_tree)
            if max_len < snt_len:
                max_len = snt_len
    if len(res_snts) == 0:
        res_pos_tags, res_snts, res_joint_gold_trees, res_parsing_gold_trees =\
            pos_tags[0], snts[0], joint_gold_trees[0], parsing_gold_trees[0]
    return (
        {
            'pos_tags': res_pos_tags,
            'snts': res_snts,
            'joint_gold_trees': res_joint_gold_trees,
            'parsing_gold_trees': res_parsing_gold_trees
        },
        len(res_snts),
        max_len
    )


def batch_spliter(insts: Dict[str, List[Union[List[str], InternalParseNode]]], max_len: int, BATCH_MAX_SNT_LENGTH: int)\
        -> List[Dict[str, List[Union[List[str], InternalParseNode]]]]:
    sub_batch_times = (max_len // BATCH_MAX_SNT_LENGTH) + 1
    res = []
    for i in range(sub_batch_times):
        res.append({key: insts[key][i::sub_batch_times] for key in insts.keys()})
    return res


def load_data(path: str, parsing_path: str, batch_size: int, accum_steps: int, shuffle: bool, num_workers: int,
              drop_last: bool) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Vocabulary]]:
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
    joint_vocabs, parsing_vocabs = None, None
    assert DATASET_LIST[0] == 'train'
    for item in DATASET_LIST:
        joint_treebank = load_trees(os.path.join(path, item+'.corpus'), strip_top=True)
        parsing_treebank = load_trees(os.path.join(parsing_path, item+'.corpus'), strip_top=True)
        joint_parse_trees = [tree.convert() for tree in joint_treebank]
        parsing_parse_trees = [tree.convert() for tree in parsing_treebank]
        print('len(%s_data): %d' % (item, len(joint_parse_trees)))

        if item == DATASET_LIST[0]:
            joint_vocabs = vocabs_init(joint_parse_trees)
            parsing_vocabs = vocabs_init(parsing_parse_trees)

        data_loader = DataLoader(
                        JointDataset(joint_parse_trees, parsing_parse_trees),
                        batch_size=batch_size if item == DATASET_LIST[0] else batch_size*2,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=aggregate_collate_fn,
                        drop_last=drop_last)
        res = res + (data_loader, )
    return res + (joint_vocabs, parsing_vocabs)


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

    snts, ner_spans_pred, ner_spans_gold = [], [], []
    for tree, ner_span_pred, ner_span_gold in \
            zip(data_dict['gold_trees'], data_dict['pred_ners'], data_dict['gold_ners']):

        leaves = list(tree.leaves())
        snt = [leaf.word for leaf in leaves]
        snts.append(snt)

        ner_spans_pred.append(ner_span_pred)
        ner_spans_gold.append(ner_span_gold)

    write_ners(
        os.path.join(path, type_+'.pred.best.ners'),
        os.path.join(path, type_+'.gold.ners'),
        {'snts': snts, 'pred_tags': ner_spans_pred, 'gold_tags': ner_spans_gold},
        span_based=True
    )


def generate_cross_labels_idx(vocab: Vocabulary) -> Dict[str, Tuple[int]]:
    cross_labels_idx: Dict[str, Tuple[int]] = dict()
    for labels in vocab.indices:
        for label in labels:
            label_list: List[str] = label.split('-')
            constitent, ner = '-'.join(label_list[:-1]), label_list[-1]
            if ner.endswith('*'):
                ner = ner[:-1]
            if ner in NER_LABELS:
                if ner not in cross_labels_idx.get(ner, tuple()):
                    cross_labels_idx[ner] = cross_labels_idx.get(ner, tuple()) + (vocab.indices[labels], )
                if constitent not in cross_labels_idx.get(constitent, tuple()):
                    cross_labels_idx[constitent] = cross_labels_idx.get(constitent, tuple()) + (vocab.indices[labels], )
    print('len(cross_labels_idx_dict):', len(cross_labels_idx))
    return cross_labels_idx
