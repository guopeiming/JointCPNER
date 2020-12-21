# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import os
from torch.utils.data import Dataset, DataLoader
from utils.vocab import Vocab
from typing import List, Dict, Tuple
from config.Constants import LANGS_NEED_SEG, DATASET_LIST


class NERDataset(Dataset):
    def __init__(self, snts: List[str], golds: List[str]):
        super(NERDataset, self).__init__()
        self.snts = snts
        self.golds = golds

    def __len__(self):
        return len(self.snts)

    def __getitem__(self, idx):
        snt: List[str] = self.snts[idx].split(' ')
        gold: List[str] = self.golds[idx].split(' ')
        assert len(snt) == len(gold)
        return {'snt': snt, 'gold': gold}


def aggregate_collate_fn(insts: List) -> Dict[str, List[str]]:
    """aggragate the instance to the max seq length in batch.
    Args:
        insts: list of sample
    Returns:

    """
    snts, golds = [], []
    for inst in insts:
        snts.append(inst['snt'])
        golds.append(inst['gold'])
    assert len(snts) == len(golds)

    return {'snts': snts, 'golds': golds}


def batch_filter(insts: Dict[str, List[List[str]]], language: str, DATASET_MAX_SNT_LENGTH: int)\
        -> Tuple[Dict[str, List[List[str]]], int, int]:
    snts, golds = insts['snts'], insts['golds']
    res_snts, res_golds = [], []
    max_len = 0
    assert len(snts) == len(golds)
    for snt, gold in zip(snts, golds):
        if language in LANGS_NEED_SEG:
            snt_len = sum([len(word) for word in snt])
        else:
            snt_len = len(snt)
        if snt_len <= DATASET_MAX_SNT_LENGTH:
            res_snts.append(snt)
            res_golds.append(gold)
            if max_len < snt_len:
                max_len = snt_len
    if len(res_snts) == 0:
        res_snts, res_golds = snts[0], golds[0]
    return {'snts': res_snts, 'golds': res_golds}, len(res_snts), max_len


def batch_spliter(insts: Dict[str, List[List[str]]], max_len: int, BATCH_MAX_SNT_LENGTH: int)\
        -> List[Dict[str, List[List[str]]]]:
    sub_batch_times = (max_len // BATCH_MAX_SNT_LENGTH) + 1
    res = []
    for i in range(sub_batch_times):
        res.append({key: insts[key][i::sub_batch_times] for key in insts.keys()})
    return res


def load_data(path: str, batch_size: int, accum_steps: int, shuffle: bool, num_workers: int, drop_last: bool)\
        -> Tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    """load the datasets.
    Args:
        path: path of input data.
        batch_size: the number of insts in a batch.
        accum_steps: sccumulation steps.
        shuffle: whether to shuffle the dataset.
        num_workers: the number of process to load data.
        drop_last: whether to drop the last data.
    Returns:
        treebank and Dataloader of train, dev, test data. trainbank. batch_filter. batch_spliter. vocabs
    """
    print('data loading starts...', flush=True)
    res = tuple()
    vocab = None
    assert DATASET_LIST[0] == 'train'
    for item in DATASET_LIST:

        snts, golds = load_data_from_file(os.path.join(path, item+'.corpus'))

        assert len(snts) == len(golds)
        print('len(%s_data): %d' % (item, len(snts)))

        if item == DATASET_LIST[0]:
            vocab = vocabs_init(golds)

        data_loader = DataLoader(
                        NERDataset(snts, golds),
                        batch_size=batch_size if item == DATASET_LIST[0] else batch_size*accum_steps,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=aggregate_collate_fn,
                        drop_last=drop_last)
        res = res + (data_loader, )
    return res + (vocab, )


def load_data_from_file(path: str) -> Tuple[List[str], List[str]]:
    snts, golds = [], []
    with open(os.path.join(path), 'r', encoding='utf-8') as reader:
        snt, gold = [], []
        for line in reader:
            line: str = line.strip()
            if len(line) == 0:
                assert len(snt) > 0
                snts.append(' '.join(snt))
                golds.append(' '.join(gold))
                snt, gold = [], []
                continue
            word, ner = line.split('\t')
            snt.append(word)
            gold.append(ner)
    return snts, golds


def vocabs_init(train_data: List[str]) -> Vocab:
    print("Constructing vocabularies...", flush=True)

    vocab = Vocab(train_data)

    print('len(labels_vocab): %d' % len(vocab))

    return vocab


def write_ners(path_pred: str, path_gold: str, insts: Dict[str, List[List[str]]]):
    snts, tags_pred, tags_gold = insts['snts'], insts['pred_tags'], insts['gold_tags']
    assert len(snts) == len(tags_pred) == len(tags_gold)
    with open(path_pred, 'w', encoding='utf-8') as writer_pred, open(path_gold, 'w', encoding='utf-8') as writer_gold:
        for snt, tag_pred, tag_gold in zip(snts, tags_pred, tags_gold):
            assert len(snt) == len(tag_pred) == len(tag_gold)
            for word, ner_pred, ner_gold in zip(snt, tag_pred, tag_gold):
                writer_pred.write(word + '\t' + ner_pred + '\n')
                writer_gold.write(word + '\t' + ner_gold + '\n')
            writer_pred.write('\n')
            writer_gold.write('\n')
