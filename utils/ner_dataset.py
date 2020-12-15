# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from torch.utils.data import Dataset
from typing import List, Dict, Union
from utils.trees import InternalParseNode


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


def pad_collate_fn(insts) -> Dict[str, List[Union[List[str], InternalParseNode]]]:
    """Pad the instance to the max seq length in batch
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
    for pos_tag, word in zip(pos_tags, snts):
        assert len(pos_tag) == len(word)
    return {'pos_tags': pos_tags, 'snts': snts, 'gold_trees': gold_trees}
