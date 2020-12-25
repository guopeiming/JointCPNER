# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from typing import List, Set, Tuple, Dict, Union
from queue import Queue
from utils import evaluate
from utils.trees import InternalParseNode, InternalTreebankNode, LeafParseNode
from utils.ner_evaluate import NERFScore


class JointFScore(object):
    def __init__(
            self, parsing_f: float, parsing_p: float, parsing_r: float, complete_match: float,
            pos_f: float, pos_p: float, pos_r: float,
            seg_f: float, seg_p: float, seg_r: float,
    ):
        super(JointFScore, self).__init__()

        self.parsing_f: float = parsing_f
        self.parsing_p: float = parsing_p
        self.parsing_r: float = parsing_r
        self.complete_match: float = complete_match

        self.pos_f: float = pos_f
        self.pos_p: float = pos_p
        self.pos_r: float = pos_r

        self.seg_f: float = seg_f
        self.seg_p: float = seg_p
        self.seg_r: float = seg_r

    def __str__(self):
        return '''(parsing_P=%.02f, parsing_R=%.02f, parsing_F1=%.02f, CompleteMatch=%.02f) \
        (pos_P=%.02f, pos_R=%.02f, pos_F1=%.02f) \
        (seg_P=%.02f, seg_R=%.02f, seg_F1=%.02f)''' %\
            (
                self.parsing_p, self.parsing_r, self.parsing_f, self.complete_match,
                self.pos_p, self.pos_r, self.pos_f,
                self.seg_p, self.seg_r, self.seg_f
            )


def cal_preformence(evalb_path: str, trees_gold: List[InternalParseNode], trees_pred: List[InternalParseNode])\
        -> Tuple[JointFScore, Dict[str, List[Union[InternalTreebankNode, Set[Tuple[str, Tuple[int, int]]]]]]]:
    """joint cal perf.
    Args:
        evab_path:
        trees_gold:
        trees_pred:

    Returns:
        JointFscore, predicted parsing tree, predict ner spans.
    """
    assert len(trees_gold) == len(trees_pred)

    trees_gold_treebank, trees_pred_treebank = [], []

    true_positive_pos, false_positive_pos, false_negative_pos = 0, 0, 0
    true_positive_seg, false_positive_seg, false_negative_seg = 0, 0, 0
    for pred_tree, gold_tree in zip(trees_pred, trees_gold):
        pred_pos_seg_spans = generate_pos_seg_spans(pred_tree)
        gold_pos_seg_spans = generate_pos_seg_spans(gold_tree)

        # pos, seg metric
        gold_seg_spans = set()
        for span in gold_pos_seg_spans:
            gold_seg_spans.add(span[1])
        for span in pred_pos_seg_spans:
            if span in gold_pos_seg_spans:
                true_positive_pos += 1
                gold_pos_seg_spans.remove(span)
            else:
                false_positive_pos += 1

            if span[1] in gold_seg_spans:
                true_positive_seg += 1
                gold_seg_spans.remove(span[1])
            else:
                false_positive_seg += 1
        false_negative_pos += len(gold_pos_seg_spans)
        false_negative_seg += len(gold_seg_spans)

        trees_gold_treebank.append(gold_tree.convert())
        trees_pred_treebank.append(pred_tree.convert())

    pos_fscore = NERFScore(true_positive_pos, false_positive_pos, false_negative_pos)
    seg_fscore = NERFScore(true_positive_seg, false_positive_seg, false_negative_seg)

    parsing_fscore = evaluate.evalb(evalb_path, trees_gold_treebank, trees_pred_treebank)

    return (
        JointFScore(
            parsing_fscore.fscore, parsing_fscore.precision, parsing_fscore.recall, parsing_fscore.complete_match,
            pos_fscore.fscore, pos_fscore.precision, pos_fscore.recall,
            seg_fscore.fscore, seg_fscore.precision, seg_fscore.recall,
        ),
        {
            'pred_trees': trees_pred_treebank,
            'gold_trees': trees_gold_treebank,
        },
    )


def generate_pos_seg_spans(tree: InternalParseNode) -> Set[Tuple[str, Tuple[int, int]]]:
    pos_seg_spans = []
    snt_len = len(list(tree.leaves()))
    q = Queue()
    q.put(tree)
    while not q.empty():
        tree = q.get()

        # generate pos_seg_spans
        if tree.label[-1].startswith('POSTAG'):
            pos_tag = tree.label[-1].split('-')[1]
        else:
            pos_tag = 'NOPOS'

        flag = False
        for child in tree.children:
            assert isinstance(child, InternalParseNode) or isinstance(child, LeafParseNode)
            if isinstance(child, InternalParseNode):
                q.put(child)
                flag = False
            else:
                if flag:
                    pos_seg_spans[-1][1][1] = child.right
                else:
                    pos_seg_spans.append([pos_tag, [child.left, child.right]])
                flag = True

    pos_seg_spans = sorted(pos_seg_spans, key=lambda item: item[1][0])
    pos_seg_spans = tuple((item[0], (item[1][0], item[1][1])) for item in pos_seg_spans)
    assert all(item0[1][1] == item1[1][0] for item0, item1 in zip(pos_seg_spans, pos_seg_spans[1:]))
    assert pos_seg_spans[0][1][0] == 0 and pos_seg_spans[-1][1][1] == snt_len

    return set(pos_seg_spans)
