# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from typing import List, Set, Tuple, Dict, Union
from queue import Queue
from utils import evaluate
from utils.trees import InternalParseNode, InternalTreebankNode
from utils.ner_evaluate import NERFScore


class JointFScore(object):
    def __init__(
            self, parsing_f: float, parsing_p: float, parsing_r: float, complete_match: float, ner_f: float,
            ner_p: float, ner_r: float):
        super(JointFScore, self).__init__()

        self.parsing_f: float = parsing_f
        self.parsing_p: float = parsing_p
        self.parsing_r: float = parsing_r
        self.complete_match: float = complete_match
        self.ner_f: float = ner_f
        self.ner_p: float = ner_p
        self.ner_r: float = ner_r

    def __str__(self):
        return '(parsing_P=%.02f, parsing_R=%.02f, parsing_F1=%.02f, CompleteMatch=%.02f) \
            (ner_P=%.02f, ner_R=%.02f, ner_F1=%.02f)' %\
            (self.parsing_p, self.parsing_r, self.parsing_f, self.complete_match,
             self.ner_p, self.ner_r, self.ner_f)


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
    res_pred_ner_spans, res_gold_ner_spans = [], []

    true_positive, false_positive, false_negative = 0, 0, 0
    for pred_tree, gold_tree in zip(trees_pred, trees_gold):
        pred_ner_spans = generate_ner_spans(pred_tree)
        gold_ner_spans = generate_ner_spans(gold_tree)
        res_pred_ner_spans.append(pred_ner_spans)
        res_gold_ner_spans.append(gold_ner_spans)

        for span in pred_ner_spans:
            if span in gold_ner_spans:
                true_positive += 1
                gold_ner_spans.remove(span)
            else:
                false_positive += 1
        false_negative += len(gold_ner_spans)

        trees_gold_treebank.append(gold_tree.convert())
        trees_pred_treebank.append(pred_tree.convert())

    ner_fscore = NERFScore(true_positive, false_positive, false_negative)

    parsing_fscore = evaluate.evalb(evalb_path, trees_gold_treebank, trees_pred_treebank)

    return (
        JointFScore(
            parsing_fscore.fscore, parsing_fscore.precision, parsing_fscore.recall, parsing_fscore.complete_match,
            ner_fscore.fscore, ner_fscore.precision, ner_fscore.recall
        ),
        {
            'pred_trees': trees_pred_treebank,
            'gold_trees': trees_gold_treebank,
            'pred_ners': res_pred_ner_spans,
            'gold_ners': res_gold_ner_spans
        },
    )


def generate_ner_spans(tree: InternalParseNode) -> Set[Tuple[str, Tuple[int, int]]]:
    spans = set()
    q = Queue()
    q.put(tree)
    while not q.empty():
        tree = q.get()
        for label in tree.label:
            if label.endswith('*'):
                spans.add((label.split('-')[1][:-1], (tree.left, tree.right)))
        for child in tree.children:
            if isinstance(child, InternalParseNode):
                q.put(child)
    return spans
