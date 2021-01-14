# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from typing import List, Set, Tuple, Dict, Union
from queue import Queue
from utils import evaluate
from utils.trees import InternalParseNode, InternalTreebankNode, LeafParseNode
from utils.ner_evaluate import NERFScore
from config.Constants import LANGS_NEED_SEG, CHARACTER_BASED


class JointFScore(object):
    def __init__(
            self, parsing_f: float, parsing_p: float, parsing_r: float, complete_match: float,
            ner_f: float, ner_p: float, ner_r: float,
            pos_f: float, pos_p: float, pos_r: float,
            seg_f: float, seg_p: float, seg_r: float,
    ):
        super(JointFScore, self).__init__()

        self.parsing_f: float = parsing_f
        self.parsing_p: float = parsing_p
        self.parsing_r: float = parsing_r
        self.complete_match: float = complete_match

        self.ner_f: float = ner_f
        self.ner_p: float = ner_p
        self.ner_r: float = ner_r

        self.pos_f: float = pos_f
        self.pos_p: float = pos_p
        self.pos_r: float = pos_r

        self.seg_f: float = seg_f
        self.seg_p: float = seg_p
        self.seg_r: float = seg_r

    def __str__(self):
        return '''(parsing_P=%.02f, parsing_R=%.02f, parsing_F1=%.02f, CompleteMatch=%.02f) \
        (ner_P=%.02f, ner_R=%.02f, ner_F1=%.02f) \
        (pos_P=%.02f, pos_R=%.02f, pos_F1=%.02f) \
        (seg_P=%.02f, seg_R=%.02f, seg_F1=%.02f)''' %\
            (
                self.parsing_p, self.parsing_r, self.parsing_f, self.complete_match,
                self.ner_p, self.ner_r, self.ner_f,
                self.pos_p, self.pos_r, self.pos_f,
                self.seg_p, self.seg_r, self.seg_f
            )


def cal_performance(
    language: str, subword: str, evalb_path: str, trees_gold: List[InternalParseNode],
    trees_pred: List[InternalParseNode]
) -> Tuple[JointFScore, Dict[str, List[Union[InternalTreebankNode, Set[Tuple[str, Tuple[int, int]]]]]]]:
    if language in LANGS_NEED_SEG and subword == CHARACTER_BASED:
        return cal_performance_seg_pos_par(evalb_path, trees_gold, trees_pred)
    elif (language in LANGS_NEED_SEG and subword != CHARACTER_BASED) or (language not in LANGS_NEED_SEG):
        return cal_performance_pos_par(evalb_path, trees_gold, trees_pred)
    else:
        print(language, subword, 'cal_performance error')
        exit(-1)


def cal_performance_pos_par(
    evalb_path: str, trees_gold: List[InternalParseNode], trees_pred: List[InternalParseNode]
) -> Tuple[JointFScore, Dict[str, List[Union[InternalTreebankNode, Set[Tuple[str, Tuple[int, int]]]]]]]:

    assert len(trees_gold) == len(trees_pred)

    trees_gold_treebank, trees_pred_treebank = [], []
    res_pred_ner_spans, res_gold_ner_spans = [], []

    true_positive_ner, false_positive_ner, false_negative_ner = 0, 0, 0
    true_positive_pos, false_positive_pos, false_negative_pos = 0, 0, 0
    for pred_tree, gold_tree in zip(trees_pred, trees_gold):
        pred_ner_spans, pred_pos_tags = generate_ner_pos_spans(pred_tree)
        gold_ner_spans, gold_pos_tags = generate_ner_pos_spans(gold_tree)

        # NER metric
        temp_set = set()
        for span in pred_ner_spans:
            if span in gold_ner_spans:
                true_positive_ner += 1
                gold_ner_spans.remove(span)
                temp_set.add(span)
            else:
                false_positive_ner += 1
        false_negative_ner += len(gold_ner_spans)

        # NER return result
        res_pred_ner_spans.append(pred_ner_spans)
        res_gold_ner_spans.append(gold_ner_spans.union(temp_set))

        # pos metric
        assert len(pred_pos_tags) == len(gold_pos_tags)
        for pred_pos_tag, gold_pos_tag in zip(pred_pos_tags, gold_pos_tags):
            if pred_pos_tag == gold_pos_tag:
                true_positive_pos += 1
            else:
                false_positive_pos += 1
                false_negative_pos += 1

        trees_gold_treebank.append(gold_tree.convert())
        trees_pred_treebank.append(pred_tree.convert())

    ner_fscore = NERFScore(true_positive_ner, false_positive_ner, false_negative_ner)
    pos_fscore = NERFScore(true_positive_pos, false_positive_pos, false_negative_pos)
    seg_fscore = NERFScore(1, 0, 0)

    parsing_fscore = evaluate.evalb(evalb_path, trees_gold_treebank, trees_pred_treebank)

    return (
        JointFScore(
            parsing_fscore.fscore, parsing_fscore.precision, parsing_fscore.recall, parsing_fscore.complete_match,
            ner_fscore.fscore, ner_fscore.precision, ner_fscore.recall,
            pos_fscore.fscore, pos_fscore.precision, pos_fscore.recall,
            seg_fscore.fscore, seg_fscore.precision, seg_fscore.recall,
        ),
        {
            'pred_trees': trees_pred_treebank,
            'gold_trees': trees_gold_treebank,
            'pred_ners': res_pred_ner_spans,
            'gold_ners': res_gold_ner_spans
        },
    )


def generate_ner_pos_spans(tree: InternalParseNode) -> Tuple[Set[Tuple[str, Tuple[int, int]]], List[str]]:
    ner_spans, pos_tags = set(), []
    snt_len = len(list(tree.leaves()))
    q = Queue()
    q.put(tree)
    while not q.empty():
        tree = q.get()

        # generate ner spans
        for label in tree.label:
            if label.endswith('*'):
                ner_spans.add((label.split('-')[-1][:-1], (tree.left, tree.right)))

        # generate pos_seg_spans
        if tree.label[-1].startswith('POSTAG'):
            pos_tag = tree.label[-1].split('-')[1]
        else:
            pos_tag = 'NOPOS'

        for child in tree.children:
            assert isinstance(child, InternalParseNode) or isinstance(child, LeafParseNode)
            if isinstance(child, InternalParseNode):
                q.put(child)
            else:
                pos_tags.append((pos_tag, child.left))

    pos_tags = sorted(pos_tags, key=lambda item: item[1])
    assert all(item0[1]+1 == item1[1] for item0, item1 in zip(pos_tags, pos_tags[1:]))
    assert len(pos_tags) == snt_len
    pos_tags = [item[0] for item in pos_tags]

    return ner_spans, pos_tags


def cal_performance_seg_pos_par(
    evalb_path: str, trees_gold: List[InternalParseNode], trees_pred: List[InternalParseNode]
) -> Tuple[JointFScore, Dict[str, List[Union[InternalTreebankNode, Set[Tuple[str, Tuple[int, int]]]]]]]:
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

    true_positive_ner, false_positive_ner, false_negative_ner = 0, 0, 0
    true_positive_pos, false_positive_pos, false_negative_pos = 0, 0, 0
    true_positive_seg, false_positive_seg, false_negative_seg = 0, 0, 0
    for pred_tree, gold_tree in zip(trees_pred, trees_gold):
        pred_ner_spans, pred_pos_seg_spans = generate_ner_pos_seg_spans(pred_tree)
        gold_ner_spans, gold_pos_seg_spans = generate_ner_pos_seg_spans(gold_tree)

        # NER metric
        temp_set = set()
        for span in pred_ner_spans:
            if span in gold_ner_spans:
                true_positive_ner += 1
                gold_ner_spans.remove(span)
                temp_set.add(span)
            else:
                false_positive_ner += 1
        false_negative_ner += len(gold_ner_spans)

        # NER return result
        res_pred_ner_spans.append(pred_ner_spans)
        res_gold_ner_spans.append(gold_ner_spans.union(temp_set))

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

    ner_fscore = NERFScore(true_positive_ner, false_positive_ner, false_negative_ner)
    pos_fscore = NERFScore(true_positive_pos, false_positive_pos, false_negative_pos)
    seg_fscore = NERFScore(true_positive_seg, false_positive_seg, false_negative_seg)

    parsing_fscore = evaluate.evalb(evalb_path, trees_gold_treebank, trees_pred_treebank)

    return (
        JointFScore(
            parsing_fscore.fscore, parsing_fscore.precision, parsing_fscore.recall, parsing_fscore.complete_match,
            ner_fscore.fscore, ner_fscore.precision, ner_fscore.recall,
            pos_fscore.fscore, pos_fscore.precision, pos_fscore.recall,
            seg_fscore.fscore, seg_fscore.precision, seg_fscore.recall,
        ),
        {
            'pred_trees': trees_pred_treebank,
            'gold_trees': trees_gold_treebank,
            'pred_ners': res_pred_ner_spans,
            'gold_ners': res_gold_ner_spans
        },
    )


def generate_ner_pos_seg_spans(tree: InternalParseNode)\
        -> Tuple[Set[Tuple[str, Tuple[int, int]]], Set[Tuple[str, Tuple[int, int]]]]:
    ner_spans, pos_seg_spans = set(), []
    snt_len = len(list(tree.leaves()))
    q = Queue()
    q.put(tree)
    while not q.empty():
        tree = q.get()

        # generate ner spans
        for label in tree.label:
            if label.endswith('*'):
                ner_spans.add((label.split('-')[-1][:-1], (tree.left, tree.right)))

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

    return ner_spans, set(pos_seg_spans)
