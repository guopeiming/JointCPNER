# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from typing import Union, List


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
            self.children: List[Tree] = children_or_word

        self.left = span_start_idx
        self.right = span_end_idx
        assert self.left < self.right
        if not self.is_leaf:
            assert all(left.right == right.left for left, right in zip(self.children, self.children[1:]))
            assert self.left == self.children[0].left and self.right == self.children[-1].right

    def ner_match(self, start: int, end: int, change_laebl: bool = False, span_label: str = None) -> int:
        """identify type of matching between ner and constitent label.
        Args:
            start and end: span idx [start, end)

        Returns:
            0: not match
            1: match one label exactly
            2: match continuous label
        """
        assert self.left <= start < end <= self.right
        if change_laebl:
            assert span_label is not None

        # [start, end) match a span exactly
        if self.left == start and self.right == end:
            assert not self.is_leaf
            if change_laebl:
                self.change_label(span_label)
                self.label = self.label + '*'
            return 1

        # [start, end) in a subtree
        if not self.is_leaf:
            for child in self.children:
                if child.left <= start < end <= child.right:
                    return child.ner_match(start, end, change_laebl, span_label)

        # because self.left <= start < end <= self.right, if self is a leaf
        # node, it must match [start, end) and return False in the code above.
        assert not self.is_leaf
        # [start, end) match contiguous children.
        start_match_flag, end_match_flag = False, False
        start_subtree_idx, end_subtree_idx = 0, 0
        for subtree_idx, child in enumerate(self.children):
            if start == child.left:
                start_match_flag = True
                start_subtree_idx = subtree_idx
            if end == child.right:
                end_match_flag = True
                end_subtree_idx = subtree_idx
        if start_match_flag and end_match_flag:
            if change_laebl:

                subtrees_left = self.children[0:start_subtree_idx]
                subtrees_right = self.children[end_subtree_idx+1:]
                ner_subtrees_list: List[Tree] = self.children[start_subtree_idx: end_subtree_idx+1]

                for child in ner_subtrees_list:
                    child.change_label(span_label)

                assert start == ner_subtrees_list[0].left and end == ner_subtrees_list[-1].right
                ner_subtree = Tree('NamedEntity-'+span_label+'*', ner_subtrees_list, start, end)
                self.children = subtrees_left + [ner_subtree] + subtrees_right
            return 2
        else:
            return 0

    def change_label(self, span_label: str):
        if not self.is_leaf:
            assert '-' not in self.label
            self.label = self.label + '-' + span_label
            for child in self.children:
                child.change_label(span_label)

    def linearize(self):
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])
        return '(%s %s)' % (self.label, text)

    def leaves(self):
        if self.is_leaf:
            yield self.word
        else:
            for child in self.children:
                yield from child.leaves()

    def __str__(self) -> str:
        return self.linearize()


def load_trees(path: str) -> List[Tree]:
    trees = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()

            tree = generate_tree_from_str(line)
            trees.append(tree)
    return trees


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
