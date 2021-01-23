import os
import json
import time
from spacy import Language
from typing import Counter, List, Set, Union
import collections
from multiprocessing import Process


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

    def ner_match(self, start: int, end: int, up: bool, change_label: bool = False, span_label: str = None) -> int:
        """identify type of matching between ner and constitent label.
        Args:
            start and end: span idx [start, end)

        Returns:
            0: not match
            1: match one label exactly
            2: match continuous label
        """
        assert self.left <= start < end <= self.right
        if change_label:
            assert span_label is not None

        if up:
            # [start, end) match a span exactly
            if self.left == start and self.right == end:
                assert not self.is_leaf
                if change_label:
                    self.change_label(span_label)
                    self.label = self.label + '*'
                return 1

            # [start, end) in a subtree
            for child in self.children:
                if child.left <= start < end <= child.right and (not child.is_leaf):
                    return child.ner_match(start, end, up, change_label, span_label)
        else:
            # [start, end) in a subtree
            for child in self.children:
                if child.left <= start < end <= child.right and (not child.is_leaf):
                    return child.ner_match(start, end, up, change_label, span_label)

            # [start, end) match a span exactly
            if self.left == start and self.right == end:
                assert not self.is_leaf
                if change_label:
                    self.change_label(span_label)
                    self.label = self.label + '*'
                return 1

        # because self.left <= start < end <= self.right, if self is a leaf
        # node, it must match [start, end) and return 1 in the code above.
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
            if change_label:

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
            assert '-' not in self.label\
                or ('POSTAG' in self.label and self.label.count('-') == 1) \
                or (self.label in (
                    'POSTAG--LRB-', 'POSTAG--RRB-', 'POSTAG--LCB-', 'POSTAG--RCB-', 'POSTAG--LSB-', 'POSTAG--RSB-'))
            self.label = self.label + '-' + span_label
            for child in self.children:
                child.change_label(span_label)

    def linearize(self):
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])
        return '(%s %s)' % (self.label, text)

    def linearize_add_pos(self):
        if self.is_leaf:
            return '(POSTAG-%s (PAD_TAG %s))' % (self.label, self.word)
        else:
            text = ' '.join([child.linearize_add_pos() for child in self.children])
            return '(%s %s)' % (self.label, text)

    def linearize_convert_char(self, head_lists: List[List[int]]):
        if self.is_leaf:
            head_list = head_lists[self.left]
            assert len(head_list) == len(self.word)
            text = ''
            for i in range(len(self.word)):
                text += '(%s %s) ' % (str(head_list[i]), self.word[i])
            return text[:-1]
        else:
            text = ' '.join([child.linearize_convert_char(head_lists) for child in self.children])
        return '(%s %s)' % (self.label, text)

    def leaves(self):
        if self.is_leaf:
            yield (self.label, self.word)
        else:
            for child in self.children:
                yield from child.leaves()

    def __str__(self) -> str:
        return self.linearize()


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


ucb_parser = None


@Language.component("component")
def component_func(doc):
    doc[len(doc)-1].is_sent_start = False
    return ucb_parser(doc)


def preprocessing(language: str):
    from benepar.spacy_plugin import BeneparComponent
    import zh_core_web_trf
    import en_core_web_trf
    global ucb_parser
    if language == 'zh':
        nlp = zh_core_web_trf.load()
        ucb_parser = BeneparComponent('benepar_zh')
    elif language == 'en':
        nlp = en_core_web_trf.load()
        ucb_parser = BeneparComponent('benepar_en2')
    else:
        print('language error')
        exit(-1)

    nlp.disable_pipes('tagger', 'parser', 'attribute_ruler')
    nlp.add_pipe('component', name='cp_parser', last=True)
    return nlp


def is_chinese(check_str):
    if u'\u4e00' <= check_str <= u'\u9fff':
        return True
    else:
        return False


def sent_cleaning(sent: str, language: str, special_tokens: Set):
    res = ''
    for i in range(len(sent)):
        char = sent[i]
        if char == ' ':
            if (language == 'zh') and ((i == len(sent)-1) or i == 0 or (is_chinese(sent[i-1]) and is_chinese(sent[i+1])) or res[-1]==' '):
                char = ''
        if char in special_tokens:
            char = ('' if len(res) == 0 or res[-1] == ' ' else ' ') + char+' '
        res += char
    return res


def parsing(batch_sents: List[str], nlp):
    res_tree_string, res_entities = [], []

    docs = list(nlp.pipe(batch_sents))

    for doc in docs:
        sents = list(doc.sents)
        assert len(sents) == 1
        tree_string = sents[0]._.parse_string

        # add ner
        tree = generate_tree_from_str(tree_string)
        entities = []
        for ner in doc.ents:
            flag = tree.ner_match(ner.start, ner.end, False, False, ner.label_)
            if flag != 0:
                entities.append((ner.start, ner.end, ner.label_))

        res_tree_string.append(tree.linearize())
        res_entities.append(entities)

    return res_tree_string, res_entities


def raw_corpus_processing(corpus_file: str, language: str):
    batch_size = 32
    batch_sents = []
    nlp = preprocessing(language)

    with open(corpus_file, 'r', encoding='utf-8') as reader,\
         open(os.path.join(os.path.dirname(corpus_file), os.path.basename(corpus_file)+'.par.out'), 'w', encoding='utf-8') as par_writer,\
         open(os.path.join(os.path.dirname(corpus_file), os.path.basename(corpus_file)+'.ner.out'), 'w', encoding='utf-8') as ner_writer:

        special_tokens = set(['(', ')', '{', '}', '[', ']'])
        for i, line in enumerate(reader):
            line = line.strip()
            sent = sent_cleaning(line, language, special_tokens)
            batch_sents.append(sent)
            if len(batch_sents) == batch_size:
                tree_strings, entities = parsing(batch_sents, nlp)
                batch_sents = []
                for tree_string, entity in zip(tree_strings, entities):
                    par_writer.write(tree_string+'\n')
                    ner_writer.write(json.dumps(entity)+'\n')
            if i % 1000 == 0:
                print('pid %d: %d sentences are processed.' % (os.getpid(), i), flush=True)


def annotate_main(gpuid: int, file: str, language: str):
    print('subprocess %d starts' % os.getpid(), flush=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    # spacy.require_gpu()
    raw_corpus_processing(file, language)
    print('subprocess %d ends' % os.getpid())


def multi_process_annotate():
    file_start = 9
    p_list = []
    for i in range(4):
        file_name = os.path.join('./data/pretrain/', str(i+file_start)+'.corpus')
        p_list.append(Process(target=annotate_main, args=(7, file_name, 'zh')))
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

    print('all subprocess ends.')


def check_subtree_infor(subtree_counter: Counter, cp_tree: Tree, head_list: List[int], token_list: List[str]):
    height = 0
    if cp_tree.is_leaf:
        return 0
    else:
        for child in cp_tree.children:
            height_temp = check_subtree_infor(subtree_counter, child, head_list, token_list)
            if height < height_temp+1:
                height = height_temp+1

    if height >= 2:
        # head
        if not cp_tree.label.startswith('NamedEntity'):
            num = 0
            for head_idx in head_list[cp_tree.left: cp_tree.right]:
                if not (cp_tree.left <= head_idx < cp_tree.right):
                    num += 1
            assert num == 1

        # subtree
        subtree_str = ''
        subtree_leaf_str = ''
        subtree_str += cp_tree.label + '->/'
        subtree_leaf_str += cp_tree.label + '->/'
        for child in cp_tree.children:
            if child.is_leaf:
                subtree_str += child.word + '/'
                subtree_leaf_str += 'LEAF' + '/'
            else:
                subtree_str += child.label + '/'
                subtree_leaf_str += child.label + '/'
        subtree_counter[subtree_str] += 1
        if subtree_str != subtree_leaf_str:
            subtree_counter[subtree_leaf_str] += 1
    return height


def generate_pretrain_dataset(
    cp_file: str, dp_file: str, ner_file: str, data_file: str, subtree_file: str, token_file: str,
    fre: int
):

    special_tokens = {'(': '-LRB-', ')': '-RRB-', "{": "-LCB-", "}": "-RCB-", "[": "-LSB-", "]": "-RSB-"}
    subtree_counter = Counter()
    token_counter = Counter()
    with open(cp_file, 'r', encoding='utf-8') as cp_reader,\
         open(dp_file, 'r', encoding='utf-8') as dp_reader,\
         open(ner_file, 'r', encoding='utf-8') as ner_reader,\
         open(data_file, 'w', encoding='utf-8') as data_writer,\
         open(subtree_file, 'w', encoding='utf-8') as subtree_writer,\
         open(token_file, 'w', encoding='utf-8') as token_writer:
        for i, (cp_line, ner_line) in enumerate(zip(cp_reader, ner_reader)):
            cp_tree = generate_tree_from_str(cp_line.strip())
            ners = json.loads(ner_line.strip())
            cp_tree_str = cp_tree.linearize_add_pos()

            # reading head infor
            head_list = []
            token_list = []
            for dp_line in dp_reader:
                dp_line = dp_line.strip()
                if len(dp_line) == 0:
                    break
                dp_list = dp_line.split('\t')
                assert len(dp_list) == 4
                if dp_list[0] in special_tokens:
                    dp_list[0] = special_tokens[dp_list[0]]
                head_list.append(int(dp_list[2])-1)
                token_list.append(dp_list[0])
                token_counter[dp_list[0]] += 1

            # adding head infor to cp tree
            cp_tree_str_list = cp_tree_str.split('PAD_TAG')
            assert len(cp_tree_str_list) == len(head_list)+1 == len(token_list)+1

            cp_tree_str = cp_tree_str_list[0]
            for head_i in range(len(head_list)):
                cp_tree_str = cp_tree_str + str(head_list[head_i]) + cp_tree_str_list[head_i+1]
                assert cp_tree_str_list[head_i+1][0:len(token_list[head_i])+1] == ' '+token_list[head_i]

            # adding ner infor to cp tree
            cp_tree = generate_tree_from_str(cp_tree_str)
            for ner in ners:
                cp_tree.ner_match(ner[0], ner[1], False, True, ner[2])

            # writer to file
            data_writer.write(cp_tree.linearize()+'\n')

            # check infor
            check_subtree_infor(subtree_counter, cp_tree, head_list, token_list)

            if i % 1000 == 0:
                print('%d sentences are processed.' % i, flush=True)

        for key in subtree_counter:
            if subtree_counter[key] > fre:
                subtree_writer.write(key+'\t'+str(subtree_counter[key])+'\n')
        for key in token_counter:
            if token_counter[key] > fre:
                token_writer.write(key+'\t'+str(token_counter[key])+'\n')


def convert_word_to_char_dataset(path: str, char_path: str, subtree_path: str, token_path: str, fre: int):
    subtree_counter, token_counter = Counter(), Counter()
    with open(path, 'r', encoding='utf-8') as reader,\
         open(char_path, 'w', encoding='utf-8') as tree_writer,\
         open(subtree_path, 'w', encoding='utf-8') as subtree_writer,\
         open(token_path, 'w', encoding='utf-8') as token_writer:
        for line_i, line in enumerate(reader):
            tree = generate_tree_from_str(line.strip())
            leaves = list(tree.leaves())
            word_head_list = [int(item[0]) for item in leaves]
            word_list = [item[1] for item in leaves]
            char_head_list = []
            for i, word in enumerate(word_list):
                head_list = []
                cur_word_head = len(''.join(word_list[:i+1]))-1
                for _ in range(len(word)-1):
                    head_list.append(cur_word_head)
                head_list.append(len(''.join(word_list[:word_head_list[i]+1]))-1)
                char_head_list.append(head_list)
            tree_str = tree.linearize_convert_char(char_head_list)
            tree_writer.write(tree_str+'\n')
            tree = generate_tree_from_str(tree_str)
            leaves = list(tree.leaves())
            head_list = [int(item[0]) for item in leaves]
            token_list = [item[1] for item in leaves]
            for token in token_list:
                token_counter[token] += 1
            check_subtree_infor(subtree_counter, tree, head_list, token_list)

            if line_i % 1000 == 0:
                print('%d sentences are processed.' % line_i, flush=True)

        for key in subtree_counter:
            if subtree_counter[key] > fre:
                subtree_writer.write(key+'\t'+str(subtree_counter[key])+'\n')
        for key in token_counter:
            if token_counter[key] > fre:
                token_writer.write(key+'\t'+str(token_counter[key])+'\n')


if __name__ == '__main__':
    # annotate_main(7, './data/pretrain/small.corpus', 'zh')
    # multi_process_annotate()
    # generate_pretrain_dataset(
    #     './data/pretrain/zh.cpar.corpus', './data/pretrain/zh.dpar.corpus',
    #     './data/pretrain/zh.ner.corpus', './data/pretrain_word/train.corpus',
    #     './data/pretrain_word/subtree.vocab',
    #     './data/pretrain_word/token.vocab', 10
    # )
    start = time.time()
    convert_word_to_char_dataset(
        './data/pretrain_word/dev.corpus', './data/pretrain_char/dev.corpus',
        './data/pretrain_char/dev.subtree.vocab', './data/pretrain_char/dev.token.vocab', 10
    )
    print(time.time() - start)
