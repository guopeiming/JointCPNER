import os
import re
import glob
import itertools
from typing import Counter
from utils.preprocess.trees import load_trees
from utils.ner_dataset import load_data_from_file
from utils.ner_evaluate import _bio_tag_to_spans


def generate_collection_english(tag="train"):
    """generate corpus including ner and constituency parsing.

    notes:
        we keep the first 15 characters if a word length larger than 15, which words all are URL link.

    notes:
        if language is english, do not cut 15!!!!
        ner label: B I O
    """
    results = itertools.chain.from_iterable(
        glob.iglob(os.path.join(root, '*.v4_gold_conll'))
        for root, dirs, files in os.walk('./data/conll-2012/v4/data/'+tag+'/data/'+'english/')
    )

    with open('./data/onto/english_origin/'+tag+".corpus", 'w', encoding='utf-8') as writer:
        for cur_file in results:
            with open(cur_file, 'r', encoding='utf-8') as reader:
                # print(cur_file)
                flag = None
                for line in reader:
                    line = line.strip()
                    if len(line) == 0:
                        writer.write('\n')
                        continue
                    if line.startswith('#'):
                        continue

                    items = line.split()
                    assert len(items) >= 11

                    word, pos, cons, ori_ner = items[3], items[4], items[5], items[10]
                    ner = ori_ner
                    # print(word, pos, cons, ner)
                    if ori_ner == "*":
                        if flag is None:
                            ner = "O"
                        else:
                            ner = "I-" + flag
                    elif ori_ner == "*)":
                        assert flag is not None
                        ner = "I-" + flag
                        flag = None
                    elif ori_ner.startswith("(") and ori_ner.endswith("*"):
                        assert len(ori_ner) > 2 and flag is None
                        flag = ori_ner[1:-1]
                        ner = "B-" + flag
                    elif ori_ner.startswith("(") and ori_ner.endswith(")"):
                        assert len(ori_ner) > 2 and flag is None
                        ner = "B-" + ori_ner[1:-1]
                    else:
                        print('error!!!')
                        exit(-1)

                    if len(word) > 20:
                        if re.match(r'http|<http', word):
                            word = 'http'
                        elif re.match(r'www', word):
                            word = 'www'
                        elif re.match(r'-----|\.\.\.\.\.|=====|_____', word):
                            word = word[:5]
                        else:
                            print(word, cur_file)
                    writer.write("\t".join([word, pos, cons, ner]) + '\n')

    # for cur_file in results:
    #     with open(cur_file, 'r', encoding='utf-8') as reader:
    #         for line in reader:
    #             line = line.strip()
    #             if len(line) == 0:
    #                 continue
    #             if line.startswith('#'):
    #                 continue

    #             items = line.split()
    #             assert len(items) >= 11

    #             word, pos, cons, ori_ner = items[3], items[4], items[5], items[10]

    #             if len(word) > 15:
    #                 if re.search(r'http|ｈｔｔｐ|.com|．ｃｏｍ|．|Ｂｌｏｇ|Ｍｏｖｅ|ｅ７ｂ１|＜|＞|ＯｔｒＰ|ＣＴＲＬ|\[|ｋｉｌｌ|Ｒａｉｓｅ|ｓｕｄｏ', word):
    #                     if pos == 'URL':
    #                         word = 'http://'
    #                     elif re.search(r'Ｂｌｏｇ', word):
    #                         word = 'Blog'
    #                     elif re.search(r'ｗｗｗ．', word):
    #                         word = 'www.'
    #                     elif re.search(r'．．．', word):
    #                         word = '....'
    #                     else:
    #                         print(word, pos, cur_file)
    #                         word = word[:5]
    #                     print(word)
    #                 elif word.startswith('唐'):
    #                     word = re.sub(r'\{.*\}', '', word)
    #                     print(word)
    #                 else:
    #                     word = word
    #                 print(word, ori_ner, cur_file)


def generate_onto_cropus():
    generate_collection_english("train")
    generate_collection_english("dev")
    generate_collection_english("test")


def convert_parsing_data(data: str):
    root_dir = './data/onto'
    dataset = data + '.corpus'
    with open(os.path.join(root_dir, 'english_origin', dataset), 'r', encoding='utf-8') as reader, \
         open(os.path.join(root_dir, 'parsing_en', dataset), 'w', encoding='utf-8') as writer:
        for line in reader:
            line = line.strip()
            if len(line) == 0:
                writer.write('\n')
                continue
            items = line.split('\t')
            assert len(items) == 4
            word, pos, cons = items[0], items[1], items[2]
            assert '*' in cons
            writer.write(cons.replace('*', '(%s %s)' % (pos, word)))


def convert_parsing_dataset():
    convert_parsing_data('train')
    convert_parsing_data('dev')
    convert_parsing_data('test')


def convert_ner_data(data: str):
    root_dir = './data/onto'
    dataset = data + '.corpus'
    with open(os.path.join(root_dir, 'english_origin', dataset), 'r', encoding='utf-8') as reader, \
         open(os.path.join(root_dir, 'ner_en', dataset), 'w', encoding='utf-8') as writer:
        for line in reader:
            line = line.strip()
            if len(line) == 0:
                writer.write('\n')
                continue
            items = line.split('\t')
            assert len(items) == 4
            word, ner = items[0], items[3]
            writer.write(word+'\t'+ner+'\n')


def convert_ner_dataset():
    convert_ner_data('train')
    convert_ner_data('dev')
    convert_ner_data('test')


def match_ner_cp():
    match_counter, conti_counter, not_match_counter = Counter(), Counter(), Counter()
    match, conti_match, not_match = 0, 0, 0
    trees = load_trees('./data/onto/parsing_en/train.corpus')
    ner_snts, ner_golds = load_data_from_file('./data/onto/ner_en/train.corpus')
    assert len(trees) == len(ner_snts)
    with open('./not_mach.txt', 'w', encoding='utf-8') as writer:  #, open('./data/onto/temp/train.corpus', 'w', encoding='utf-8') as ner_writer:
        for snt, ner_gold, tree in zip(ner_snts, ner_golds, trees):
            snt, ner_gold = snt.split(), ner_gold.split()
            assert len(list(tree.leaves())) == len(snt)
            spans = _bio_tag_to_spans(ner_gold)
            for span in spans:
                match_type = tree.ner_match(span[1][0], span[1][1], False, False)
                if match_type == 0:
                    not_match += 1

                    # ======================================
                    # for i in range(span[1][0], span[1][1]):
                    #     ner_gold[i] = 'O'
                    # ======================================

                    not_match_counter.update([span[0]])
                    writer.write('\t'.join([span[0], ''.join(snt[span[1][0]:span[1][1]]), tree.linearize().replace('(', '[').replace(')', ']')+'\n']))
                elif match_type == 1:
                    match_counter.update([span[0]])
                    match += 1
                elif match_type == 2:
                    conti_counter.update([span[0]])
                    conti_match += 1
                else:
                    print('error')
                    exit(-1)

            # ======================================
            # assert len(snt) == len(ner_gold)
            # for char, ner in zip(snt, ner_gold):
            #     ner_writer.write(char+'\t'+ner+'\n')
            # ner_writer.write('\n')
            # ======================================

    print(match, conti_match, not_match, not_match/(not_match+match+conti_match))
    print(match_counter)
    print(conti_counter)
    print(not_match_counter)


def convert_joint_data(data: str, up: bool):
    root_dir = './data/onto'
    dataset = data + '.corpus'
    trees = load_trees(os.path.join(root_dir, 'parsing_en', dataset))
    ner_snts, ner_golds = load_data_from_file(os.path.join(root_dir, 'ner_en', dataset))
    match, conti_match, not_match = 0, 0, 0

    with open(os.path.join(root_dir, 'joint_en', dataset), 'w', encoding='utf-8') as writer:
        assert len(trees) == len(ner_snts)
        for snt, ner_gold, tree in zip(ner_snts, ner_golds, trees):
            snt, ner_gold = snt.split(), ner_gold.split()
            assert len(list(tree.leaves())) == len(snt)
            spans = _bio_tag_to_spans(ner_gold)

            for span in spans:
                match_type = tree.ner_match(span[1][0], span[1][1], up, change_label=True, span_label=span[0].upper())
                if match_type == 0:
                    not_match += 1
                elif match_type == 1:
                    match += 1
                elif match_type == 2:
                    conti_match += 1
                else:
                    print('error')
                    exit(-1)
            writer.write(tree.linearize() + '\n')

    print(match, conti_match, not_match, not_match/(not_match+match+conti_match))


def convert_joint_dataset():
    convert_joint_data('train', False)
    convert_joint_data('dev', False)
    convert_joint_data('test', False)


if __name__ == '__main__':
    # generate_onto_cropus()
    # convert_parsing_dataset()
    # convert_ner_dataset()
    # match_ner_cp()
    convert_joint_dataset()
