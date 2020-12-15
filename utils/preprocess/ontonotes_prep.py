import os
import glob
import itertools


def generate_collection(tag="train"):
    results = itertools.chain.from_iterable(
        glob.iglob(os.path.join(root, '*.v4_gold_conll'))
        for root, dirs, files in os.walk('./data/conll-2012/v4/data/chinese/'+tag)
    )

    with open('./data/onto/'+tag+".corpus", 'w', encoding='utf-8') as writer:
        for cur_file in results:
            with open(cur_file, 'r', encoding='utf-8') as reader:
                # print(cur_file)
                flag = None
                text = ''
                for line in reader:
                    line = line.strip()
                    if len(line) == 0:
                        if text.count('\n') > 1:
                            writer.write(text + '\n')
                        text = ''
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

                    if len(word) > 15:
                        print(word, pos)
                        word = word[:15]
                    text += "\t".join([word, pos, cons, ner]) + '\n'


def generate_onto_cropus():
    generate_collection("train")
    generate_collection("dev")
    generate_collection("test")


def convert_parsing_data(data: str):
    root_dir = './data/onto'
    dataset = data + '.corpus'
    with open(os.path.join(root_dir, dataset), 'r', encoding='utf-8') as reader, \
         open(os.path.join(root_dir, 'parsing', dataset), 'w', encoding='utf-8') as writer:
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


if __name__ == '__main__':
    generate_onto_cropus()
    convert_parsing_dataset()
