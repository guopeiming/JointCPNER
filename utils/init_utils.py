# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import os
import argparse
from typing import List, Dict, Union
from config import Constants
from utils.trees import load_trees, InternalTreebankNode
from utils.vocabulary import Vocabulary
from torch.utils.data import DataLoader
from config.Constants import DATASET_LIST
from utils import trees
from config.multi_models_config import MODEL_CLASSES, DATASET_CLASSES, MODEL_LIST


def model_init(args: argparse.Namespace, vocabs: Dict[str, Vocabulary]):
    """init the model by model name correspondingly.
    Args:
        args:
        vocabs:
    Returns:
        model instance
    """
    model = None
    if args.name in MODEL_LIST[0]:
        model = MODEL_CLASSES[args.name](
                    vocabs,
                    # Embedding
                    args.subword,
                    args.bert_path,
                    args.transliterate,
                    args.d_model,
                    args.partition,
                    args.pos_tag_emb_dropout,
                    args.position_emb_dropout,
                    args.bert_emb_dropout,
                    args.emb_dropout,
                    # Encoder
                    args.layer_num,
                    args.hidden_dropout,
                    args.attention_dropout,
                    args.dim_ff,
                    args.nhead,
                    args.kqv_dim,
                    # classifier
                    args.label_hidden,
                    args.device
                )
    # elif args.name in MODEL_LIST[1]:
    #     model = MODEL_CLASSES[args.name](args.label_num, args.device)
    else:
        print('model name does not exist.')
        exit(-1)
    return model


def load_data(name: str, path: str, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool) \
        -> List[Union[DataLoader, InternalTreebankNode]]:
    """load the datasets.
    Args:
        name: model name.
        path: path of input data.
        batch_size: the number of insts in a batch.
        shuffle: whether to shuffle the dataset.
        num_workers: the number of process to load data.
        drop_last: whether to drop the last data.
    Returns:
        treebank and Dataloader of train, dev, test data.
    """
    print('data loading starts...')
    res = []
    assert DATASET_LIST[0] == 'train'
    for item in DATASET_LIST:
        treebank = load_trees(os.path.join(path, item+'.corpus'), strip_top=False)
        res.append(treebank)
        parse_trees = [tree.convert() for tree in treebank]
        print('len(%s_data): %d' % (item, len(parse_trees)))

        dataset_class, pad_collate_fn = DATASET_CLASSES[name]
        data_loader = DataLoader(
                        dataset_class(parse_trees),
                        batch_size=batch_size,
                        shuffle=shuffle if item == DATASET_LIST[0] else False,
                        num_workers=num_workers,
                        collate_fn=pad_collate_fn,
                        drop_last=drop_last)
        res.append(data_loader)
    return res


def vocabs_init(train_data: DataLoader) -> Dict[str, Vocabulary]:
    print("Constructing vocabularies...")

    pos_tags_vocab = Vocabulary()
    pos_tags_vocab.index(Constants.START)
    pos_tags_vocab.index(Constants.STOP)
    pos_tags_vocab.index(Constants.TAG_UNK)

    words_vocab = Vocabulary()
    words_vocab.index(Constants.START)
    words_vocab.index(Constants.STOP)
    words_vocab.index(Constants.UNK)

    labels_vocab = Vocabulary()
    labels_vocab.index(Constants.EMPTY_LABEL)

    dataset = train_data.dataset
    for i in range(len(dataset)):
        tree = dataset[i]['gold_tree']
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                labels_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                pos_tags_vocab.index(node.tag)
                words_vocab.index(node.word)

    pos_tags_vocab.freeze()
    words_vocab.freeze()
    labels_vocab.freeze()

    print('len(pos_tags_vocab): %d\nlen(words_vocab): %d\nlen(labels_vocab): %d'
          % (pos_tags_vocab.size, words_vocab.size, labels_vocab.size))

    return {'pos_tags': pos_tags_vocab, 'words': words_vocab, 'labels': labels_vocab}
