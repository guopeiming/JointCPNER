# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com


FINE_TUNE_NAME = 'BERT.'  # warning: keep the model name and the value and the name in optim consistent.

LANGS_NEED_SEG = ['chinese', 'arabic']

BATCH_MAX_SNT_LENGTH = 125  # when sentence max len bigger than it, split batch to sub-batch
DATASET_MAX_SNT_LENGTH = 300  # when sentence length larger than it, drop it

DATASET_LIST = ('train', 'dev', 'test')  # train/dev/test list, warning: the order of the list.

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
TAG_UNK = "UNK"
EMPTY_LABEL = ()

PAD_STATEGY = 'longest'
TRUNCATION_STATEGY = 'longest_first'
CHARACTER_BASED = 'character_based'
