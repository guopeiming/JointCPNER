# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com

DATASET_LIST = ('train', 'dev', 'test')  # train/dev/test list, warning: the order of the list.

SENTENCE_MAX_LEN = 250

FINE_TUNE_NAME = '.BERT.'  # warning: keep the model name and the value and the name in optim consistent.

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
TAG_UNK = "UNK"
EMPTY_LABEL = ()

PAD_STATEGY = 'longest'
TRUNCATION_STATEGY = 'longest_first'

CHARACTER_BASED = 'character_based'

BATCH_SNT_MAX_LENGTH = 100  # when sentence max len bigger than it, split batch to sub-batch
SUB_BATCH_TIMES = 2
