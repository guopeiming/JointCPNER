# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Neural model for NLP')

    # [Data]
    parser.add_argument('--input', type=str, default='./data/onto/ner/', help='path of input data')
    parser.add_argument('--language', type=str, choices=['chinese', 'arabic', 'english'], default='chinese', help='language')
    # [Train]
    parser.add_argument('--debug', default=False, type=bool, help='debug mode')
    parser.add_argument('--seed', default=2021, type=int, help='seed of random')
    parser.add_argument('--cuda', default=True, type=bool, help='whether to use cuda')
    parser.add_argument('--gpuid', default=5, type=int, help='id of gpu')
    parser.add_argument('--batch_size', default=32, type=int, help='how many insts per batch to load')
    parser.add_argument('--accum_steps', default=1, type=int, help='the number of accumulated steps before backward')
    parser.add_argument('--shuffle', default=True, type=bool, help='set to True to have the data reshuffled at every epoch')
    parser.add_argument('--drop_last', default=False, type=bool, help='whether to drop the last data')
    parser.add_argument('--num_workers', default=4, type=int, help='how many subprocesses to use for data loading')
    parser.add_argument('--epoch', default=35, type=int, help='max training epoch')
    parser.add_argument('--log_interval', default=100, type=int, help='interval on print log info')
    parser.add_argument('--eval_interval', default=350, type=int, help='interval on print evaluate model')
    parser.add_argument('--early_stop', default=True, type=bool, help='early stop')
    parser.add_argument('--patience', default=3, type=int, help='early stop patience epoch')
    parser.add_argument('--save', default=False, type=bool, help='whether to save model')
    # [Optimizer]
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer used')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_fine_tune', default=0.00001, type=float, help='fine tune learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='lambda')
    parser.add_argument('--clip_grad', default=False, type=bool, help='whether to ues util.clip')
    parser.add_argument('--clip_grad_max_norm', default=4.0, type=float, help='clip_grad_max_norm')
    parser.add_argument('--warmup_steps', default=9000, type=int, help='warm up steps')
    parser.add_argument('--lr_decay_factor', default=1.00001, type=float, help='decay factor of lr after warm up')

    # [Model]
    parser.add_argument('--name', default='NERModel', type=str, help='name of model')
    parser.add_argument('--bert_path', default='/home/data/embedding/bert-base-chinese/', type=str, help='path of BERT')

    # [Constants]
    parser.add_argument('--DATASET_MAX_SNT_LENGTH', default=300, type=str, help='when sentence length larger than it, drop it')
    parser.add_argument('--BATCH_MAX_SNT_LENGTH', default=125, type=str, help='when sentence max len bigger than it, split batch to sub-batch')

    args = parser.parse_args()
    return args
