# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Neural model for NLP')

    # [Data]
    parser.add_argument('--input', type=str, default='./data/pretrain_char/', help='path of input data')
    parser.add_argument('--language', type=str, choices=['chinese', 'arabic', 'english'], default='chinese', help='language')
    parser.add_argument('--transliterate', default='', type=str, help='whether to transliterate when using BERT/XLNet')

    # [Train]
    parser.add_argument('--debug', default=False, type=bool, help='debug mode')
    parser.add_argument('--seed', default=2021, type=int, help='seed of random')
    parser.add_argument('--cuda', default=True, type=bool, help='whether to use cuda')
    parser.add_argument('--gpuid', default=0, type=int, help='id of gpu')
    parser.add_argument('--batch_size', default=16, type=int, help='how many insts per batch to load')
    parser.add_argument('--accum_steps', default=2, type=int, help='the number of accumulated steps before backward')
    parser.add_argument('--epoch', default=8, type=int, help='max training epoch')
    parser.add_argument('--log_interval', default=625, type=int, help='interval on print log info')
    parser.add_argument('--eval_interval', default=3125, type=int, help='interval on print evaluate model')
    parser.add_argument('--save_interval', default=15625, type=int, help='interval on save model')
    parser.add_argument('--early_stop', default=False, type=bool, help='early stop')
    parser.add_argument('--patience', default=6, type=int, help='early stop patience epoch')
    # [Optimizer]
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer used')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_fine_tune', default=0.00003, type=float, help='fine tune learning rate')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='lambda')
    parser.add_argument('--clip_grad', default=False, type=bool, help='whether to ues util.clip')
    parser.add_argument('--clip_grad_max_norm', default=4.0, type=float, help='clip_grad_max_norm')
    parser.add_argument('--warmup_steps', default=20000, type=int, help='warm up steps')
    parser.add_argument('--lr_decay_factor', default=1.0000001, type=float, help='decay factor of lr after warm up')

    # [Model]
    parser.add_argument('--name', default='pretrainModel', type=str, help='name of model')
    parser.add_argument('--subword', default='character_based', type=str, choices=['character_based', 'endpoint', 'startpoint', 'max_pool', 'avg_pool'], help='the method to represent word from BERT subword')
    # [Model-Embedding]
    parser.add_argument('--bert', default='./data/model/bert-base-chinese', type=str, help='BERT')
    parser.add_argument('--d_model', default=1024, type=int, help='model dimension')
    parser.add_argument('--partition', default=True, type=bool, help='whether to use content and position partition')
    parser.add_argument('--position_emb_dropout', default=0.0, type=float, help='position embedding dropout')
    parser.add_argument('--bert_emb_dropout', default=0.2, type=float, help='bert embedding dropout')
    parser.add_argument('--emb_dropout', default=0.0, type=float, help='embedding dropout')
    # [Model-Encoder]
    parser.add_argument('--layer_num', default=3, type=int, help='encoder layer num')
    parser.add_argument('--hidden_dropout', default=0.2, type=float, help='hidden states dropout in transformer')
    parser.add_argument('--attention_dropout', default=0.2, type=float, help='attention dropout in transformer')
    parser.add_argument('--dim_ff', default=2048, type=int, help='dim of ff sublayer in transformer')
    parser.add_argument('--nhead', default=8, type=int, help='head number')
    parser.add_argument('--kqv_dim', default=64, type=int, help='dimention of kqv')
    # [Model-classifier]
    parser.add_argument('--label_hidden', default=1250, type=int, help='dimention of label_hidden')

    # [Constants]
    parser.add_argument('--DATASET_MAX_SNT_LENGTH', default=200, type=str, help='when sentence length larger than it, drop it')
    parser.add_argument('--BATCH_MAX_SNT_LENGTH', default=80, type=str, help='when sentence max len bigger than it, split batch to sub-batch')

    args = parser.parse_args()
    return args
