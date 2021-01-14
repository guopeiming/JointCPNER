# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import os
import time
import torch
import fitlog
import random
import argparse
import datetime
import numpy as np
from utils.optim import Optim
from utils import ner_evaluate
from model.NERModel import BertCRFNER, BertEncoderNER, BertNER, SpanNER
from typing import Tuple, List, Dict
from config.ner_args import parse_args
from torch.utils.data import DataLoader
from utils.visual_logger import VisualLogger
from utils.ner_dataset import load_data, write_ners, batch_filter, batch_spliter


def preprocess() -> argparse.Namespace:
    """
    preprocess of training
    :return: config args
    """
    print('preprocessing starts...\n')
    # ====== parse arguments ====== #
    args = parse_args()
    # ====== set random seed ====== #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # ====== save path ====== #
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.save_path = os.path.join('./logs/', 'my_log-' + now_time)
    if not os.path.exists(args.save_path) and not args.debug:
        os.makedirs(args.save_path)
    # ====== fitlog init ====== #
    fitlog.commit(__file__)
    fitlog.debug(args.debug)
    fitlog.add_hyper(args)
    # ====== tb VisualLogger init ====== #
    args.visual_logger = VisualLogger(args.save_path) if not args.debug else None
    # ====== cuda enable ====== #
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    args.device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    # ====== others ====== #
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(4)
    print(args, end='\n\n')
    return args


def postprocess(args: argparse.Namespace, start: float):
    exe_time = time.time() - start
    print('Executing time: %dh:%dm:%ds.' % (exe_time//3600, (exe_time//60) % 60, exe_time % 60))
    fitlog.finish()
    args.visual_logger.close()
    print('training ends.')


@torch.no_grad()
def eval_model(model: torch.nn.Module, dataset: DataLoader, language: str, DATASET_MAX_SNT_LENGTH: int,
               BATCH_MAX_SNT_LENGTH: int, type_: str)\
        -> Tuple[ner_evaluate.NERFScore, Dict[str, List[List[str]]]]:
    tags_pred, tags_gold, snts = [], [], []
    for insts in dataset:
        model.eval()
        insts, _, max_len = batch_filter(insts, language, DATASET_MAX_SNT_LENGTH)
        insts_list = batch_spliter(insts, max_len, BATCH_MAX_SNT_LENGTH)
        for insts in insts_list:
            tags_batch_pred = model(insts)
            snts.extend(insts['snts'])
            tags_batch_gold = insts['golds']
            tags_pred.extend(tags_batch_pred)
            tags_gold.extend(tags_batch_gold)

    assert len(tags_pred) == len(tags_gold)
    fscore = ner_evaluate.cal_performance(tags_pred, tags_gold)
    print('Model performance in %s dataset: metric: %s' % (type_, fscore))
    torch.cuda.empty_cache()
    return fscore, {'snts': snts, 'pred_tags': tags_pred, 'gold_tags': tags_gold}


def main():
    # ====== preprocess ====== #
    args = preprocess()

    # ====== Loading dataset ====== #
    train_data, dev_data, test_data, vocab = load_data(
        args.input, args.batch_size, args.accum_steps, args.shuffle, args.num_workers, args.drop_last
    )

    # ======= Preparing Model ======= #
    print("\nModel Preparing starts...")
    model = None
    if args.name == 'BertCRFNER':
        model = BertCRFNER(
            vocab,
            # Embedding
            args.subword,
            args.transliterate,
            args.bert_path,
            args.bert_emb_dropout,
            args.device
        ).cuda()
    elif args.name == 'BertEncoderNER':
        model = BertEncoderNER(
            vocab,
            # Embedding
            args.subword,
            args.transliterate,
            args.bert_path,
            args.position_emb_dropout,
            args.bert_emb_dropout,
            args.emb_dropout,
            args.d_model,
            args.partition,
            args.layer_num,
            args.hidden_dropout,
            args.attention_dropout,
            args.dim_ff,
            args.nhead,
            args.kqv_dim,
            # classifier
            args.label_hidden,
            args.device
        ).cuda()
    elif args.name == 'SpanNER':
        model = SpanNER(
            # Embedding
            args.subword,
            args.transliterate,
            args.bert_path,
            args.position_emb_dropout,
            args.bert_emb_dropout,
            args.emb_dropout,
            args.d_model,
            args.partition,
            args.layer_num,
            args.hidden_dropout,
            args.attention_dropout,
            args.dim_ff,
            args.nhead,
            args.kqv_dim,
            # classifier
            args.label_hidden,
            args.device
        ).cuda()
    elif args.name == 'BertNER':
        model = BertNER(
            vocab,
            # Embedding
            args.subword,
            args.transliterate,
            args.bert_path,
            args.bert_emb_dropout,
            args.device
        ).cuda()
    else:
        print('model name error')
        exit(-1)
    # print(model, end='\n\n\n')
    optimizer = Optim(model, args.optim, args.lr, args.lr_fine_tune, args.warmup_steps, args.lr_decay_factor,
                      args.weight_decay, args.clip_grad, args.clip_grad_max_norm)
    optimizer.zero_grad()
    # if args.freeze_bert:
    #     optimizer.set_freeze_by_idxs([str(num) for num in range(0, config.freeze_bert_layers)], True)
    #     optimizer.free_embeddings()
    #     optimizer.freeze_pooler()
    #     print('freeze model of BERT %d layers' % config.freeze_bert_layers)

    # ========= Training ========= #
    print('Training starts...')
    start = time.time()
    steps, loss_value, total_batch_size = 1, 0., 0
    best_dev, best_test = None, None
    patience = args.patience
    for epoch_i in range(1, args.epoch):
        for batch_i, insts in enumerate(train_data, start=1):
            model.train()

            insts, batch_size, max_len = batch_filter(insts, args.language, args.DATASET_MAX_SNT_LENGTH)
            insts_list = batch_spliter(insts, max_len, args.BATCH_MAX_SNT_LENGTH)
            total_batch_size += batch_size
            for insts in insts_list:
                loss = model(insts)
                if loss.item() > 0.:
                    loss.backward()
                    loss_value += loss.item()
                    assert not isinstance(loss_value, torch.Tensor), 'GPU memory leak'

            if batch_i == args.accum_steps and not args.debug:
                args.visual_logger.visual_histogram(model, steps//args.accum_steps)
            if steps % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if steps % (args.accum_steps * args.log_interval) == 0:
                print(
                    '[%d/%d], [%d/%d] Loss: %.05f' %
                    (epoch_i, args.epoch, batch_i//args.accum_steps, len(train_data)//args.accum_steps,
                     loss_value/total_batch_size), flush=True
                )
                visual_dic = {'loss/train': loss_value, 'lr': optimizer.get_lr()[0]}
                if args.clip_grad:
                    visual_dic['norm'] = optimizer.get_dynamic_gard_norm()
                args.visual_logger.visual_scalars(visual_dic, steps // args.accum_steps)
                loss_value, total_batch_size = 0., 0
                torch.cuda.empty_cache()
            if steps % (args.accum_steps * args.eval_interval) == 0:
                print('model evaluating starts...')
                fscore_dev, pred_dev = eval_model(
                    model, dev_data, args.language, args.DATASET_MAX_SNT_LENGTH, args.BATCH_MAX_SNT_LENGTH, 'dev'
                )
                fscore_test, pred_test = eval_model(
                    model, test_data, args.language, args.DATASET_MAX_SNT_LENGTH, args.BATCH_MAX_SNT_LENGTH, 'test'
                )
                visual_dic = {'F/dev': fscore_dev.fscore, 'F/test': fscore_test.fscore}
                args.visual_logger.visual_scalars(visual_dic, steps // args.accum_steps)
                if best_dev is None or fscore_dev.fscore >= best_dev.fscore:
                    best_dev, best_test = fscore_dev, fscore_test
                    fitlog.add_best_metric({'f_dev': best_dev.fscore, 'f_test': best_test.fscore})
                    patience = args.patience
                    write_ners(
                        os.path.join(args.save_path, 'dev.pred.best.ners'),
                        os.path.join(args.save_path, 'dev.gold.ners'), pred_dev)
                    write_ners(
                        os.path.join(args.save_path, 'test.pred.best.ners'),
                        os.path.join(args.save_path, 'test.gold.ners'), pred_test)
                    if args.save:
                        torch.save(model.pack_state_dict(), os.path.join(args.save_path, args.name+'.best.model.pt'))
                print('best performance:\ndev: %s\ntest: %s' % (best_dev, best_test))
                print('model evaluating ends...', flush=True)
                del pred_dev, pred_test
                if args.debug:
                    exit(0)
            steps += 1

        if args.early_stop:
            patience -= 1
            if patience < 0:
                print('early stop')
                break

    # ====== postprocess ====== #
    postprocess(args, start)


if __name__ == '__main__':
    main()
