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
from utils import evaluate
from utils.optim import Optim
from typing import Tuple, List
from model.CPModel import CPModel
from config.cp_args import parse_args
from torch.utils.data import DataLoader
from utils.visual_logger import VisualLogger
from utils.trees import write_trees, InternalTreebankNode
from utils.joint_dataset import load_data, batch_filter, batch_spliter


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
    args.device = \
        torch.device('cuda:'+str(args.gpuid)) if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(args.gpuid)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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
               BATCH_MAX_SNT_LENGTH: int, evalb_path: str, type_: str)\
        -> Tuple[evaluate.FScore, List[InternalTreebankNode]]:
    trees_pred, trees_gold = [], []
    for insts in dataset:
        model.eval()
        insts, _, max_len = batch_filter(insts, language, DATASET_MAX_SNT_LENGTH)
        insts_list = batch_spliter(insts, max_len, BATCH_MAX_SNT_LENGTH)
        for insts in insts_list:
            trees_batch_pred, _ = model(insts)
            trees_batch_gold = insts['gold_trees']
            trees_pred.extend([tree.convert() for tree in trees_batch_pred])
            trees_gold.extend([tree.convert() for tree in trees_batch_gold])

    assert len(trees_pred) == len(trees_gold)
    eval_fscore = evaluate.evalb(evalb_path, trees_gold, trees_pred)
    print('Model performance in %s dataset: evalb: %s' % (type_, eval_fscore))
    torch.cuda.empty_cache()
    return eval_fscore, trees_pred


def main():
    # ====== preprocess ====== #
    args = preprocess()

    # ====== Loading dataset ====== #
    train_data, dev_data, test_data, vocabs = load_data(
        args.input, args.batch_size, args.shuffle, args.num_workers, args.drop_last
    )

    # ======= Preparing Model ======= #
    print("\nModel Preparing starts...")
    model = CPModel(
                vocabs,
                # Embedding
                args.subword,
                args.use_pos_tag,
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
                args.language,
                args.device
            ).to(args.device)
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
    patience = args.patience * (len(train_data)//(args.accum_steps*args.eval_interval))
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
                if args.early_stop:
                    patience -= 1
                print('model evaluating starts...', flush=True)
                fscore_dev, pred_dev = eval_model(
                    model, dev_data, args.language, args.DATASET_MAX_SNT_LENGTH, args.BATCH_MAX_SNT_LENGTH,
                    args.evalb_path, 'dev')
                fscore_test, pred_test = eval_model(
                    model, test_data, args.language, args.DATASET_MAX_SNT_LENGTH, args.BATCH_MAX_SNT_LENGTH,
                    args.evalb_path, 'test')
                visual_dic = {'F/dev': fscore_dev.fscore, 'F/test': fscore_test.fscore}
                args.visual_logger.visual_scalars(visual_dic, steps // args.accum_steps)
                if best_dev is None or fscore_dev.fscore > best_dev.fscore:
                    best_dev, best_test = fscore_dev, fscore_test
                    fitlog.add_best_metric({'f_dev': best_dev.fscore, 'f_test': best_test.fscore})
                    patience = args.patience * (len(train_data)//(args.accum_steps*args.eval_interval))
                    write_trees(os.path.join(args.save_path, 'dev.best.trees'), pred_dev)
                    write_trees(os.path.join(args.save_path, 'test.best.trees'), pred_test)
                    if args.save:
                        torch.save(model.pack_state_dict(), os.path.join(args.save_path, args.name+'.best.model.pt'))
                print('best performance:\ndev:%s\ntest:%s' % (best_dev, best_test))
                print('model evaluating ends...', flush=True)
                del pred_dev, pred_test
                if args.debug:
                    exit(0)
            steps += 1

        if args.early_stop:
            if patience < 0:
                print('early stop')
                break
        print('\n\n')

    # ====== postprocess ====== #
    postprocess(args, start)


if __name__ == '__main__':
    main()
