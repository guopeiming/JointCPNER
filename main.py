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
from config.args import parse_args
from torch.utils.data import DataLoader
from utils.visual_logger import VisualLogger
from utils.trees import write_trees, InternalTreebankNode
from utils.init_utils import model_init, load_data, vocabs_init
from config.Constants import BATCH_SNT_MAX_LENGTH, SUB_BATCH_TIMES


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
    # ====== fitlog init ====== #
    fitlog.commit(__file__)
    fitlog.debug(args.debug)
    fitlog.add_hyper(args)
    # ====== tb VisualLogger init ====== #
    args.visual_logger = VisualLogger(args.name) if not args.debug else None
    # ====== save path ====== #
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.save_path = os.path.join('./logs/', 'my_log-' + now_time)
    if not os.path.exists(args.save_path) and not args.debug:
        os.makedirs(args.save_path)
    # ====== cuda enable ====== #
    args.device = torch.device('cuda:'+str(args.gpuid)) if args.cuda and torch.cuda.is_available() else torch.device('cpu')
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
def eval_model(model: torch.nn.Module, dataset: DataLoader, treebank: List[InternalTreebankNode], evalb_path: str,
               type_: str) -> Tuple[evaluate.FScore, List[InternalTreebankNode]]:
    trees_predicted = []
    for insts in dataset:
        model.eval()
        trees_batch, _ = model(insts)
        trees_predicted.extend([tree.convert() for tree in trees_batch])
    eval_fscore = evaluate.evalb(evalb_path, treebank, trees_predicted)
    print('Model performance in %s dataset: %s' % (type_, eval_fscore))
    torch.cuda.empty_cache()
    return eval_fscore, trees_predicted


def main():
    # ====== preprocess ====== #
    args = preprocess()

    # ====== Loading dataset ====== #
    train_treebank, train_data, dev_treebank, dev_data, test_treebank, test_data = load_data(
        args.name, args.input, args.batch_size, args.shuffle, args.num_workers, args.drop_last
    )
    write_trees(os.path.join(args.save_path, 'dev.gold.trees'), dev_treebank)
    write_trees(os.path.join(args.save_path, 'test.gold.trees'), test_treebank)
    vocabs = vocabs_init(train_data)

    # ======= Preparing Model ======= #
    print("Model Preparing starts...")
    model = model_init(args, vocabs).to(args.device)
    print(model, end='\n\n\n')
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
    steps, loss_value = 1, 0.
    best_dev, best_test = None, None
    patience = args.patience * (len(train_data)//(args.accum_steps*args.eval_interval))
    for epoch_i in range(1, args.epoch):
        for batch_i, insts in enumerate(train_data, start=1):
            model.train()

            max_len = max([len(pos_tag) for pos_tag in insts['pos_tags']])
            batch_size = len(insts['pos_tags'])
            sub_batch_times = 1 if batch_size == 1 or max_len < BATCH_SNT_MAX_LENGTH else SUB_BATCH_TIMES
            for times_i in range(sub_batch_times):
                loss = model({key: insts[key][times_i:batch_size:sub_batch_times] for key in insts.keys()})
                if loss.item() > 0.:
                    loss.backward()
                    loss_value += loss.item()
                    assert not isinstance(loss_value, torch.Tensor), 'GPU memory leak'

            if batch_i == args.accum_steps and not args.debug:
                args.visual_logger.visual_histogram(model, steps//args.accum_steps)
            if steps % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            if steps % (args.accum_steps * args.log_interval) == 0:
                print(
                    '[%d/%d], [%d/%d] Loss: %.05f' %
                    (epoch_i, args.epoch, batch_i//args.accum_steps, len(train_data)//args.accum_steps, loss_value),
                    flush=True
                )
                visual_dic = {'loss/train': loss_value, 'lr': optimizer.get_lr()[0]}
                if args.clip_grad:
                    visual_dic['norm'] = optimizer.get_dynamic_gard_norm()
                args.visual_logger.visual_scalars(visual_dic, steps // args.accum_steps)
                loss_value = 0.
            if steps % (args.accum_steps * args.eval_interval) == 0:
                if args.early_stop:
                    patience -= 1
                print('model evaluating starts...')
                fscore_dev, predict_dev = eval_model(model, dev_data, dev_treebank, args.evalb_path, 'dev')
                fscore_test, predict_test = eval_model(model, test_data, test_treebank, args.evalb_path, 'test')
                visual_dic = {'F/dev': fscore_dev.fscore, 'F/test': fscore_test.fscore}
                args.visual_logger.visual_scalars(visual_dic, steps // args.accum_steps)
                if best_dev is None or fscore_dev.fscore >= best_dev.fscore:
                    best_dev, best_test = fscore_dev, fscore_test
                    fitlog.add_best_metric({'f_dev': best_dev.fscore, 'f_test': best_test.fscore})
                    patience = args.patience * (len(train_data)//(args.accum_steps*args.eval_interval))
                    write_trees(os.path.join(args.save_path, 'dev.best.trees'), predict_dev)
                    write_trees(os.path.join(args.save_path, 'test.best.trees'), predict_test)
                    if args.save:
                        torch.save(model.pack_state_dict(), os.path.join(args.save_path, args.name+'.best.model.pt'))
                print('best performance:\ndev:%s\ntest:%s' % (best_dev, best_test))
                print('model evaluating ends...', flush=True)
                del predict_dev, predict_test
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
