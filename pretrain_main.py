# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import os
import time
import torch
import random
import argparse
import datetime
import numpy as np
from utils.optim import Optim
from config.pretrain_args import parse_args
from model.PretrainModel import PretrainModel
from torch.utils.data import DataLoader
from utils.pretrain_dataset import load_data, batch_filter, batch_spliter


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
    # ====== cuda enable ====== #
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    args.device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    # ====== others ====== #
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # torch.set_num_threads(6)
    print(args, end='\n\n')
    return args


def postprocess(args: argparse.Namespace, start: float):
    exe_time = time.time() - start
    print('Executing time: %dh:%dm:%ds.' % (exe_time//3600, (exe_time//60) % 60, exe_time % 60))
    print('training ends.')


@torch.no_grad()
def eval_model(
    model: torch.nn.Module, dataset: DataLoader, DATASET_MAX_SNT_LENGTH: int, BATCH_MAX_SNT_LENGTH: int, type_: str
) -> float:
    torch.cuda.empty_cache()
    total_subtree, tp_subtree = 0, 0
    total_head, tp_head = 0, 0
    total_mask_lm, tp_mask_lm = 0, 0
    for insts in dataset:
        model.eval()
        insts, _, max_len = batch_filter(insts, DATASET_MAX_SNT_LENGTH)
        insts_list = batch_spliter(insts, max_len, BATCH_MAX_SNT_LENGTH)
        for insts in insts_list:
            subtree_pred, head_pred, mask_lm_pred, subtree_gold, head_gold, mask_lm_gold = model(insts)
            assert len(subtree_pred) == len(subtree_gold)
            for p, g in zip(subtree_pred, subtree_gold):
                if p == g:
                    tp_subtree += 1
                total_subtree += 1
            assert len(head_pred) == len(head_gold)
            for p, g in zip(head_pred, head_gold):
                if p == g:
                    tp_head += 1
                total_head += 1
            assert len(mask_lm_pred) == len(mask_lm_gold)
            for p, g in zip(mask_lm_pred, mask_lm_gold):
                if p == g:
                    tp_mask_lm += 1
                total_mask_lm += 1
    print(
        'Model performance in %s dataset: subtree_acc: %.03f, head_acc: %.03f, mask_lm: %.03f, total_acc: %.03f' %
        (
            type_, tp_subtree/total_subtree*100, tp_head/total_head*100, tp_mask_lm/total_mask_lm*100,
            (tp_subtree+tp_head+tp_mask_lm)/(total_subtree+total_head+total_mask_lm)*100
        )
    )
    torch.cuda.empty_cache()
    return (tp_subtree+tp_head+tp_mask_lm)/(total_subtree+total_head+total_mask_lm)*100


def main():
    # ====== preprocess ====== #
    args = preprocess()
    # ====== Loading dataset ====== #
    train_data, dev_data, subtree_vocab, token_vocab = load_data(
        args.input, args.batch_size, args.language, args.subword, args.debug
    )

    # ======= Preparing Model ======= #
    print("\nModel Preparing starts...")
    model = PretrainModel(
                subtree_vocab,
                token_vocab,
                # Embedding
                args.subword,
                args.bert,
                args.transliterate,
                args.d_model,
                args.partition,
                args.position_emb_dropout,
                args.bert_emb_dropout,
                args.emb_dropout,
                args.layer_num,
                args.hidden_dropout,
                args.attention_dropout,
                args.dim_ff,
                args.nhead,
                args.kqv_dim,
                args.label_hidden,
                # classifier
                args.language,
                args.device
            ).cuda()
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
    best_dev = 0.
    patience = args.patience
    for epoch_i in range(1, args.epoch+1):
        for batch_i, insts in enumerate(train_data, start=1):
            model.train()

            insts, batch_size, max_len = batch_filter(insts, args.DATASET_MAX_SNT_LENGTH)
            insts_list = batch_spliter(insts, max_len, args.BATCH_MAX_SNT_LENGTH)
            total_batch_size += batch_size
            for insts in insts_list:
                loss = model(insts)
                if loss.item() > 0.:
                    loss.backward()
                    loss_value += loss.item()
                    assert not isinstance(loss_value, torch.Tensor), 'GPU memory leak'

            if steps % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if steps % (args.accum_steps * args.log_interval) == 0:
                print(
                    '[%d/%d], [%d/%d] Loss: %.05f' %
                    (epoch_i, args.epoch, batch_i//args.accum_steps, len(train_data)//args.accum_steps,
                     loss_value/total_batch_size), flush=True
                )
                loss_value, total_batch_size = 0., 0
                torch.cuda.empty_cache()
            if steps % (args.accum_steps * args.eval_interval) == 0:
                patience -= 1
                print('model evaluating starts...', flush=True)
                dev_acc = eval_model(
                    model, dev_data, args.DATASET_MAX_SNT_LENGTH, args.BATCH_MAX_SNT_LENGTH, 'dev'
                )
                if best_dev < dev_acc:
                    best_dev = dev_acc
                    patience = args.patience
                    model.save_models(os.path.join(args.save_path, 'best.model/'))
                print('best performance: ACC: %.03f' % (best_dev*100))
                print('model evaluating ends...', flush=True)
                if args.early_stop:
                    if patience < 0:
                        break
            if steps % (args.accum_steps * args.save_interval) == 0:
                model.save_models(os.path.join(args.save_path, str(steps)+'.steps.model/'))
            steps += 1

        if args.early_stop:
            if patience < 0:
                print('early stop')
                break

    # ====== postprocess ====== #
    postprocess(args, start)


if __name__ == '__main__':
    main()
