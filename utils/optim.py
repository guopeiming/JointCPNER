# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import torch.optim
from typing import List
from config import Constants
import torch.nn.utils as utils
from collections.abc import Iterable
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Optim:
    """
    My Optim class
    """
    name_list = ['Adam', 'SGD']

    def __init__(self, model: torch.nn.Module, opti: str, lr: float, lr_fine_tune: float, warmup_steps: int,
                 lr_decay_factor: float, weight_decay: float, clip_grad: bool, clip_grad_max_norm: float):
        assert opti in Optim.name_list, 'optimizer name is wrong.'

        params_list = []
        lr_lambdas = []
        params_finetune, params_total = 0, 0
        for name, param in model.named_parameters():
            if name.find(Constants.FINE_TUNE_NAME) != -1:
                params_list.append({'params': param, 'lr': lr_fine_tune})
                lr_lambdas.append(get_lr_scheduler_lambda(lr_fine_tune, warmup_steps, lr_decay_factor))
                params_finetune += param.numel()
            else:
                params_list.append({'params': param, 'lr': lr})
                lr_lambdas.append(get_lr_scheduler_lambda(lr, warmup_steps, lr_decay_factor))
            params_total += param.numel()

        if opti == 'Adam':
            self._optimizer = torch.optim.Adam(params_list, weight_decay=weight_decay)
        elif opti == 'SGD':
            self._optimizer = torch.optim.SGD(params_list, lr, weight_decay)
        else:
            print('optimizer name is wrong.')
            exit(-1)

        self.model = model
        self.clip_grad = clip_grad
        self.clip_grad_max_norm = clip_grad_max_norm
        self.scheduler = LRScheduler(self._optimizer, lr_lambdas)
        self.dynamic_norm = 0.0
        print('len(total_parameters): %d, len(fine_tuned_parameters): %d, ratio: %.05f'
              % (params_total, params_finetune, params_finetune/params_total), end='\n\n')

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        if self.clip_grad:
            total_norm = utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_max_norm).item()
            assert not isinstance(total_norm, torch.Tensor)
            if total_norm > self.clip_grad_max_norm:
                print('norm is large, which value is %f' % total_norm)
            self.dynamic_grad_norm = total_norm

        self._optimizer.step()

        self.scheduler.step()

    def get_optimizer(self):
        return self._optimizer

    def get_lr(self):
        return self.scheduler.get_lr()

    def get_dynamic_gard_norm(self):
        return self.dynamic_grad_norm

    def set_freeze_by_idxs(self, idxs, free):
        if not isinstance(idxs, Iterable):
            idxs = [idxs]

        for name, model_layer in self.model.bert_model.encoder.layer.named_children():
            if name not in idxs:
                continue
            for param in model_layer.parameters():
                param.requires_grad_(not free)

    def freeze_pooler(self, free=True):
        for param in self.model.bert_model.pooler.parameters():
            param.requires_grad_(not free)

    def free_embeddings(self, free=True):
        for param in self.model.bert_model.embeddings.parameters():
            param.requires_grad_(not free)


class LRScheduler(LambdaLR):
    """
    My lr scheduler class, it can convert to warm_up, no_warm_up, constant, warm_up_reduce learning rate mode
    by parameters.
    """
    def __init__(self, optimizer: Optimizer, lr_lambdas: List, last_epoch=-1):
        self.curr_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
        super(LRScheduler, self).__init__(optimizer, lr_lambdas, last_epoch)

    def get_lr(self):
        self.curr_lrs = [lambda_(self.last_epoch, lr) for lambda_, lr in zip(self.lr_lambdas, self.curr_lrs)]
        return self.curr_lrs.copy()


def get_lr_scheduler_lambda(init_lr, warmup_step, decay_factor):
    return lambda step, curr_lr: ((step+1)/warmup_step)*init_lr if step < warmup_step else curr_lr**decay_factor
