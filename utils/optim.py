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

        params_list_finetune, params_list_norm = [], []
        params_finetune, params_total = 0, 0
        for name, param in model.named_parameters():
            if name.find(Constants.FINE_TUNE_NAME) != -1:
                params_list_finetune.append(param)
                params_finetune += param.numel()
            else:
                params_list_norm.append(param)
            params_total += param.numel()

        params_group = [
            {'params': params_list_finetune, 'lr': lr_fine_tune},
            {'params': params_list_norm, 'lr': lr},
        ]
        lr_lambdas_group = [
            LrLambdaExp(lr_fine_tune, warmup_steps, lr_decay_factor),
            LrLambdaExp(lr, warmup_steps, lr_decay_factor)
        ]
        assert len(params_group) == len(lr_lambdas_group)
        assert all(params['lr'] == lr_lambdas.init_lr for params, lr_lambdas in zip(params_group, lr_lambdas_group))

        if opti == 'Adam':
            self._optimizer = torch.optim.Adam(params_group, weight_decay=weight_decay)
        elif opti == 'SGD':
            self._optimizer = torch.optim.SGD(params_group, weight_decay=weight_decay)
        else:
            print('optimizer name is wrong.')
            exit(-1)

        self.model = model
        self.clip_grad = clip_grad
        self.clip_grad_max_norm = clip_grad_max_norm
        self.scheduler = LRScheduler(self._optimizer, lr_lambdas_group)
        self.dynamic_norm = 0.0
        print('len(total_parameters): %d, len(fine_tuned_parameters): %d, ratio: %.03f'
              % (params_total, params_finetune, params_finetune/params_total*100), end='\n\n')

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


class LrLambdaExp(object):
    """My lr lambda expression.
    """
    def __init__(self, init_lr: float, warmup_steps: int, decay_factor: float):
        super(LrLambdaExp, self).__init__()
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor

    def __call__(self, step: int, curr_lr: float) -> float:
        return ((step+1)/self.warmup_steps)*self.init_lr if step < self.warmup_steps else curr_lr**self.decay_factor
