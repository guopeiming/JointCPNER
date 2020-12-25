# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class VisualLogger:
    def __init__(self, path: str):
        self.writer = SummaryWriter(path)

    def visual_scalars(self, dic, step):
        for tag in dic:
            self.writer.add_scalar(tag, dic[tag], step)

    def visual_histogram(self, model: nn.Module, step):
        for tag, values in model.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram(tag, values, step)
            if values.grad is not None:
                self.writer.add_histogram(tag+'/grad', values.grad, step)

    def visual_graph(self, model, input_to_model, verbose=False):
        self.writer.add_graph(model, input_to_model, verbose)

    def close(self):
        self.writer.close()
