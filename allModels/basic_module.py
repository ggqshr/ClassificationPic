import time

import torch as t
from torch import nn


class BasicModule(nn.Module):
    def __int__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, accuracy, name=None):
        if name is None:
            prefix = "checkpoints/" + self.model_name + "_"
            name = time.strftime(prefix + "$$$" + str(accuracy) + "$$$%m%d_%H%M%S.pth")
        t.save(self.state_dict(), name)
        return name


class Flattern(t.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor):
        return x.view(x.size(0), -1)
