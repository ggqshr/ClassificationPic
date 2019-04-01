from torchvision.models import alexnet,vgg
from .basic_module import BasicModule
from torch import nn
from torch.optim import Adam


class AlexModel(BasicModule):
    def __init__(self):
        super().__init__()
        self.model_name = "alexnet"
        self.model = alexnet(pretrained=True)
