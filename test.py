from torchvision.datasets import ImageFolder
from torch.utils import data
import torch as t
from util import PairDataSet, Visualizer
from allModels import AlexModel


def test1():
    vis = Visualizer(server="cal")
    dd = PairDataSet(r"./data/totrain")
    loader = data.DataLoader(dd, shuffle=True, batch_size=2)
    d1 = loader.__iter__().next()

    vis.img("11", (d1[0][0][0] * 0.255 + 0.45).clamp(min=0, max=1))
    vis.img("22", (d1[0][1][0] * 0.255 + 0.45).clamp(min=0, max=1))
    print(d1[1])


model = AlexModel()
for para in model.model.features.parameters():
    para.requires_grad=False
print(model)
for para in model.model.named_parameters():
    print(para)
