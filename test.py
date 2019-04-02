from torchvision.datasets import ImageFolder
from torch.utils import data
import torch as t
from util import PairDataSet, Visualizer

vis = Visualizer(server="cal")
dd = PairDataSet(r"D:\project\pytorch\Pytorch_Project\DataGenerator\train")
loader = data.DataLoader(dd, shuffle=True)
d1 = loader.__iter__().next()

vis.img("11", (d1[0][0][0] * 0.255 + 0.45).clamp(min=0, max=1))
vis.img("22", (d1[0][1][0] * 0.255 + 0.45).clamp(min=0, max=1))
print(d1[1])
