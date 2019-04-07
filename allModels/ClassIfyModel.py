import torch as t
import torch.nn as nn
from .basic_module import BasicModule
from .AlexModel import AlexModel


class ClassifyModel(BasicModule):
    def __init__(self, path="./checkpoints/squeezenet_###No###0407_135347.pth"):
        super().__init__()
        self.model_name = "ClassifyModel"
        self.feature = AlexModel()
        self.feature.load(path)
        self.feature.eval()

        self.classify = nn.Linear(128, 1)

    def forward(self, x):
        img1 = x[[i for i in range(x.shape[0])], 0]
        img2 = x[[i for i in range(x.shape[0])], 1]

        img1_feature = self.feature(img1)  # feature of img1
        img2_feature = self.feature(img2)  # feature of img2

        temp = ((img1_feature - img2_feature) ** 2) / (img1_feature + img2_feature + 4e-5)  # 加上0.4 防止两个特征全部为0的情况

        final_feature: t.Tensor = self.classify(temp.view(temp.shape[0], -1))

        return final_feature.view(final_feature.shape[0])
