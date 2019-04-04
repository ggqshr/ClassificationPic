from torchvision.models import alexnet, vgg16, squeezenet1_1
from .basic_module import BasicModule, Flattern
from torch import nn
import warnings
import torch as t


class AlexModel(BasicModule):
    def __init__(self, feature_d=128):
        super().__init__()
        self.model_name = "squeezenet"
        self.model = squeezenet1_1(pretrained=True)
        self.extract = nn.Sequential(  # fune-tuning
            nn.Dropout(p=0.5),
            nn.Conv2d(512, feature_d, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )
        # self.model.classifier = nn.Linear(feature_d, 128)

    def forward(self, x: t.Tensor):
        if len(x.shape) == 5:
            img1 = x[[i for i in range(x.shape[0])], 0]
            img2 = x[[i for i in range(x.shape[0])], 1]
            img1_feature = self.model.features(img1)  # feature of img1
            img2_feature = self.model.features(img2)  # feature of img2

            img1_feature = self.extract(img1_feature)
            img2_feature = self.extract(img2_feature)

            temp = (img1_feature - img2_feature) ** 2
            # ((img1_feature - img2_feature) ** 2) / (img1_feature + img2_feature)

            # final_feature: t.Tensor = self.model.classifier(temp.view(temp.shape[0], -1))

            return temp.view(temp.shape[0], -1).sum(dim=1) ** 1. / 2
        elif len(x.shape) == 4:
            x = self.model.features(x)
            x = self.extract(x)
            return x.view(x.shape[0], -1)
