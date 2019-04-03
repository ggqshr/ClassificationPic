from torchvision.models import alexnet, vgg16
from .basic_module import BasicModule, Flattern
from torch import nn
import warnings
import torch as t


class AlexModel(BasicModule):
    def __init__(self, feature_d=256):
        super().__init__()
        self.model_name = "alexnet"
        self.model = vgg16(pretrained=True)
        self.extract = nn.Sequential(  # fune-tuning
            Flattern(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, feature_d),
        )
        self.model.classifier = nn.Linear(feature_d, 128)

    def forward(self, x: t.Tensor):
        img1 = x[[i for i in range(x.shape[0])], 0]
        img2 = x[[i for i in range(x.shape[0])], 1]
        img1_feature = self.model.features(img1)  # feature of img1
        img2_feature = self.model.features(img2)  # feature of img2

        img1_feature = self.extract(img1_feature)
        img2_feature = self.extract(img2_feature)

        temp = ((img1_feature - img2_feature) ** 2) / (img1_feature + img2_feature)

        final_feature: t.Tensor = self.model.classifier(temp)

        return final_feature.sum(dim=1)


