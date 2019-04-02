from torchvision.models import alexnet
from .basic_module import BasicModule, Flattern
from torch import nn
from torch.optim import Adam
import warnings


class AlexModel(BasicModule):
    def __init__(self, feature_d=256):
        super().__init__()
        self.model_name = "alexnet"
        self.model = alexnet(pretrained=True)
        self.extract = nn.Sequential(  # fune-tuning
            Flattern(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, feature_d)
        )
        self.model.classifier = nn.Linear(256, 2)

    def forward(self, *inx):
        if len(inx) == 1:
            warnings.warn("the input size not correct!!")
        img1_feature = self.model.features(inx[0])  # feature of img1
        img2_feature = self.model.features(inx[1])  # feature of img2

        img1_feature = self.extract(img1_feature)
        img2_feature = self.extract(img2_feature)

