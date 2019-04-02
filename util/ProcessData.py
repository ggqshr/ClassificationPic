import torchvision.transforms as T
import torch as t
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils import data
import random

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


class PairDataSet(data.Dataset):
    def __init__(self, path2img):
        transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),  # 随机翻转
            T.ColorJitter(brightness=0.4, hue=0.3),  # 变化图片的色彩等
            T.ToTensor(),  # 转换成向量
            normalize
        ])
        self.path = path2img
        self.dataset = ImageFolder(self.path, transform=transform)
        inx2class = {v: k for k, v in self.dataset.class_to_idx.items()}
        self.final_data_set = []
        for anchor_data, anchor_label in self.dataset:
            temp = filter(lambda k: k[1] == anchor_label and not t.equal(k[0], anchor_data), self.dataset)
            temp = data.DataLoader([i for i in temp], 1, True)
            positive = temp.__iter__().next()
            self.final_data_set.append(((anchor_data, positive[0]), 1))
            temp = filter(lambda k: k[1] != anchor_label, self.dataset)
            temp = data.DataLoader([i for i in temp], 1, True)
            negative = temp.__iter__().next()
            self.final_data_set.append(((anchor_data, negative[0]), 0))

    def __getitem__(self, item):
        this_data = self.final_data_set[item]
        return this_data[0], this_data[1]

    def __len__(self):
        return self.final_data_set.__len__()
