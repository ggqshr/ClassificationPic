import torchvision.transforms as T
import torch as t
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils import data
import random
import os

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


class PairDataSet(data.Dataset):
    def __init__(self, path2img):
        transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),  # 随机翻转
            T.ToTensor(),  # 转换成向量
            normalize
        ])
        self.path = path2img
        self.dataset = ImageFolder(self.path, transform=transform)
        inx2class = {v: k for k, v in self.dataset.class_to_idx.items()}
        self.final_data_set = []
        sample_index = 0
        for anchor_data, anchor_label in self.dataset:
            positive = self.generate_positive_class_pic(sample_index)
            self.final_data_set.append(((anchor_data, transform(positive)), 1))

            this_classes = inx2class.get(anchor_label)
            nagetive = self.generate_nagetive_class_pic(path2img, this_classes)
            self.final_data_set.append(((anchor_data, transform(nagetive)), 0))
            sample_index += 1

    def generate_nagetive_class_pic(self, path2img, this_classes):
        class_list = self.dataset.classes.copy()
        class_list.remove(this_classes)

        random_classes = class_list[random.randint(0, len(class_list) - 1)]

        different_classes_img_list = os.walk(os.path.join(path2img, random_classes)).__next__()[2]

        nagetive = different_classes_img_list[random.randint(0, len(different_classes_img_list) - 1)]
        nagetive = Image.open(os.path.join(path2img, random_classes, nagetive))

        return nagetive

    def generate_positive_class_pic(self, sample_index):
        this_image_dir_and_filename = os.path.split(self.dataset.samples[sample_index][0])

        this_image_dir = this_image_dir_and_filename[0]  # 获得当前文件的目录
        this_image_file_name = this_image_dir_and_filename[1]  # 获得当前文件的文件名

        same_class_img_list = os.walk(this_image_dir).__next__()[2]  # 获得相同类别的图片的列表

        random_index = random.randint(0, len(same_class_img_list) - 1)
        positive = same_class_img_list[random_index]  # 随机选取一个相同类别的

        while positive == this_image_file_name:  # 如果选取的是相同的图片，就在随机一个
            positive = same_class_img_list[(random_index + 1) % (len(same_class_img_list) - 1)]
        positive = Image.open(os.path.join(this_image_dir, positive))

        return positive

    def __getitem__(self, item):
        this_data = self.final_data_set[item]
        return this_data[0], this_data[1]

    def __len__(self):
        return self.final_data_set.__len__()
