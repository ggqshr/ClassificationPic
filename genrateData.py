from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
from tqdm import trange
import random

"""
使用方法
python 文件名 trans_pic --file_dir="path/to/img" --out_dir="/path/to/store" 
"""


def trans_pic(file_dir, out_dir, max_iter=10, gen_test=False, test_scala=0.3, pic_size=(224, 224)):
    """

    :param file_dir: 原始图片的地址
    :param out_dir: 处理后的文件存储的地址
    :param max_iter: 增多多少倍
    :param pic_size: resize图片的大小
    :param gen_test: 去否同时生成测试集
    :param test_scala: 测试集的比例
    :return:
    """
    trans = T.Compose([
        T.Resize(pic_size),
        T.RandomApply(
            [
                T.RandomHorizontalFlip(0.8),  # 水平翻转
                T.RandomVerticalFlip(0.8),  # 垂直翻转
                T.RandomRotation(30)  # 随机旋转
            ]
        ),
        T.ColorJitter(brightness=0.4, contrast=0.4),  # 变换图像的颜色等
        T.ToTensor()
    ])

    if not os.path.exists(file_dir):
        print("the data dir not exists!")
        return None

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if os.path.isdir(file_dir):
        image_data = ImageFolder(file_dir, transform=trans)  # 如果是文件夹
        class_dict = {v: k for k, v in image_data.class_to_idx.items()}
        for i in trange(max_iter):
            target_dir = os.path.join(out_dir, "totrain")
            if gen_test and i + 1 > max_iter * (1 - test_scala):  # 生成测试集
                target_dir = os.path.join(out_dir, "totest")

            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            for d, l in image_data:
                target_d = os.path.join(target_dir, str(class_dict[l]))
                if not os.path.exists(target_d):
                    os.mkdir(target_d)  # 如果目标文件夹不存在，就创建
                file_name = "resize" + str(i) + str(random.randint(1, max_iter * 10)) + ".jpg"
                file = os.path.join(target_d, file_name)
                save_image(d, file)

    else:
        image_data = Image.open(file_dir)
        for i in trange(max_iter):
            porcess_pic = trans(image_data)  # 将图片变换
            file_name = "resize" + str(random.randint(1, max_iter * 10)) + ".jpg"
            file = os.path.join(out_dir, file_name)
            save_image(porcess_pic, file)


if __name__ == '__main__':
    import fire

    fire.Fire()
