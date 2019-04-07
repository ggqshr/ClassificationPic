import torch as t
import torchvision.transforms as T
from util import PairDataSet, Visualizer
from allModels import AlexModel, BasicModule, ClassifyModel
from config import obj
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn
from torch import optim
from torchnet import meter
import tqdm
from PIL import Image


def contrastiveLoss(y, D, m=0.4):
    N = D.shape[0]
    return (1. / 2 * N) * (y * D ** 2 + (1 - y) * t.max(m - D, t.zeros_like(D)) ** 2).sum(dim=0)


def trainExtractFeature(**kwargs):
    obj._update_para(kwargs)
    vis = Visualizer(obj.env, port=obj.vis_port)

    train_data = PairDataSet(obj.train_data_path)
    val_data = PairDataSet(obj.test_data_path)

    device = t.device("cuda") if obj.use_gpu else t.device("cpu")

    model = AlexModel()
    if obj.load_model_path:
        model.load(obj.load_model_path)
    model.to(device)

    train_dataloader = DataLoader(train_data, obj.batch_size, shuffle=True, num_workers=obj.num_worker)
    val_dataloader = DataLoader(val_data, obj.batch_size, num_workers=obj.num_worker)

    ceiterion = contrastiveLoss  # nn.BCEWithLogitsLoss()
    lr = obj.lr

    for parm in model.model.features.parameters():  # type:t.Tensor
        parm.requires_grad = False
    optimizer = optim.Adam(filter(lambda x: x.requires_grad is True, model.parameters()), lr=lr,
                           weight_decay=obj.weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    previous_loss = 1e10

    for epoch in tqdm.trange(obj.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for index, (data, label) in tqdm.tqdm(enumerate(train_dataloader)):  # type:int,(t.Tensor,t.Tensor)
            input = data.to(device)
            target = label.float().to(device)

            optimizer.zero_grad()
            y_hat = nn.parallel.data_parallel(model, input, [0, 1])

            loss: t.Tensor = ceiterion(target, y_hat, m=0.75)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            # confusion_matrix.add(y_hat.softmax(dim=0).detach(), target.detach())

            if (index + 1) % obj.print_freq == 0:
                vis.plot("loss", loss_meter.value()[0])
                vis.log("loss:{loss}".format(loss=loss.item()))

        model.save("No")

        val_cm, val_accuracy = val(model, val_dataloader)

        vis.plot("val_accuracy", val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #     epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
        #     lr=lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr * obj.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # if epoch == 3:
        #     lr = 1e-4
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #
        # if epoch == 6:
        #     lr = 1e-5
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


def trainClassify(**kwargs):
    obj._update_para(kwargs)
    vis = Visualizer(obj.env, port=obj.vis_port)

    train_data = PairDataSet(obj.train_data_path)
    val_data = PairDataSet(obj.test_data_path)

    device = t.device("cuda") if obj.use_gpu else t.device("cpu")

    model = ClassifyModel()
    if obj.load_model_path is not None:
        model.load(obj.load_model_path)
    model.to(device)

    train_dataloader = DataLoader(train_data, obj.batch_size, shuffle=True, num_workers=obj.num_worker)
    val_dataloader = DataLoader(val_data, obj.batch_size, num_workers=obj.num_worker)

    ceiterion = nn.BCEWithLogitsLoss()  # nn.BCEWithLogitsLoss()
    lr = obj.lr

    optimizer = optim.Adam(model.classify.parameters(), lr=lr,
                           weight_decay=obj.weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    previous_loss = 1e10
    sigmod = nn.Sigmoid()
    for epoch in tqdm.trange(obj.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for index, (data, label) in tqdm.tqdm(enumerate(train_dataloader)):  # type:int,(t.Tensor,t.Tensor)
            input = data.to(device)
            target = label.float().to(device)

            optimizer.zero_grad()
            y_hat = nn.parallel.data_parallel(model, input, [0, 1])

            loss: t.Tensor = ceiterion(y_hat, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            confusion_matrix.add(sigmod(y_hat.detach()).ge(0.5), target.detach())

            if (index + 1) % obj.print_freq == 0:
                vis.plot("loss", loss_meter.value()[0])
                # vis.log("loss:{loss}".format(loss=loss.item()))

        val_cm, val_accuracy = valCLassify(model, val_dataloader)
        model.save(val_accuracy)
        vis.plot("train_accuracy", 100 * (confusion_matrix.value().trace()) / (confusion_matrix.value().sum()))
        vis.plot("val_accuracy", val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr * obj.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # if epoch == 2:
        #     lr = 5e-4
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # if epoch == 6:
        #     lr = 1e-5
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, dataloader):
    """
    :type model:BasicModule
    :param model:
    :param dataloader:
    :return:
    """
    device = t.device("cuda") if obj.use_gpu else t.device("cpu")
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm.tqdm(enumerate(dataloader)):
        val_input = val_input.to(device)
        y_hat = model(val_input)
        # confusion_matrix.add(y_hat.softmax(dim=0).detach().squeeze(), label.long())

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value.trace()) / (cm_value.sum())
    return confusion_matrix, accuracy


@t.no_grad()
def valCLassify(model, dataloader):
    """
    :type model:BasicModule
    :param model:
    :param dataloader:
    :return:
    """
    sigmod = nn.Sigmoid()
    device = t.device("cuda") if obj.use_gpu else t.device("cpu")
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm.tqdm(enumerate(dataloader)):
        val_input = val_input.to(device)
        y_hat = model(val_input)
        confusion_matrix.add(sigmod(y_hat.detach()).ge(0.5), label.long())

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value.trace()) / (cm_value.sum())
    return confusion_matrix, accuracy


@t.no_grad()
def test(**kwargs):
    obj._update_para(kwargs)
    model = AlexModel()
    if obj.load_model_path is None:
        print("no model path point")
        return
    model.load(obj.load_model_path)
    model.to(t.device("cuda"))
    model.eval()

    train_data = PairDataSet(obj.test_data_path)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=1)
    for d, l in train_data_loader:
        dis = model(d.cuda())
        print("the label is {label} the dis is {dis}".format(label=l, dis=dis))


def classify_img(test_img_path, **kwargs):
    """
    使用需要传入一个模型的加载地址
    :param kwargs:
    :return:
    """
    vis = Visualizer("classify", port=obj.vis_port)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    sigmod = nn.Sigmoid()
    obj._update_para(kwargs)
    model = ClassifyModel()
    model.load(obj.load_model_path)
    model.eval()

    class_data = ImageFolder(obj.class_data_path, transform=transform)
    inx2class = {v: k for k, v in class_data.class_to_idx.items()}
    class_img_data, class_img_label = DataLoader(class_data, batch_size=len(class_data.classes)).__iter__().next()

    test_img_data = transform(Image.open(test_img_path))

    expand_test_img_data = test_img_data.expand_as(class_img_data)

    target = model(t.stack((expand_test_img_data, class_img_data), dim=1))

    classes_hat = sigmod(target).argmax()

    vis.img(inx2class.get(class_data.samples[classes_hat.item()][1]), test_img_data)

    print("the image name is {name},and the predict label is {label}".format(name=test_img_path, label=inx2class.get(
        class_data.samples[classes_hat.item()][1])))


def get_dis_between(img1, img2, **kwargs):
    obj._update_para(kwargs)
    model = AlexModel()
    model.load(obj.load_model_path)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    img1_data = transform(Image.open(img1))
    img2_data = transform(Image.open(img2))

    dis = model(t.stack((img1_data, img2_data)).unsqueeze(0))

    print("the dis between is {dis}".format(dis=dis))


def get_calss_use_dis(img1, **kwargs):
    obj._update_para(kwargs)
    vis = Visualizer("classify", port=obj.vis_port)

    model = AlexModel()
    model.load(obj.load_model_path)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    class_data = ImageFolder(obj.class_data_path, transform=transform)
    inx2class = {v: k for k, v in class_data.class_to_idx.items()}
    class_img_data, class_img_label = DataLoader(class_data, batch_size=len(class_data.classes)).__iter__().next()

    test_img_data = transform(Image.open(img1))

    expand_test_img_data = test_img_data.expand_as(class_img_data)
    dis = model(t.stack((expand_test_img_data, class_img_data), dim=1))

    print("the dis between is {dis}".format(dis=dis))

    class_hat = class_data.samples[dis.argmin().item()][1]
    print("the class is {classes}".format(classes=inx2class.get(class_hat)))
    vis.img(inx2class.get(class_hat), test_img_data)


if __name__ == '__main__':
    import fire

    fire.Fire()
