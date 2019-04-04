import torch as t
from util import PairDataSet, Visualizer
from allModels import AlexModel, BasicModule, ClassifyModel
from config import obj
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchnet import meter
import tqdm


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

            loss: t.Tensor = ceiterion(target, y_hat, m=0.6)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            # confusion_matrix.add(y_hat.softmax(dim=0).detach(), target.detach())

            if (index + 1) % obj.print_freq == 0:
                vis.plot("loss", loss_meter.value()[0])
                vis.log("loss:{loss}".format(loss=loss.item()))

        model.save()

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

        previous_loss = loss_meter.value()[0]


def trainClassify(**kwargs):
    obj._update_para(kwargs)
    vis = Visualizer(obj.env, port=obj.vis_port)

    train_data = PairDataSet(obj.train_data_path)
    val_data = PairDataSet(obj.test_data_path)

    device = t.device("cuda") if obj.use_gpu else t.device("cpu")

    model = ClassifyModel(obj.load_model_path)
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

        model.save()

        val_cm, val_accuracy = valCLassify(model, val_dataloader)

        vis.plot("val_accuracy", val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr * obj.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

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
    confusion_matrix = meter.ConfusionMeter(1)
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
    sigmod  = nn.Sigmoid()
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


if __name__ == '__main__':
    import fire

    fire.Fire()
