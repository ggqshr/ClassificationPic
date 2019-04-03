import torch as t
from util import PairDataSet, Visualizer
from allModels import AlexModel, BasicModule
from config import obj
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchnet import meter
import tqdm


def train(**kwargs):
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
    val_dataloader = DataLoader(train_data, obj.batch_size, num_workers=obj.num_worker)

    ceiterion = nn.BCEWithLogitsLoss()
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

            loss: t.Tensor = ceiterion(y_hat, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(y_hat.softmax(dim=0).detach(), target.detach())

            if (index + 1) % obj.print_freq == 0:
                vis.plot("loss", loss_meter.value()[0])

        model.save()

        val_cm, val_accuracy = val(model, val_dataloader)

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
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm.tqdm(enumerate(dataloader)):
        val_input = val_input.to(device)
        y_hat = model(val_input)
        confusion_matrix.add(y_hat.softmax(dim=0).detach().squeeze(), label.long())

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value.trace()) / (cm_value.sum())
    return confusion_matrix, accuracy


if __name__ == '__main__':
    import fire

    fire.Fire()
