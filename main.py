import torch as t
from util import PairDataSet, Visualizer
from models import AlexModel
from config import obj


def train(**kwargs):
    obj._update_para(kwargs)

    train_data = PairDataSet(obj.train_data_path)
    val_data = PairDataSet(obj.test_data_path)

    device = t.device("cuda") if obj.use_gpu else t.device("cpu")
