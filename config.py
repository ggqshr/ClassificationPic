import warnings


class ConfigObj(object):
    env = 'tt'
    vis_port = 8097

    train_data_path = "./seedPic/totrain"
    test_data_path = "./seedPic/totest"
    load_model_path = None
    class_data_path = "./classdata"

    batch_size = 8
    use_gpu = True
    num_worker = 4
    print_freq = 20

    max_epoch = 10
    lr = 1e-3
    weight_decay = 1e-4
    lr_decay = 0.5

    def _update_para(self, kwargs) -> None:
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():  # type:str,str
            if not k.startswith("_"):
                print(k, getattr(self, k))


obj = ConfigObj()
