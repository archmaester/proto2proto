from . import protopnet_kd


_factory = {
    'protopnet_kd': protopnet_kd.Trainer
}


def get_names():
    return list(_factory.keys())


def init_module(factory_type, data_loader):

    if factory_type not in list(_factory.keys()):
        raise KeyError('Unknown Trainer: {}'.format(factory_type))

    return _factory[factory_type](data_loader)
