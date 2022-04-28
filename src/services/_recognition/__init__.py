from . import protopnet_basic

_factory = {
    'protopnet_basic': protopnet_basic.Trainer
}


def get_names():
    return list(_factory.keys())


def init_module(factory_type, data_loader):

    if factory_type not in list(_factory.keys()):
        raise KeyError('Unknown Trainer: {}'.format(factory_type))

    return _factory[factory_type](data_loader)