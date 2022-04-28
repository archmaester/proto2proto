from . import protopnet_evaluate

_factory = {
    'protopnet_evaluate': protopnet_evaluate.Service
}


def get_names():
    return list(_factory.keys())


def init_module(factory_type, data_loader):

    if factory_type not in list(_factory.keys()):
        raise KeyError('Unknown Service: {}'.format(factory_type))

    return _factory[factory_type](data_loader)