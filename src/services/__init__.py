from . import _recognition
from . import _kd
from . import _evaluate

_service_factory = {
    'recognition': _recognition.init_module,
    'kd': _kd.init_module,
    'evaluate': _evaluate.init_module
}


def get_names():
    return list(_service_factory.keys())


def init_service(serviceType, serviceName, data_loader):

    if serviceType not in list(_service_factory.keys()):
        raise KeyError('Unknown Trainer: {}'.format(serviceType))

    return _service_factory[serviceType](serviceName, data_loader)