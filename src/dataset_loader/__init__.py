from . import cars
from . import cub

__dataset_factory = {
    'cars': cars.DatasetLoader,
    'cub': cars.DatasetLoader,
}


def get_names():
    return list(__dataset_factory.keys())


def init_dataset(dataset_type):

    if dataset_type not in list(__dataset_factory.keys()):
        raise KeyError('Unknown Dataset Loader: {}'.format(dataset_type))

    return __dataset_factory[dataset_type]()
