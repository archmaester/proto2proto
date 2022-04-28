import torch
import pprint
import numpy as np

from src.mgr import manager
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda


def _init_fn():
    np.random.seed(cfg.RNG_SEED)


class DatasetLoader(object):

    def __init__(self):
        '''
        '''

        img_size = manager.dataConfig.imgSize
        train_dir = manager.dataConfig.trainDir
        project_dir = manager.dataConfig.projectDir
        test_dir = manager.dataConfig.testDir
        augment = manager.dataConfig.augment

        self.shape = (3, img_size, img_size)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean=mean,std=std)
        transform_no_augment = transforms.Compose([
                                transforms.Resize(size=(img_size, img_size)),
                                transforms.ToTensor(),
                                normalize
                            ])

        transform_push = transforms.Compose([
                                transforms.Resize(size=(img_size, img_size)),
                                transforms.ToTensor()])

        if augment:
            transform = transforms.Compose([
                transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
                transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
                transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15,shear=(-2,2)),
                ]),
                transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transform_no_augment

        train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform)
        project_set = torchvision.datasets.ImageFolder(project_dir, transform=transform_push)
        test_set = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
        self.classes = train_set.classes

        cuda = manager.common.cuda and torch.cuda.is_available()

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=manager.common.trainBatchSize,
                                                  shuffle=True, num_workers=manager.common.trainNumWorkers,
                                                  pin_memory=cuda
                                                  )
        self.project_loader = torch.utils.data.DataLoader(project_set,
                                                  batch_size=int(manager.common.trainBatchSize // 4), 
                                                  shuffle=False, num_workers=manager.common.trainNumWorkers // 4,
                                                  pin_memory=cuda
                                                  )
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=manager.common.testBatchSize,
                                                 shuffle=False, num_workers=manager.common.trainNumWorkers,
                                                 pin_memory=cuda
                                                 )
        print("Num classes (k) = ", len(self.classes), flush=True)
