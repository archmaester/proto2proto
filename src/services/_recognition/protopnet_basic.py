import os
import math
import torch
from tqdm import tqdm
from src.mgr import manager
from src.utils.dirs import create_dirs
import torch.nn.functional as F
from lib import init_proto_model
from lib.protopnet.losses import cluster_sep_loss_fn, l1_loss_fn
from lib.protopnet.optimizer import get_optimizer, last_only, warm_only, joint
from lib.protopnet import push, save, preprocess

from lib.utils import evaluate
from src.utils import _common


class Trainer(object):

    def __init__(self, dataset_loader):

        self.manager = manager
        self.mgpus = self.manager.common.mgpus
        self.dataset_loader = dataset_loader

        self.model, checkpoint, self.trainable_param_names = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.backbone)

        self.start_epoch = 1
        self.initialize_training(checkpoint)
        self.createLogger()
        self.model_dir = os.path.join(self.manager.base_dir, "models")
        create_dirs(self.model_dir)

    def createLogger(self):
        return _common.createLogger(self)

    def save_model(self, epoch, append=None):
        save.save_basic(self, epoch, self.model_dir, append=append)

    def evaluate(self, epoch, log=True):

        result = evaluate.evaluate_model(self.model, self.dataset_loader.test_loader,
                                         mgpus=self.mgpus)

        if manager.settingsConfig.train.useTensorboard:
            self.logger.add_scalars("test", result, epoch)

        return result

    def initialize_training(self, checkpoint):

        self.joint_optimizer, self.last_optimizer, self.warm_optimizer = \
            get_optimizer(self.model, manager.settingsConfig, mgpus=self.mgpus)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.joint_optimizer, step_size=manager.settingsConfig.train.stepSize,
            gamma=manager.settingsConfig.train.gamma)

        if checkpoint is not None:
            self.start_epoch = checkpoint["epoch"]
            self.joint_optimizer.load_state_dict(checkpoint["joint_optimizer"])
            self.last_optimizer.load_state_dict(checkpoint["last_optimizer"])
            self.warm_optimizer.load_state_dict(checkpoint["warm_optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def __call__(self):

        max_epochs = self.manager.settingsConfig.train.maxEpochs
        warm_epochs = self.manager.settingsConfig.train.warmEpochs
        step_start = self.manager.settingsConfig.train.stepStart
        self.iters_per_epoch = len(self.dataset_loader.train_loader)

        push_start = self.manager.settingsConfig.train.pushStart
        push_epochs = [ii for ii in range(push_start, max_epochs, 10)]

        if self.mgpus:
            # Optimize class distributions in leafs
            self.eye = torch.eye(self.model.module._num_classes)
        else:
            self.eye = torch.eye(self.model._num_classes)

        for epoch in tqdm(range(self.start_epoch, max_epochs + 1)):

            if epoch <= warm_epochs:
                warm_only(self.model, self.trainable_param_names, self.mgpus)
                self.train_epoch(epoch, self.warm_optimizer)
            else:
                joint(self.model, self.trainable_param_names, self.mgpus)
                self.train_epoch(epoch, self.joint_optimizer)
                if epoch >= step_start:
                    self.scheduler.step()

            if epoch % manager.settingsConfig.train.saveEpoch == 0:
                self.save_model(epoch)

            if epoch % self.manager.settingsConfig.train.evalEpoch == 0:
                result = self.evaluate(epoch)
                print(epoch, result)

            if epoch in push_epochs:
                self.push(epoch)
                self.save_model(epoch, append="push")
                print("Epoch", "Push", self.evaluate(epoch, log=False))
                last_only(self.model, self.trainable_param_names, self.mgpus)
                for ii in tqdm(range(10)):
                    self.train_epoch(epoch, self.last_optimizer, log=False)
                    print("Iteration: ", ii, " Epoch", " Push-tune", self.evaluate(epoch, log=False))

                self.save_model(epoch, append="push_tuned")
                print("Epoch", "Push tuned", self.evaluate(epoch, log=False))

    def train_epoch(self, epoch, optimizer, log=True):

        data_iter = iter(self.dataset_loader.train_loader)

        self.model.train()
        for step in tqdm(range(self.iters_per_epoch), leave=False):
            save_step = (epoch - 1) * self.iters_per_epoch + step
            data = next(data_iter)
            self.train_step(data, save_step, optimizer, log=log)

    def train_step(self, data, step, optimizer, log=True):

        scalar_dict = {}
        self.model.train()
        self.model.zero_grad()

        xs, ys = data
        ys = ys.cuda()
        xs = xs.cuda()
        ys_pred, info = self.model.forward(xs)

        loss_list = self.manager.settingsConfig.lossList
        total_loss = torch.tensor(0).cuda().float()

        if loss_list.crossEntropy.consider:
            loss_ce = torch.nn.functional.cross_entropy(ys_pred, ys)
            loss_ce = loss_list.crossEntropy.weight*loss_ce
            total_loss += loss_ce
            scalar_dict["loss_ce"] = loss_ce.item()

        if loss_list.clusterSep.consider:
            loss_cluster, loss_sep = cluster_sep_loss_fn(self.model, info[0], ys, self.mgpus)
            loss_cluster = loss_list.clusterSep.clusterWeight*loss_cluster
            loss_sep = loss_list.clusterSep.sepWeight*loss_sep
            total_loss += loss_cluster
            total_loss += loss_sep
            scalar_dict["loss_cluster"] = loss_cluster.item()
            scalar_dict["loss_sep"] = loss_sep.item()

        if loss_list.l1.consider:
            loss_l1 = l1_loss_fn(self.model, self.mgpus)
            loss_l1 = loss_list.l1.weight*loss_l1
            total_loss += loss_l1
            scalar_dict["loss_l1"] = loss_l1.item()

        optimizer.zero_grad()

        # Optimize
        total_loss.backward()

        # Update model parameters
        optimizer.step()

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred_max, ys))
        scalar_dict["accuracy"] = correct.item() / float(len(xs))
        scalar_dict["loss"] = total_loss.item()

        # Visualize
        if manager.settingsConfig.train.useTensorboard and log:
            self.logger.add_scalars("train", scalar_dict, step)

    def push(self, epoch):

        push.push_prototypes(
            self.dataset_loader.project_loader,
            prototype_network_parallel=self.model, # pytorch network with prototype_vectors
            class_specific=True,
            preprocess_input_function=preprocess.preprocess_input_function,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix='prototype-img',
            prototype_self_act_filename_prefix='prototype-self-act',
            proto_bound_boxes_filename_prefix='bb',
            save_prototype_class_identity=True,
            log=print)
