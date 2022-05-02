import os
import math
import torch
from tqdm import tqdm
from src.mgr import manager
import torch.nn.functional as F
from lib import init_proto_model
from src.utils.dirs import create_dirs
from lib.protopnet.losses import cluster_sep_loss_fn, l1_loss_fn
from lib.protopnet.optimizer_kd import get_optimizer, last_only, warm_only, joint_head_frozen, joint
from lib.protopnet import push, save, preprocess
from src.utils import _common
from src.utils.losses import featureLoss_mask
import torch.nn as nn

from lib.utils import evaluate

class Trainer(object):

    def __init__(self, dataset_loader):

        self.manager = manager
        self.mgpus = self.manager.common.mgpus
        self.dataset_loader = dataset_loader

        self.teacher_model, _, _ = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.backbone)

        self.model, checkpoint, self.trainable_param_names = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.target)

        if checkpoint is None:
            self.model.stu_feature_adap = nn.Sequential(nn.Conv2d(512, 2048,
                                                       kernel_size=1, padding=0), nn.ReLU()).cuda()

        self.model_dir = os.path.join(self.manager.base_dir, "models")
        create_dirs(self.model_dir)

        self.start_epoch = 1
        self.initialize_training(checkpoint)
        self.createLogger()

    def createLogger(self):
        return _common.createLogger(self)

    def evaluate(self, epoch, log=True):

        result = evaluate.evaluate_model(self.model, self.dataset_loader.test_loader,
                                         mgpus=self.mgpus)

        if manager.settingsConfig.train.useTensorboard:
            self.logger.add_scalars("test", result, epoch)

        return result

    def save_model(self, epoch, append=None):
        save.save_kd(self, epoch, self.model_dir, append=append)

    def initialize_training(self, checkpoint):

        self.warm_optimizer, self.joint_frozen_optimizer, self.joint_optimizer = \
            get_optimizer(self.model, manager.settingsConfig, mgpus=self.mgpus)

        last_paramlist = [
            {'params': self.model.module.last_layer.parameters(), 'lr': manager.settingsConfig.train.lrLastLayer}
        ]
        self.last_optimizer = torch.optim.Adam(last_paramlist)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.joint_frozen_optimizer, step_size=manager.settingsConfig.train.stepSize,
            gamma=manager.settingsConfig.train.gamma)

        self.last_scheduler = torch.optim.lr_scheduler.StepLR(
            self.joint_optimizer, step_size=manager.settingsConfig.train.stepSize,
            gamma=manager.settingsConfig.train.gamma)

        if checkpoint is not None:
            self.start_epoch = checkpoint["epoch"]
            self.warm_optimizer.load_state_dict(checkpoint["warm_optimizer"])
            self.joint_frozen_optimizer.load_state_dict(checkpoint["joint_frozen_optimizer"])
            self.joint_optimizer.load_state_dict(checkpoint["joint_optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.last_scheduler.load_state_dict(checkpoint["last_scheduler"])

        self.pdist = nn.PairwiseDistance(p=2)

    def __call__(self):

        max_epochs = self.manager.settingsConfig.train.maxEpochs
        warm_epochs = self.manager.settingsConfig.train.warmEpochs
        step_start = self.manager.settingsConfig.train.stepStart
        stage1_epochs = self.manager.settingsConfig.train.stage1Epochs
        self.iters_per_epoch = len(self.dataset_loader.train_loader)
        stage1_loss_list = self.manager.settingsConfig.stage1LossList
        stage2_loss_list = self.manager.settingsConfig.stage2LossList
        self.tau_train = self.manager.settingsConfig.tauTrain

        push_start = self.manager.settingsConfig.train.pushStart
        push_epochs = [ii for ii in range(push_start, max_epochs, 10)]

        if self.mgpus:
            # Optimize class distributions in leafs
            self.eye = torch.eye(self.model.module._num_classes)
            self.model.module.last_layer.weight.data.copy_(self.teacher_model.module.last_layer.weight.data)
        else:
            self.eye = torch.eye(self.model._num_classes)
            self.model.last_layer.weight.data.copy_(self.teacher_model.last_layer.weight.data)

        for epoch in tqdm(range(self.start_epoch, max_epochs + 1)):

            if epoch <= warm_epochs:
                warm_only(self.model, self.trainable_param_names, self.mgpus)
                self.train_epoch(epoch, self.warm_optimizer, loss_list=stage1_loss_list)
            elif epoch <= stage1_epochs:
                joint_head_frozen(self.model, self.trainable_param_names, self.mgpus)
                self.train_epoch(epoch, self.joint_frozen_optimizer, loss_list=stage1_loss_list)
                if epoch >= step_start:
                    self.scheduler.step()
            else:
                joint(self.model, self.trainable_param_names, self.mgpus)
                self.train_epoch(epoch, self.joint_optimizer, loss_list=stage2_loss_list)
                if epoch >= step_start:
                    self.last_scheduler.step()

            if epoch % manager.settingsConfig.train.saveEpoch == 0:
                self.save_model(epoch)

            if epoch % self.manager.settingsConfig.train.evalEpoch == 0:
                result = self.evaluate(epoch)
                print(epoch, result)

            if epoch in push_epochs:
                self.evaluate(epoch)
                self.push(epoch)
                self.save_model(epoch, append="push")
                print("Epoch", "Push", self.evaluate(epoch, log=False))
                last_only(self.model, self.mgpus)
                for ii in tqdm(range(10)):
                    self.train_epoch(epoch, self.last_optimizer, log=False, loss_list=stage2_loss_list)
                self.save_model(epoch, append="push_tuned")
                print("Epoch", "Push tuned", self.evaluate(epoch, log=False))

    def train_epoch(self, epoch, optimizer, log=True, loss_list=[]):

        data_iter = iter(self.dataset_loader.train_loader)

        self.model.train()
        for step in tqdm(range(self.iters_per_epoch), leave=False):
            save_step = (epoch - 1) * self.iters_per_epoch + step
            data = next(data_iter)
            self.train_step(data, save_step, optimizer, log=log, loss_list=loss_list)

    def train_step(self, data, step, optimizer, log=True, loss_list=[]):

        scalar_dict = {}
        self.teacher_model.eval()
        self.model.train()
        self.model.zero_grad()

        xs, ys = data
        ys = ys.cuda()
        xs = xs.cuda()

        ys_pred, info = self.model.forward(xs)
        ys_pred_tchr, info_tchr = self.teacher_model.forward(xs)

        #Getting teacher/student distances:
        teacher_distances = info[3]

        if len(loss_list) == 0:
            loss_list = self.manager.settingsConfig.lossList

        total_loss = torch.tensor(0).cuda().float()

        if hasattr(loss_list, "addOnLoss") and loss_list.addOnLoss.consider:
            mask = self.get_vectorized_mask(teacher_distances)
            loss_add_on = featureLoss_mask(mask, info[1], info_tchr[1]).mean()
            loss_add_on = loss_list.addOnLoss.weight*loss_add_on
            total_loss += loss_add_on
            scalar_dict["loss_add_on"] = loss_add_on.item()

        if hasattr(loss_list, "protoLoss") and loss_list.protoLoss.consider:

            if self.mgpus:
                input1 = self.model.module.prototype_vectors
                input2 = self.teacher_model.module.prototype_vectors
            else:
                input1 = self.model.prototype_vectors
                input2 = self.teacher_model.prototype_vectors

            loss_proto = self.pdist(input1, input2).mean()
            loss_proto = loss_list.protoLoss.weight*loss_proto
            total_loss += loss_proto
            scalar_dict["loss_proto"] = loss_proto.item()

        if hasattr(loss_list, "crossEntropy") and loss_list.crossEntropy.consider:
            loss_ce = torch.nn.functional.cross_entropy(ys_pred, ys)
            loss_ce = loss_list.crossEntropy.weight*loss_ce
            total_loss += loss_ce
            scalar_dict["loss_ce"] = loss_ce.item()

        if hasattr(loss_list, "clusterSep") and loss_list.clusterSep.consider:
            loss_cluster, loss_sep = cluster_sep_loss_fn(self.model, info[0], ys, self.mgpus)
            loss_cluster = loss_list.clusterSep.clusterWeight*loss_cluster
            loss_sep = loss_list.clusterSep.sepWeight*loss_sep
            total_loss += loss_cluster
            total_loss += loss_sep
            scalar_dict["loss_cluster"] = loss_cluster.item()
            scalar_dict["loss_sep"] = loss_sep.item()

        if hasattr(loss_list, "l1") and loss_list.l1.consider:
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
            self.logger.add_scalars("info", scalar_dict, step)

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

    def get_vectorized_mask(self, t_dist):
        #t_dist -> (batch, num_p, h, w)
        teacher_scores, _ = F.max_pool2d(-t_dist, kernel_size=(t_dist.size()[2], t_dist.size()[3]),
                                         return_indices=True)
        mask = (teacher_scores == -t_dist)
        mask = (t_dist < self.tau_train)*mask
        mask = torch.any(mask, dim = 1)
        return mask
