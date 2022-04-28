import os
import torch


def save_basic(self, epoch, save_dir, append=None):
    '''
    '''
    if append is None:
        save_name = os.path.join(
            save_dir, 'protopnet_{}.pth'.format(epoch))
    else:
        save_name = os.path.join(
            save_dir, 'protopnet_{}_{}.pth'.format(epoch, append))

    torch.save({
        'epoch': epoch,
        'model': self.model.module if self.manager.common.mgpus else self.model,
        'warm_optimizer': self.warm_optimizer.state_dict(),
        'joint_optimizer': self.joint_optimizer.state_dict(),
        'last_optimizer': self.last_optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict()
    }, save_name)

    print('save model: {}'.format(save_name))


def save_pshare(self, epoch, save_dir, append=None):
    '''
    '''
    if append is None:
        save_name = os.path.join(
            save_dir, 'protopnet_{}.pth'.format(epoch))
    else:
        save_name = os.path.join(
            save_dir, 'protopnet_{}_{}.pth'.format(epoch, append))

    torch.save({
        'epoch': epoch,
        'model': self.model.module if self.manager.common.mgpus else self.model,
        'warm_optimizer': self.warm_optimizer,
        'joint_optimizer': self.joint_optimizer,
        'last_optimizer': self.last_optimizer,
        'scheduler': self.scheduler.state_dict()
    }, save_name)

    print('save model: {}'.format(save_name))


def save_kd(self, epoch, save_dir, append=None):
    '''
    '''
    if append is None:
        save_name = os.path.join(
            save_dir, 'protopnet_{}.pth'.format(epoch))
    else:
        save_name = os.path.join(
            save_dir, 'protopnet_{}_{}.pth'.format(epoch, append))

    torch.save({
        'epoch': epoch,
        'model': self.model.module if self.manager.common.mgpus else self.model,
        'warm_optimizer': self.warm_optimizer.state_dict(),
        'joint_optimizer': self.joint_optimizer.state_dict(),
        'joint_frozen_optimizer': self.joint_frozen_optimizer.state_dict(),
        'last_optimizer': self.last_optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'last_scheduler': self.last_scheduler.state_dict()
    }, save_name)

    print('save model: {}'.format(save_name))
