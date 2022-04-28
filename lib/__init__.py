import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import init_backbone
from .protopnet.model import PPNet

_prototypical_model = {

    "protopnet": PPNet
}


def init_proto_model(manager, classes, backbone):
    """
        Create network with pretrained features and 1x1 convolutional layer

    """
    # Creating tree (backbone+add-on+prototree) architecture

    prototypical_model = backbone.prototypicalModel
    use_chkpt_opt = manager.settingsConfig.useCheckpointOptimizer

    features, trainable_param_names = init_backbone(backbone)

    model = _prototypical_model[prototypical_model](
        num_classes=len(classes), feature_net=features, args=manager.settingsConfig)

    if backbone.loadPath is not None:
        checkpoint = torch.load(backbone.loadPath)
        model = checkpoint['model']
        print("Loaded model from ", backbone.loadPath)

        if not use_chkpt_opt:
            checkpoint = None
    else:
        checkpoint = None

    if manager.common.mgpus:
        print("Multi-gpu setting")
        model = nn.DataParallel(model)

    if manager.common.cuda > 0:
        print("Using GPU")
        model.cuda()

    return model, checkpoint, trainable_param_names
