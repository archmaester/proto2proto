import torch


def get_optimizer(ppnet, args, mgpus=False):

    if mgpus:

        warm_paramlist = [
            {'params': ppnet.module._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module.prototype_vectors, 'lr': args.train.lrProto}
        ]
        joint_frozen_paramlist = [
            {'params': ppnet.module.features.parameters(), 'lr': args.train.lrNet,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module.prototype_vectors, 'lr': args.train.lrProto}
        ]
        joint_paramlist = [
            {'params': ppnet.module.features.parameters(), 'lr': args.train.lrNet,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module.prototype_vectors, 'lr': args.train.lrProto},
            {'params': ppnet.module.last_layer.parameters(), 'lr': args.train.lrLastLayer}
        ]
#         last_paramlist = [
#             {'params': ppnet.module.last_layer.parameters(), 'lr': args.train.lrLastLayer}
#         ]

    else:
        warm_paramlist = [
            {'params': ppnet._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.prototype_vectors, 'lr': args.train.lrProto}
        ]
        joint_frozen_paramlist = [
            {'params': ppnet.features.parameters(), 'lr': args.train.lrNet,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.prototype_vectors, 'lr': args.train.lrProto}
        ]
        joint_paramlist = [
            {'params': ppnet.features.parameters(), 'lr': args.train.lrNet,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.prototype_vectors, 'lr': args.train.lrProto},
            {'params': ppnet.last_layer.parameters(), 'lr': args.train.lrLastLayer}
        ]

#         last_paramlist = [
#             {'params': ppnet.last_layer.parameters(), 'lr': args.train.lrLastLayer}
#         ]

    return torch.optim.Adam(warm_paramlist), torch.optim.Adam(joint_frozen_paramlist), torch.optim.Adam(joint_paramlist)


def last_only(model, mgpus=False):

    if mgpus:
        for p in model.module.features.parameters():
            p.requires_grad = False
        for p in model.module._add_on.parameters():
            p.requires_grad = False
        for p in model.module.last_layer.parameters():
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = False
    else:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model._add_on.parameters():
            p.requires_grad = False
        for p in model.last_layer.parameters():
            p.requires_grad = True
        model.prototype_vectors.requires_grad = False        


def warm_only(model, trainable_param_names, mgpus=False):

    if mgpus:
        for name, p in model.module.features.named_parameters():
            if name in trainable_param_names:
                p.requires_grad = False
        for p in model.module._add_on.parameters():
            p.requires_grad = True
        for p in model.module.last_layer.parameters():
            p.requires_grad = False
        model.module.prototype_vectors.requires_grad = True
    else:
        for name, p in model.features.named_parameters():
            if name in trainable_param_names:
                p.requires_grad = False
        for p in model._add_on.parameters():
            p.requires_grad = True
        for p in model.last_layer.parameters():
            p.requires_grad = False
        model.prototype_vectors.requires_grad = True   


def joint(model, trainable_param_names, mgpus=False):

    if mgpus:
        for name, p in model.module.features.named_parameters():
#             if name in trainable_param_names:
            p.requires_grad = True
        for p in model.module._add_on.parameters():
            p.requires_grad = True
        for p in model.module.last_layer.parameters():
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = True
    else:
        for name, p in model.features.named_parameters():
#             if name in trainable_param_names:
            p.requires_grad = True
        for p in model._add_on.parameters():
            p.requires_grad = True
        for p in model.last_layer.parameters():
            p.requires_grad = True
        model.prototype_vectors.requires_grad = True


def joint_head_frozen(model, trainable_param_names, mgpus=False):

    if mgpus:
        for name, p in model.module.features.named_parameters():
#             if name in trainable_param_names:
            p.requires_grad = True
        for p in model.module._add_on.parameters():
            p.requires_grad = True
        for p in model.module.last_layer.parameters():
            p.requires_grad = False
        model.module.prototype_vectors.requires_grad = True
    else:
        for name, p in model.features.named_parameters():
#             if name in trainable_param_names:
            p.requires_grad = True
        for p in model._add_on.parameters():
            p.requires_grad = True
        for p in model.last_layer.parameters():
            p.requires_grad = False
        model.prototype_vectors.requires_grad = True
