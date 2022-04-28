import torch


def get_optimizer(ppnet, args, mgpus=False, ttype=False):

    if mgpus:
        joint_paramlist = [
            {'params': ppnet.module.features.parameters(), 'lr': args.train.lrNet,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module.prototype_vectors, 'lr': args.train.lrProto}
        ]
        last_paramlist = [
            {'params': ppnet.module.last_layer.parameters(), 'lr': args.train.lrLastLayer}
        ]
        warm_paramlist = [
            {'params': ppnet.module._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.module.prototype_vectors, 'lr': args.train.lrProto}
        ]

    else:
        joint_paramlist = [
            {'params': ppnet.features.parameters(), 'lr': args.train.lrNet,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.prototype_vectors, 'lr': args.train.lrProto}
        ]
        last_paramlist = [
            {'params': ppnet.last_layer.parameters(), 'lr': args.train.lrLastLayer}
        ]
        warm_paramlist = [
            {'params': ppnet._add_on.parameters(), 'lr': args.train.lrBlock,
             'weight_decay': args.train.weightDecay},
            {'params': ppnet.prototype_vectors, 'lr': args.train.lrProto}
        ]

    if ttype:
        adapt_params = {'params': ppnet.stu_feature_adap.parameters(), 'lr': args.train.lrBlock,
                        'weight_decay': args.train.weightDecay}
        warm_paramlist.append(adapt_params)
        joint_paramlist.append(adapt_params)

    return torch.optim.Adam(joint_paramlist), torch.optim.Adam(last_paramlist), torch.optim.Adam(warm_paramlist)


def last_only(model, trainable_param_names, mgpus=False):

    if mgpus:
        for name, p in model.module.features.named_parameters():
#             if name in trainable_param_names:
            p.requires_grad = False
        for p in model.module._add_on.parameters():
            p.requires_grad = False
        for p in model.module.last_layer.parameters():
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = False
    else:
        for name, p in model.features.named_parameters():
#             if name in trainable_param_names:
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
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = True
    else:
        for name, p in model.features.named_parameters():
            if name in trainable_param_names:
                p.requires_grad = False
        for p in model._add_on.parameters():
            p.requires_grad = True
        for p in model.last_layer.parameters():
            p.requires_grad = True
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
