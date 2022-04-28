import os
import shutil
import numpy as np
import torch

from .preprocess import preprocess_input_function


def join_prototypes_by_activations(ppnet, p, data_loader,
                                   joint_optimizer=None, warm_optimizer=None, last_layer_optimizer=None,
                                   no_p=None, preprocess_input_function=None):

    """
    Join similar prototypes
    :param ppnet: model
    :param p: percentage of closest prototypes which should be joined
    :param data_loader loader for data
    """

    for idx, (search_batch_input, search_y) in enumerate(data_loader):

        if preprocess_input_function is not None:
            search_batch = preprocess_input_function(search_batch_input)
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.cuda()
            _, info = ppnet.forward(search_batch)
            activation = info[0]
        if idx == 0:
            activations = activation
        else:
            activations = torch.cat((activations, activation))

    distances_act = calculate_distances(torch.transpose(activations, 1, 0))

    assert 0 < p < 1
    ind = np.diag_indices(distances_act.shape[0])
    distances_act[ind[0], ind[1]] = np.inf

    if no_p is None:
        k = torch.kthvalue(distances_act.min(0)[0].cpu(), int(p * distances_act.cpu().size(0)))[0].item()
    else: 
        k = torch.kthvalue(distances_act.min(0)[0].cpu(), no_p)[0].item()

    dist_iterator = 0
    no_of_prototypes = len(distances_act)
    proto_joined = []
    print(f"distances smaller than {k:.4g}: {(distances_act.detach().cpu().numpy() <= k).sum()}")
    distances = distances_act
    protos_ = np.arange(0, 2000)

    while dist_iterator < no_of_prototypes:
        proto_distanses = distances[dist_iterator].cpu().detach().numpy()
        if (proto_distanses <= k).any():
            to_join = np.argwhere(proto_distanses <= k)[:, 0]
            ppnet.module.last_layer.weight.data[:, dist_iterator] = \
                ppnet.module.last_layer.weight.data[:, [dist_iterator, *to_join]].sum(1)
            ppnet.module.prototype_class_identity[dist_iterator] = \
                ppnet.module.prototype_class_identity[[dist_iterator, *to_join], :].max(0)[0]
            left_proto = np.setdiff1d(np.arange(ppnet.module.last_layer.weight.data.shape[1]), to_join)
            joined = protos_[to_join]
            protos_ = protos_[left_proto]
            proto_joined.append([protos_[dist_iterator], joined])
            with torch.no_grad():
                ppnet.module.last_layer.weight = torch.nn.Parameter(ppnet.module.last_layer.weight[:, left_proto])
                ppnet.module.prototype_class_identity = ppnet.module.prototype_class_identity[left_proto]
                ppnet.module.prototype_vectors = torch.nn.Parameter(ppnet.module.prototype_vectors[left_proto])
                ppnet.module.ones = torch.nn.Parameter(ppnet.module.ones[left_proto])
                distances = distances[np.ix_(left_proto, left_proto)]
                if joint_optimizer:
                    joint_optimizer.param_groups[2]['params'][0] = joint_optimizer.param_groups[2]['params'][0][left_proto]
                if warm_optimizer:    
                    warm_optimizer.param_groups[1]['params'][0] = warm_optimizer.param_groups[1]['params'][0][left_proto]
                if last_layer_optimizer:
                    last_layer_optimizer.param_groups[0]['params'][0] = last_layer_optimizer.param_groups[0]['params'][0][:, left_proto]
            no_of_prototypes = len(left_proto)
        dist_iterator += 1

    ppnet.module.num_prototypes = no_of_prototypes
    ppnet.module.prototype_shape = ppnet.module.prototype_vectors.shape
    print(f"prototypes after join: {no_of_prototypes}")
    return proto_joined


def calculate_distances(x):

    n, _ = x.shape
    x2 = torch.einsum('ij,ij->i', x, x)
    y2 = x2.view(1, -1)
    x2 = x2.view(-1, 1)
    xy = torch.einsum('ij,kj->ik', x, x)
    x2 = x2.repeat(1, n)
    y2 = y2.repeat(n, 1)
    norm2 = x2 - 2 * xy + y2
    norm2 = norm2.abs()

    norm2[range(n), range(n)] = np.inf

    return norm2
