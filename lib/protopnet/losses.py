import torch


def cluster_sep_loss_fn(model, min_distances, label, mgpus):

    if mgpus:
        max_dist = (model.module.prototype_shape[1] * model.module.prototype_shape[2] * model.module.prototype_shape[3])
        prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
    else:
        max_dist = (model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3])
        prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,label]).cuda()

    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
    loss_cluster = torch.mean(max_dist - inverted_distances)

    prototypes_of_wrong_class = 1 - prototypes_of_correct_class

    inverted_distances_to_nontarget_prototypes, _ = \
        torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
    loss_sep = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

#     avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) \
#                             / torch.sum(prototypes_of_wrong_class, dim=1)
#     avg_separation_cost = torch.mean(avg_separation_cost)

    return loss_cluster, loss_sep #, avg_separation_cost

def l1_loss_fn(model, mgpus):

    if mgpus:
        l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
        loss_l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
    else:
        l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
        loss_l1 = (model.last_layer.weight * l1_mask).norm(p=1)        

    return loss_l1
