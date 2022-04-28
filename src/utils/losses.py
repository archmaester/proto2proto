import torch
import torch.nn as nn


def featureLoss_mask(masks, student_features, teacher_features):
    '''
    '''

    norms = masks.sum() * 2

    _, tc, th, tw = teacher_features.size()
    _, sc, sh, sw = student_features.size()

    spl = spr = spt = spb = 0
    tpl = tpr = tpt = tpb = 0

    if th > sh:
        diff = th - sh
        spt = diff // 2
        spb = diff - spt
    else:
        diff = sh - th
        tpt = diff // 2
        tpb = diff - tpt

    if tw > sw:
        diff = tw - sw
        spl = diff // 2
        spr = diff - spl
    else:
        diff = sw - tw
        tpl = diff // 2
        tpr = diff - tpl

    student_features = nn.ConstantPad2d((spl, spr, spt, spb), 0)(student_features)
    teacher_features = nn.ConstantPad2d((tpl, tpr, tpt, tpb), 0)(teacher_features)
    mask_batch = nn.ConstantPad2d((tpl, tpr, tpt, tpb), 0)(masks)
    mask_batch = mask_batch.unsqueeze(1).to('cuda')
    sup_loss = (torch.pow(teacher_features - student_features, 2) * mask_batch).sum() / (norms+1)
    return sup_loss
