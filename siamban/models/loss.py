# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.iou_loss import linear_iou

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze()
    neg = label.data.eq(0).nonzero().squeeze()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def weight_l1_loss(pred_loc, label_loc, loss_weight=1/(cfg.TRAIN.PROPOSAL_POS),batch_weight=None):
    pos=batch_weight.view(-1).data.eq(1).nonzero().squeeze()
    pred_loc = torch.index_select(pred_loc, 0, pos)
    label_loc = torch.index_select(label_loc, 0, pos)
    b= pred_loc.size()[0]

    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=2)
    loss = diff * loss_weight
    return loss.sum().div(b)

def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return linear_iou(pred_loc, label_loc)
