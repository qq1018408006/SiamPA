import torch
from torch import nn

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            if losses.mean()<0:
                print(losses.mean())
            return losses.mean()

class iou_loss(nn.Module):
    def __init__(self,):
        super(iou_loss, self).__init__()

    def forward(self, preds, bbox, eps=1e-6, reduction='mean'):
        '''
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py
        :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :return: loss
        '''
        x1 = torch.max(preds[:, 0], bbox[:, 0])
        y1 = torch.max(preds[:, 1], bbox[:, 1])
        x2 = torch.min(preds[:, 2], bbox[:, 2])
        y2 = torch.min(preds[:, 3], bbox[:, 3])

        w = (x2 - x1 + 1.0).clamp(0.)
        h = (y2 - y1 + 1.0).clamp(0.)

        inters = w * h

        uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
                bbox[:, 3] - bbox[:, 1] + 1.0) - inters

        ious = (inters / uni).clamp(min=eps)
        # loss = -ious.log()
        loss=1-ious

        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError
        return loss

linear_iou = IOULoss(loc_loss_type='linear_iou')

iouloss=iou_loss()
