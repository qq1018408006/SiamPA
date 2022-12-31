from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

def _build_proposal_target_cuda(rpn_rois, gt_boxes):
    gt_boxes=gt_boxes.unsqueeze(1).expand_as(rpn_rois)

    delta=torch.zeros_like(rpn_rois)
    tcx,tcy,tw,th=(gt_boxes[..., 0] + gt_boxes[..., 2]) * 0.5,(gt_boxes[..., 1] + gt_boxes[..., 3]) * 0.5,\
                  gt_boxes[..., 2] - gt_boxes[..., 0],gt_boxes[..., 3] - gt_boxes[..., 1]
    cx,cy,w,h=rpn_rois[..., 0] + rpn_rois[..., 2] * 0.5,rpn_rois[..., 1] + rpn_rois[..., 3] * 0.5,\
              rpn_rois[..., 2],rpn_rois[..., 3]

    delta[...,0] = (tcx - cx) / w
    delta[...,1] = (tcy - cy) / h
    delta[...,2] = torch.log(tw / w)
    delta[...,3] = torch.log(th / h)
    return delta

