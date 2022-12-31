# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss,weight_l1_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from siamban.models.refine import get_refine
from siamban.models.refine.proposals2_2 import GetProposal
from siamban.utils.mask_target_builder1 import _build_proposal_target_cuda
from siamban.core.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from siamban.utils.get_prroi_pool import GetPrroiPoolFeature

# noinspection DuplicatedCode
class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

        ## 2-stages
        if cfg.REFINE.REFINE:
            self.refine = get_refine(cfg.REFINE.TYPE,
                                         **cfg.REFINE.KWARGS)

            self.avg_poolz = PrRoIPool2D(cfg.TRAIN.ROIPOOL_OUTSIZE, cfg.TRAIN.ROIPOOL_OUTSIZE,
                                        1/8)
            self.avg_poolx = PrRoIPool2D(cfg.TRAIN.ROIPOOL_OUTSIZE, cfg.TRAIN.ROIPOOL_OUTSIZE,
                                    1/8)
            self.get_proposals = GetProposal()

    def template(self, z,gt_bbox=None):
        with torch.no_grad():
            zf = self.backbone(z)
            if cfg.ADJUST.ADJUST:
                zf_ = self.neck(zf)

            zf = [f[0] for f in zf_]
            zf_raw = [f[1] for f in zf_]

            self.zf = zf
            if cfg.REFINE.REFINE and gt_bbox is not None:

                self.z_roi=GetPrroiPoolFeature(zf_raw, gt_bbox,
                                               cfg.TRAIN.EXEMPLAR_SIZE,
                                               self.avg_poolz,type='template')

    def track(self, x):
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)

            self.xf = xf
            cls, loc = self.head(self.zf, xf)
            if cfg.REFINE.REFINE:
                all_rois=self.get_proposals(loc=loc,xf=self.xf,avg_poolz=self.avg_poolz,type='track')

                z_rois=[self.z_roi[i].expand_as(all_rois[i]) for i in range(len(all_rois))]
                final_scores = self.refine.scores_refine(z_rois, all_rois)
                final_scores = F.softmax(final_scores, dim=1).data[...,1]

                cls,_=self.head(self.zf,self.xf,ref_scores=final_scores)
                return {
                        'cls': cls,
                        'loc': loc,
                       }
            else:
                return {
                    'cls': cls,
                    'loc': loc
                }

    def refine_process(self,box):
        with torch.no_grad():

            all_proposals_roi= \
                GetPrroiPoolFeature(self.xf, torch.from_numpy(box), cfg.TRACK.INSTANCE_SIZE, self.avg_poolx,type='template')

            final_delta=self.refine.box_refine(self.z_roi, all_proposals_roi)
        return final_delta

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data,epoch=None):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        if cfg.REFINE.REFINE:
            template_bbox = data['template_bbox']

        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf_ = self.neck(zf)
            xf = self.neck(xf)

        zf = [f[0] for f in zf_]
        zf_raw = [f[1] for f in zf_]

        cls, loc = self.head(zf, xf)

        loc_ = None
        outputs = {}
        if cfg.REFINE.REFINE:

            search_bbox = data['search_bbox'].numpy()
            search_bbox_cuda = data['search_bbox'].cuda()
            batch_weight=data['batch_weight'].numpy()
            batch_weight_cuda=data['batch_weight'].cuda()
            if epoch >= cfg.TRAIN.HNM_EPOCH:
                loc_=loc.cpu().detach().numpy()

            support_rois, proposal_rois, matching_label,pos_proposals_score,pos_proposals_rois_box,pos_proposals_box=self.get_proposals(self.avg_poolz,self.avg_poolx,loc=loc_,##16+1+1个pos
                                      zf=zf_raw, xf=xf,template_bbox=template_bbox, search_bbox=search_bbox, batch_weight=batch_weight,type='train')

            final_scores,final_deltas=self.refine(support_rois, proposal_rois,pos_proposals_rois_box)

            final_scores = F.log_softmax(final_scores, dim=1)

            cls_refine_loss=select_cross_entropy_loss(final_scores, torch.from_numpy(matching_label).cuda().contiguous())

            regression_target = _build_proposal_target_cuda(torch.from_numpy(pos_proposals_box).cuda(),
                                                            search_bbox_cuda)
            loc_refine_loss = weight_l1_loss(final_deltas,regression_target ,
                                         batch_weight=batch_weight_cuda)

            outputs['cls_refine_loss']=cls_refine_loss

            outputs['loc_refine_loss'] = loc_refine_loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        if cfg.REFINE.REFINE:
            outputs['total_loss'] = outputs['total_loss'] + cfg.TRAIN.REFINE_WEIGHT_SCORE * cls_refine_loss + cfg.TRAIN.REFINE_WEIGHT_BOX * loc_refine_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs
