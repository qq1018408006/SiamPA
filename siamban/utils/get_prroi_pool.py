#from siamban.core.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
#from siamban.core.config import cfg
from siamban.utils.bbox import center2corner_rect,lt2corner_rect
import torch
#import torch.nn as nn
import numpy as np
#from siamban.models.DCN.DCN.dcn_v2 import DCNPooling

# class GetPrroiPoolFeature():
#     def __init__(self):
#         super(GetPrroiPoolFeature, self).__init__()
#         # if use_prpooling:
#         #     self.avg_pool = PrRoIPool2D(cfg.TRAIN.ROIPOOL_OUTSIZE, cfg.TRAIN.ROIPOOL_OUTSIZE,
#         #                                 1 / 8).cuda()
#         # else:
#         #     self.avg_pool=DCNPooling(spatial_scale=1/8,
#         #                              pooled_size=cfg.TRAIN.ROIPOOL_OUTSIZE ,
#         #                              output_dim=256,
#         #                              no_trans=False,
#         #                              trans_std=0.1).cuda()
#         #self.avg_pool=avg_pool
def GetPrroiPoolFeature(feature, bboxes, origin_size, avg_pool,type=None):
    ##bboxes:x1y1x2y2
    roi_feature_temp = []
    for i in range(len(feature)):
        feature[i]=feature[i].detach()
    # batch_size = cfg.TRAIN.BATCH_SIZE
    batch_size = feature[0].shape[0] #if cfg.BACKBONE.TYPE=='res50' else feature.shape[0]
    if type == 'template':##推理和训练时模板用，box（4，）和（b，4）
        # dim = 5,(batch_id, x1, y1, x2, y2)
        _bboxes = torch.zeros(batch_size, 5)
        _bboxes[:, 0] = torch.tensor(range(0, batch_size))
        _bboxes[:,1:] = bboxes##boxes===(4,)

        for i in range(len(feature)):
            roi_feature_temp.append(avg_pool(feature[i], _bboxes.cuda()))

        # for i,j in enumerate(roi_feature_temp):
        #     roi_feature_temp[i]=j.unsqueeze(1)

    else:
        if type == 'track':##cls refine时用，box（625，4）
            # bbox is lt-based, need to change to corner-base
            ##bboxes_num_per_batch = 1##boxes===(4,)##loc_refine时使用

            bboxes_num_per_batch = bboxes.shape[0]
            bboxes = center2corner_rect(bboxes)
        elif type == 'search':##训练时用，box（b,n,4）
            # bboxes=bboxes.transpose(1,0)
            # bboxes_num_per_batch = bboxes.shape[0]
            # bboxes = center2corner_rect(bboxes)

            bboxes_num_per_batch=bboxes.shape[1]
            bboxes=lt2corner_rect(bboxes).reshape(batch_size*bboxes_num_per_batch,4)
            bboxes = bboxes.reshape(batch_size * bboxes_num_per_batch, 4)

        # change to corner-base

        # check value (<0 or >255/127)
        bboxes[np.where(bboxes < 0)] = 0
        bboxes[np.where(bboxes > origin_size)] = origin_size

        _bboxes = torch.zeros(batch_size * bboxes_num_per_batch, 4+1)
        _bboxes[:, 0] = \
            torch.tensor(list(range(batch_size)) * bboxes_num_per_batch).view(-1, batch_size).permute(1, 0).reshape(-1)
        _bboxes[:, 1:] = torch.from_numpy(bboxes)

        for i in range(len(feature)):
            roi_feature_temp.append(avg_pool(feature[i], _bboxes.cuda()))

        # for i, j in enumerate(roi_feature_temp):
        #     roi_feature_temp[i] = j.reshape(batch_size , bboxes_num_per_batch, j.shape[1],j.shape[2],j.shape[3])
    return roi_feature_temp
