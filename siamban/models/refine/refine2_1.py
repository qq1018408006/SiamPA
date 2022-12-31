from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.attention.non_local import SEModule

'''corner prediction version'''
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))

class PtCorr(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, pool_size=7, use_post_corr=True):
        super().__init__()
        num_corr_channel = pool_size*pool_size
        self.use_post_corr = use_post_corr
        if use_post_corr:
            self.post_corr = nn.Sequential(
                nn.Conv2d(num_corr_channel, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, num_corr_channel, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(num_corr_channel),
                nn.ReLU(),
            )
        self.channel_attention = SEModule(num_corr_channel,reduction=4)
        self.spatial_attention = nn.Sequential()

    def get_ref_kernel(self, feat1):
        sr = 2  # TODO: adaptive crop region according to search region rate
        self.ref_kernel = feat1[:, :,
                          feat1.shape[2]*(sr-1)//(sr*2):feat1.shape[2]*(sr+1)//(sr*2),
                          feat1.shape[3]*(sr-1)//(sr*2):feat1.shape[3]*(sr+1)//(sr*2)]

    def fuse_feat(self, feat1,feat2):
        """ fuse features from reference and test branch """

        # Step1: pixel-wise correlation
        feat_corr, _ = self.corr_fun(feat1, feat2)

        # Step2: channel attention: Squeeze and Excitation
        if self.use_post_corr:
            feat_corr = self.post_corr(feat_corr)
        feat_ca = self.channel_attention(feat_corr)
        return feat_ca

    def corr_fun(self, ker, feat):
        return self.corr_fun_mat(ker, feat)

    def corr_fun_mat(self, ker, feat):
        b, c, h, w = feat.shape
        ker = ker.reshape(b, c, -1).transpose(1, 2)
        feat = feat.reshape(b, c, -1)
        corr = torch.matmul(ker, feat)
        corr = corr.reshape(*corr.shape[:2], h, w)
        return corr, ker

    def corr_fun_loop(self, Kernel_tmp, Feature, KERs=None):
        b, c, _, _ = Kernel_tmp.shape
        size = Kernel_tmp.size()
        CORR = []
        Kernel = []
        for i in range(len(Feature)):
            ker = Kernel_tmp[i:i + 1]
            fea = Feature[i:i + 1]
            ker = ker.reshape(size[1], size[2] * size[3]).transpose(0, 1)
            ker = ker.unsqueeze(2).unsqueeze(3)
            if not (type(KERs) == type(None)):
                ker = torch.cat([ker, KERs[i]], 0)
            co = F.conv2d(fea, ker)
            CORR.append(co)
            ker = ker.unsqueeze(0)
            Kernel.append(ker)
        corr = torch.cat(CORR, 0)
        Kernel = torch.cat(Kernel, 0)
        return corr, Kernel

    def forward(self,zf,xf):
        return self.fuse_feat(zf,xf)


class Refinement(nn.Module):
    def __init__(self,hidden_channels=256):
        super(Refinement, self).__init__()
        self.score_conv = nn.Sequential(nn.Conv2d(cfg.TRAIN.ROIPOOL_OUTSIZE*cfg.TRAIN.ROIPOOL_OUTSIZE, hidden_channels, kernel_size=3, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(inplace=True),
                                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(hidden_channels),
                                    nn.AdaptiveAvgPool2d((1, 1))
                                  )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels*2, 2)
        )
        self.box_head = RPN_head(cfg.TRAIN.ROIPOOL_OUTSIZE*cfg.TRAIN.ROIPOOL_OUTSIZE)
        self.gamma = nn.Parameter(torch.ones(len(cfg.BACKBONE.KWARGS.used_layers)))

        self.num = len(cfg.BACKBONE.KWARGS.used_layers)
        for i in range(self.num):

            self.add_module('xcorr' + str(i),
                            PtCorr(pool_size=cfg.TRAIN.ROIPOOL_OUTSIZE))

    def forward(self,support_rois, proposal_rois,pos_proposals_rois_box):
        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        b, _, n, c, w, h = proposal_rois[0].shape
        support_rois_box=[i[:,0,...].reshape(-1, c, w, h) for i in support_rois]
        pos_proposals_rois_box=[i.reshape(-1, c, w, h) for i in pos_proposals_rois_box]
        support_rois=[i.reshape(-1,c,w,h) for i in support_rois]
        proposal_rois = [i.reshape(-1, c, w, h) for i in proposal_rois]
        z_out = []
        x_out = []
        for i in range(self.num):
            xcorr_model= getattr(self, 'xcorr' + str(i))
            zout = xcorr_model(support_rois[i], proposal_rois[i])

            xout=xcorr_model(support_rois_box[i], pos_proposals_rois_box[i])
            z_out.append(zout)
            x_out.append(xout)

        gamma = F.softmax(self.gamma, 0)
        score_feats = weighted_avg(z_out, gamma)
        box_feats = weighted_avg(x_out, gamma)
        c=box_feats.shape[1]
        score_feats=self.score_conv(score_feats).squeeze(-1).squeeze(-1)
        scores=self.score_head(score_feats)

        box=self.box_head(box_feats)
        box=box.reshape(b,n,-1)

        return scores, box

    def scores_refine(self,z_rois,x_rois):
        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        z_out = []
        for i in range(self.num):
            xcorr_model = getattr(self, 'xcorr' + str(i))
            zout = xcorr_model(z_rois[i], x_rois[i])

            z_out.append(zout)

        gamma = F.softmax(self.gamma, 0)
        score_feats = weighted_avg(z_out, gamma)
        score_feats = self.score_conv(score_feats).squeeze(-1).squeeze(-1)
        scores = self.score_head(score_feats)
        return scores

    def box_refine(self,z_rois,x_rois):
        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        x_out = []
        for i in range(self.num):

            xcorr_model = getattr(self, 'xcorr' + str(i))
            xout = xcorr_model(z_rois[i], x_rois[i])
            x_out.append(xout)

        gamma = F.softmax(self.gamma, 0)
        box_feats = weighted_avg(x_out, gamma)
        box=self.box_head(box_feats).reshape(1,-1)
        return box

class RPN_head(nn.Module):
    def __init__(self, inplanes=64, channel=256):
        super(RPN_head, self).__init__()

        self.conv1_tl = conv(inplanes, channel)
        self.conv2_tl = conv(channel, channel // 2)
        self.conv3_tl = conv(channel // 2, channel // 4)
        self.conv4_tl = conv(channel // 4, channel // 8)
        self.conv5_tl = nn.Conv2d(channel // 8, 4, kernel_size=1)

    def forward(self, x):
        score_map_tl = self.get_score_map(x)
        b,c,w,h = score_map_tl.shape
        return score_map_tl.view(b,c,-1).mean(-1)

    def get_score_map(self, x):
        x_tl1 = self.conv1_tl(x)
        x_tl1 = self.conv2_tl(x_tl1)
        x_tl1 = self.conv3_tl(x_tl1)
        x_tl1 = self.conv4_tl(x_tl1)
        score_map_tl = self.conv5_tl(x_tl1)

        return score_map_tl

