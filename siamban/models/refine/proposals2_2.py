from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from siamban.utils.get_prroi_pool import GetPrroiPoolFeature
from siamban.utils.gen_proposal import SampleGenerator
from siamban.datasets.proposals_target1 import ProposalTarget
from siamban.datasets.maching_net_rois_label1 import GenerateRoiLabel
from siamban.core.config import cfg

class GetProposal:

    def __init__(self):
        self.get_proposals = ProposalTarget()
        self.generate_roi_label = GenerateRoiLabel()
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.points=self.generate_points(cfg.POINT.STRIDE,(cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE)

    def __call__(self, avg_poolz=None,avg_poolx=None, loc=None, zf=None, xf=None,
                 template_bbox=None, search_bbox=None,batch_weight=None,type='train'):
        '''
        template_bbox:x1y1x2y2
        search_bbox:x1y1x2y2
        '''
        if type == 'train':
            center_pos = np.array([cfg.TRAIN.SEARCH_SIZE / 2, cfg.TRAIN.SEARCH_SIZE / 2])

            pos_proposals_score = SampleGenerator(cfg.REFINE.POS_SAMPLETYPE, (cfg.TRAIN.SEARCH_SIZE,cfg.TRAIN.SEARCH_SIZE),
                                            cfg.REFINE.TRANS_POS, cfg.REFINE.SCALE_POS)(search_bbox,
                                                                                          cfg.REFINE.N_POS_INIT,
                                                                                          cfg.REFINE.OVERLAP_POS_INIT)
            pos_proposals_box = SampleGenerator(cfg.REFINE.POS_SAMPLETYPE, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE),

                                            cfg.REFINE.TRANS_POS, cfg.REFINE.SCALE_POS)(search_bbox,
                                                                                        cfg.REFINE.N_POS_INIT,
                                                                                        cfg.REFINE.OVERLAP_POS_BOX)

            if loc is not None:
                all_proposals = self._convert_bbox(loc, np.expand_dims(self.points,axis=0).repeat(cfg.TRAIN.BATCH_SIZE,axis=0), type='train_lt')

                all_proposals[:,0,:]+=center_pos[0]
                all_proposals[:, 1, :] += center_pos[1]

                all_proposals[:, 0, :], all_proposals[:, 1, :], all_proposals[:, 2, :], all_proposals[:, 3,:] = self._bbox_clip(
                    all_proposals[:, 0, :], all_proposals[:, 1, :], all_proposals[:, 2, :],
                    all_proposals[:, 3, :], [cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE])

                select_proposals_pos, select_proposals_neg, pos_num, neg_num = self.get_proposals(all_proposals,search_bbox)

                for b, n in enumerate(pos_num):
                    if n < cfg.TRAIN.POS_NUM:
                        select_proposals_pos[b][n:, :] = pos_proposals_score[b][:cfg.TRAIN.POS_NUM - n, :]

                pos_proposals_score = select_proposals_pos
                # TODO 添加低分搞iou的样本训练
                neg_proposals = select_proposals_neg
            else:
                    neg_proposals = SampleGenerator(cfg.REFINE.NEG_SAMPLETYPE, (cfg.TRAIN.SEARCH_SIZE,cfg.TRAIN.SEARCH_SIZE),
                                                    cfg.REFINE.TRANS_NEG_INIT, cfg.REFINE.SCALE_NEG_INIT)(search_bbox,
                                                                                                            int(0.5 * cfg.REFINE.N_NEG_INIT),
                                                                                                            cfg.REFINE.OVERLAP_NEG_INIT)

            pos_sup_roi = GetPrroiPoolFeature(zf, template_bbox, cfg.TRAIN.EXEMPLAR_SIZE,avg_poolz,type='template')

            pos_proposals_rois_box = GetPrroiPoolFeature(xf, pos_proposals_box, cfg.TRAIN.SEARCH_SIZE,avg_poolx, type='search')
            pos_proposals_rois_score = GetPrroiPoolFeature(xf, pos_proposals_score, cfg.TRAIN.SEARCH_SIZE, avg_poolx,
                                                         type='search')
            neg_proposals_rois = GetPrroiPoolFeature(xf, neg_proposals, cfg.TRAIN.SEARCH_SIZE,avg_poolx, type='search')

            support_rois, proposal_rois, matching_label,support_rois_for_loc = self.generate_roi_label(pos_sup_roi,pos_proposals_rois_score,
                                                                                  neg_proposals_rois,batch_weight)

            return support_rois, proposal_rois, matching_label,pos_proposals_score,pos_proposals_rois_box,pos_proposals_box


        elif type == 'track':
            all_proposals = self._convert_bbox(loc, self.points, type='track_center')
            center_pos = np.array([cfg.TRACK.INSTANCE_SIZE / 2, cfg.TRACK.INSTANCE_SIZE / 2], dtype=np.float32)
            all_proposals[0, :] += center_pos[0]
            all_proposals[1, :] += center_pos[1]
            all_rois = GetPrroiPoolFeature(xf, all_proposals.transpose(1, 0),
                                           cfg.TRACK.INSTANCE_SIZE,
                                           avg_poolz, type='track')
            return all_rois
    def _convert_bbox(self, delta, point, type='center'):
        if 'train' in type:
            delta=delta.reshape(delta.shape[0],4,cfg.TRAIN.OUTPUT_SIZE*cfg.TRAIN.OUTPUT_SIZE)
            point=point.reshape(point.shape[0],2,cfg.TRAIN.OUTPUT_SIZE*cfg.TRAIN.OUTPUT_SIZE)
            delta[:, 0, :] = point[:, 0, :] - delta[:, 0, :]
            delta[:, 1, :] = point[:, 1, :] - delta[:, 1, :]
            delta[:, 2, :] = point[:, 0, :] + delta[:, 2, :]
            delta[:, 3, :] = point[:, 1, :] + delta[:, 3, :]
            if type == 'train_center':
                # center based
                delta_c = delta.copy()
                delta_c[:, 2, :] = delta[:, 2, :] - delta[:, 0, :]
                delta_c[:, 3, :] = delta[:, 3, :] - delta[:, 1, :]
                delta_c[:, 0, :] = delta[:, 0, :] + 1 / 2 * delta_c[:, 2, :]
                delta_c[:, 1, :] = delta[:, 1, :] + 1 / 2 * delta_c[:, 3, :]
                return delta_c
            elif type == 'train_lt':
                # left top based
                delta_lt = delta.copy()
                delta_lt[:, 2, :] = delta_lt[:, 2, :] - delta_lt[:, 0, :]
                delta_lt[:, 3, :] = delta_lt[:, 3, :] - delta_lt[:, 1, :]
                return delta_lt
        elif 'track' in type:
            delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
            delta = delta.detach().cpu().numpy()
            point = point.reshape(2, -1)
            delta[0, :] = point[0, :] - delta[0, :]
            delta[1, :] = point[1, :] - delta[1, :]
            delta[2, :] = point[0, :] + delta[2, :]
            delta[3, :] = point[1, :] + delta[3, :]
            if type == 'track_center':
                delta_c = delta.copy()
                delta_c[0, :] = (delta[0, :] + delta[2, :]) / 2
                delta_c[1, :] = (delta[1, :] + delta[3, :]) / 2
                delta_c[2, :] = delta[2, :] - delta[0, :] + 1
                delta_c[3, :] = delta[3, :] - delta[1, :] + 1
                return delta_c
            if type == 'track_lt':
                delta_lt = delta.copy()
                delta_lt[2, :] = delta_lt[2, :] - delta_lt[0, :]
                delta_lt[3, :] = delta_lt[3, :] - delta_lt[1, :]
                return delta_lt
        else:
            assert False, 'convert box is no type matching'

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
        points=points.transpose(1,0).reshape(2,size,size)
        return points

    def _bbox_clip(self,x1, y1, width, height, boundary):

        x1 = np.maximum(0, np.minimum(x1, boundary[1]))
        y1 = np.maximum(0, np.minimum(y1, boundary[0]))
        width = np.maximum(10, np.minimum(width, boundary[1]))
        height = np.maximum(10, np.minimum(height, boundary[0]))

        return x1, y1, width, height
