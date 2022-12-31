import torch
import numpy as np

from siamban.core.config import cfg

class GenerateRoiLabel:
    def __init__(self):
        super(GenerateRoiLabel, self).__init__()

    def __call__(self, pos_sup_roi, pos_proposals_rois, neg_proposals_rois,batch_weight):
        length = len(pos_sup_roi)
        batch_size = pos_sup_roi[0].shape[0]

        wh = pos_sup_roi[0].size(-1)
        pos_proposals_rois = [roi.reshape(batch_size, 1,-1, 256, wh, wh) for roi in pos_proposals_rois]
        neg_proposals_rois_1 = [roi.reshape(batch_size, 1,-1, 256, wh, wh)[:, :,0:16, ...] for roi in neg_proposals_rois]
        neg_proposals_rois_2 = [roi.reshape(batch_size, 1,-1, 256, wh, wh)[:, :,16:32, ...] for roi in neg_proposals_rois]
        neg_proposals_rois_3 = [roi.reshape(batch_size, 1,-1, 256, wh, wh)[:, :,32:48, ...] for roi in neg_proposals_rois]
        pos_sup_rois = [roi.view(batch_size, 1, 1, 256, wh, wh).expand_as(pos_proposals_rois[0]) for roi in pos_sup_roi]

        if cfg.TRAIN.CE_LOSS:
            training_pairs = [(1, pos_sup_rois, pos_proposals_rois), (0, pos_sup_rois, neg_proposals_rois_1),
                              (0, pos_sup_rois, neg_proposals_rois_2), (0, pos_sup_rois, neg_proposals_rois_3)]
        elif cfg.TRAIN.MSE_LOSS:
            training_pairs = [([1, 0], pos_sup_rois, pos_proposals_rois), ([0, 1], pos_sup_rois, neg_proposals_rois_1),
                              ([0, 1], pos_sup_rois, neg_proposals_rois_2), ([0, 1], pos_sup_rois, neg_proposals_rois_3)]

        final_proposal_rois=[torch.cat((training_pairs[0][2][i], training_pairs[1][2][i],training_pairs[2][2][i], training_pairs[3][2][i]), dim=1) for i in range(length)]
        final_support_rois=[torch.cat((training_pairs[0][1][i], training_pairs[1][1][i],training_pairs[2][1][i], training_pairs[3][1][i]), dim=1) for i in range(length)]

        tmp_label=np.array([[training_pairs[0][0]] * 16, [training_pairs[1][0]] * 16,[training_pairs[2][0]] * 16, [training_pairs[3][0]] * 16])
        final_label=np.tile(tmp_label,(batch_size,1,1))

        final_label = final_label * batch_weight.reshape(-1, 1, 1)
        return final_support_rois, final_proposal_rois, final_label, [roi.reshape(-1, 256, wh, wh) for roi in pos_sup_rois]