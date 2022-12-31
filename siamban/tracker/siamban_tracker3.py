from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center
from siamban.utils.point import Point

class SiamPATracker(SiameseTracker):
    def __init__(self, model):
        super(SiamPATracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()
        self.points_template=Point(cfg.POINT.STRIDE, 15, cfg.TRAIN.EXEMPLAR_SIZE//2)


    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)

        scale_ratio = cfg.TRACK.EXEMPLAR_SIZE / s_z
        size = np.array([bbox[2], bbox[3]]) * scale_ratio
        ##x1y1x2y2
        bbox = np.array([cfg.TRACK.EXEMPLAR_SIZE / 2 - size[0] / 2,
                                    cfg.TRACK.EXEMPLAR_SIZE / 2 - size[1] / 2,
                                    cfg.TRACK.EXEMPLAR_SIZE / 2 + size[0] / 2,
                                    cfg.TRACK.EXEMPLAR_SIZE / 2 + size[1] / 2])

        self.model.template(z_crop,torch.from_numpy(bbox))
        self.initial_box=bbox

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])

        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        track_results={}

        if cfg.REFINE.REFINE:
            bbox = pred_bbox[:,best_idx]

            center_pos = np.array([cfg.TRACK.INSTANCE_SIZE / 2, cfg.TRACK.INSTANCE_SIZE / 2], dtype=np.float32)

            cx = bbox[0] + center_pos[0]
            cy = bbox[1] + center_pos[1]

            cx, cy, width, height = self._bbox_clip(cx, cy, bbox[2],
                                                    bbox[3], [cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])

            bbox = np.array([cx - width*0.5+0.5,
                    cy - height*0.5+0.5,
                    cx + width*0.5-0.5,
                    cy + height*0.5-0.5])
            pred_bbox=self.model.refine_process(np.array(bbox))

            pred_bbox=pred_bbox.detach().cpu().numpy().squeeze()

            pred_bbox[0] = pred_bbox[0] * width + cx
            pred_bbox[1] = pred_bbox[1] * height + cy
            pred_bbox[2] = np.exp(pred_bbox[2]) * width
            pred_bbox[3] = np.exp(pred_bbox[3]) * height

            bbox = pred_bbox / scale_z
            lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

            cx = bbox[0] + crop_box[0]
            cy = bbox[1] + crop_box[1]


        else:
            bbox = pred_bbox[:, best_idx] / scale_z
            lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        track_results.update({
                'bbox': bbox,
                'best_score': best_score,
                'final_scores':score
               })
        return track_results
