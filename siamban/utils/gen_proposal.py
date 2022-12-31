import numpy as np
import cv2
import torch

from siamban.utils.bbox import center2corner_rect, corner2lt

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x1,y1,w,h] or
            2d array of N x [x1,y1,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

class SampleGenerator():
    def __init__(self, type_, img_size, trans=1, scale=1, aspect=None, valid=False):
        self.type = type_
        self.img_size = np.array(img_size)  # (w, h)
        self.trans = trans
        self.scale = scale
        self.aspect = aspect
        self.valid = valid

    def _gen_samples(self, bb, n):
        bb = np.array(bb, dtype='float32')

        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None, :], (n, 1))

        # vary aspect ratio
        if self.aspect is not None:
            ratio = np.random.rand(n, 2) * 2 - 1
            samples[:, 2:] *= self.aspect ** ratio

        # sample generation
        if self.type == 'gaussian':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

        elif self.type == 'uniform':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == 'whole':
            m = int(2 * np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range
        samples[:, 2:] = np.clip(samples[:, 2:], 1, self.img_size-1)
        if self.valid:
            samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
        else:
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        samples[:, :2] -= samples[:, 2:] / 2

        np.random.shuffle(samples)
        return samples

    def __call__(self, bboxes, n, overlap_range=None, scale_range=None):
        '''
        :param bboxes:  [b, xmin,ymin,xmax,ymax]
        :param n:
        :param overlap_range:
        :param scale_range:
        :return:x1y1wh
        '''
        if isinstance(bboxes, torch.Tensor):
            bboxes = np.array(bboxes)
        bs = bboxes.shape[0]
        ss = np.zeros((bs, n, 4),dtype=np.float32)
        for i in range(bs):
            bbox = bboxes[i]
            bbox = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            if overlap_range is None and scale_range is None:
                return self._gen_samples(bbox, n)

            else:
                while True:
                    samples=self._generator(n,bbox,overlap_range,scale_range)
                    try:
                        ss[i, ...] = samples
                        break
                    except ValueError:
                        print(bbox)
                        print(i)
                        self.trans /= 2
                        self.scale = (self.scale+1)/2
        return ss

    def _generator(self,n,bbox,overlap_range,scale_range):
        '''
        Args:
            n:
            bbox: x1y1wh
            overlap_range:
            scale_range:

        Returns:
        '''
        samples = None
        remain = n
        factor = 2
        while remain > 0 and factor < 32:
            samples_ = self._gen_samples(bbox, remain * factor)

            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                r = overlap_ratio(samples_, bbox)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])

            samples_ = samples_[idx, :]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None or len(samples) == 0:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor * 2
        return samples

