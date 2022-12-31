import numpy as np

from siamban.core.config import cfg
from siamban.utils.bbox import corner2lt, rect_iou


class ProposalTarget:
    def __init__(self):
        self.select_batchind_pos=np.tile(np.arange(cfg.TRAIN.BATCH_SIZE).reshape(-1, 1), (1, cfg.TRAIN.PROPOSAL_POS))
        self.select_batchind_neg =np.tile(np.arange(cfg.TRAIN.BATCH_SIZE).reshape(-1, 1), (1, cfg.TRAIN.PROPOSAL_NEG))
        super(ProposalTarget, self).__init__()

    def __call__(self, all_proposals, gt):
        '''
        :param all_proposals: x1y1wh
        :param gt: x1y1x2y2
        :return:
        '''
        batchsize, num = all_proposals.shape[0], all_proposals.shape[2]     # 28, 625
        select_proposals_pos, select_proposals_neg, pos_num, neg_num  = self.get_select_proposal_label(all_proposals, gt, batchsize, num)
        return select_proposals_pos.transpose(0, 2, 1), \
               select_proposals_neg.transpose(0, 2, 1), \
               pos_num, neg_num

    def get_iou(self, proposals, gt, batchsize, num):
        proposals = proposals.transpose(0, 2, 1).reshape(-1, 4)

        gt = np.tile(corner2lt(gt), num).reshape(-1, 4)
        assert gt.shape == proposals.shape, "proposals size don't match the gt size"
        ious = rect_iou(proposals, gt)
        return ious

    def get_box_center(self, box):
        return (box[2] + box[0]) / 2, (box[3] + box[1]) / 2

    def get_box_center2(self, box,type='gt'):
        if type=='gt':
            return np.stack(((box[:,2] + box[:,0]) / 2, (box[:,3] + box[:,1]) / 2),axis=0)
        elif type=='proposal':
            return np.stack(((box[:, 2,:] + box[:, 0,:]) / 2, (box[:, 3,:] + box[:, 1,:]) / 2), axis=0)
    def dist_2_points(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    def dist_2_points2(self, p1, p2):
        return (p1[...,0] - p2[...,0])**2 + (p1[...,1] - p2[...,1])**2

    def get_select_proposal_label(self, all_proposals, gt, batchsize, num):
        '''
        :param all_proposals: x1y1wh  np(48,4,625)
        :param gt:x1y1x2y2  tensor(48,4)
        :param batchsize:  b
        :param num:  n
        :return:
        '''
        ious = self.get_iou(all_proposals, gt, batchsize, num).reshape(batchsize, num)

        def select(position, keep_num=2,type='pos'):
            if type=='pos':
                count=-1*np.ones([cfg.TRAIN.BATCH_SIZE,cfg.TRAIN.OUTPUT_SIZE*cfg.TRAIN.OUTPUT_SIZE],dtype=np.int16)
                count[(position[:,0],position[:,1])]=position[:,1]
                count=np.take_along_axis(count,np.argsort(-count,axis=1),axis=1)
                num=np.sum(count>-1,axis=1)
                select_num=np.minimum(num,keep_num)

                count[...,-2]=num
                count[..., -1] = select_num

                def select_elements(array):
                    array2=-1*np.ones((keep_num),dtype=np.int16)
                    edge=array[-2]
                    n=array[-1]

                    temp=np.random.choice(array[:edge],n,replace=False)
                    array2[:temp.size]=temp
                    return array2
                return np.apply_along_axis(select_elements,1,count)
            elif type=='neg':
                gt_center2 = self.get_box_center2(gt, type='gt')
                gt_center2 = gt_center2.transpose(1, 0)

                proposal_center2 = self.get_box_center2(all_proposals, type='proposal')
                proposal_center2 = proposal_center2.transpose(2, 1, 0)

                dist2 = self.dist_2_points2(proposal_center2, gt_center2)

                disttmp = 1e10 * np.ones_like(dist2)
                disttmp[position[:, 1], position[:, 0]] = dist2[position[:, 1], position[:, 0]]
                neg_sort_id2 = np.argsort(disttmp.transpose(1, 0), axis=1)

                select_num = keep_num
                select2 = neg_sort_id2[:, :select_num]
                return select2

        pos = np.argwhere(ious > 0.6)
        neg = np.argwhere(ious < 0.3)

        count_neg = select(neg, cfg.TRAIN.PROPOSAL_NEG,type='neg')
        count_pos = select(pos, cfg.TRAIN.PROPOSAL_POS,type='pos')

        all_proposals=np.concatenate((all_proposals,np.zeros((cfg.TRAIN.BATCH_SIZE,4,1))),axis=2)

        select_proposals_neg=all_proposals[(self.select_batchind_neg,...,count_neg)].transpose(0,2,1)
        select_proposals_pos=all_proposals[(self.select_batchind_pos,...,count_pos)].transpose(0,2,1)

        neg_num = [np.sum(count_neg[i]>-1) for i in range(cfg.TRAIN.BATCH_SIZE)]
        pos_num = [np.sum(count_pos[i]>-1) for i in range(cfg.TRAIN.BATCH_SIZE)]

        return select_proposals_pos, select_proposals_neg, pos_num, neg_num
