# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import cPickle 

from .generate_anchors import generate_anchors
from .nms_hw import nms_hw
from config import cfg
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.nms_wrapper import nms
from utils.fixed_point import convert_to_float_py
from utils.refModel_log import print_msg

DEBUG = True
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, fl_cls_prob, fl_bbox_pred, feat_stride=[16,], anchor_scales = [8, 16, 32], base_size = 10, ratios =[0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0], pre_nms_topN = 2000, max_nms_topN = 400, isHardware=False, num_stddev=2.0):
        """
        Parameters
        ----------
        rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                                                 NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
        rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
        im_info: a list of [image_height, image_width, scale_ratios]
        cfg_key: 'TRAIN' or 'TEST'
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
        """
        _anchors =      generate_anchors(base_size, ratios, anchor_scales)
        _num_anchors = _anchors.shape[0]
        im_info = im_info[0]

        assert rpn_cls_prob_reshape.shape[0] == 1, \
                'Only single item batches are supported'

        # Convert fixed point int to floats fror internal calculations ! 
        rpn_cls_prob_reshape = convert_to_float_py(rpn_cls_prob_reshape, fl_cls_prob)
        rpn_bbox_pred = convert_to_float_py(rpn_bbox_pred, fl_bbox_pred)

        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh        = cfg[cfg_key].RPN_NMS_THRESH
        min_size          = cfg[cfg_key].RPN_MIN_SIZE

        height, width = rpn_cls_prob_reshape.shape[1:3]

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        # (1, H, W, A)
        scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:,:,:,:,1],
                                                [1, height, width, _num_anchors])

        # TODO: NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
        # TODO: if you use the old trained model, VGGnet_fast_rcnn_iter_70000.ckpt, uncomment this line
        scores = rpn_cls_prob_reshape[:,:,:,_num_anchors:]

        bbox_deltas = rpn_bbox_pred
        #im_info = bottom[2].data[0, :]

        if DEBUG:
                print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                print 'scale: {}'.format(im_info[2])
                print 'min_size: {}'.format(min_size)
                print 'max_nms_topN: {}'.format(max_nms_topN)
                print 'post_nms_topN: {}'.format(post_nms_topN)

        # 1. Generate proposals from bbox deltas and shifted anchors
        if DEBUG:
                print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * feat_stride
        shift_y = np.arange(0, height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = _num_anchors
        K = shifts.shape[0]
        anchors = _anchors.reshape((1, A, 4)) + \
                          shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.reshape((-1, 4)) #(HxWxA, 4)

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, isHardware)
        proposals = proposals.astype(bbox_deltas.dtype)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        #KM:  Move filtering into NMS (after estimating parameters
        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        #keep = _filter_boxes(proposals, min_size * im_info[2])
        #proposals = proposals[keep, :]
        #
        #print '[Ref Model Log] Num total Proposals before NMS : ' + str(proposals.shape)
        #scores = scores[keep]

        # # remove irregular boxes, too fat too tall
        # keep = _filter_irregular_boxes(proposals)
        # proposals = proposals[keep, :]
        # scores = scores[keep]

        # Hardware modeling             
        if (isHardware): 
        #if (0): 
                #proposals1 = np.copy(proposals)
                #scores1 = np.copy(scores)
                #KM:  Proposal inputs to NMS need to be in same order as HW or final results will be different!
                proposals1 = np.zeros(proposals.shape)
                scores1 = np.zeros(scores.shape)
                idy = 0
                for k in range(0,A):
                        for j in range(0,width):
                                for i in range(0,height):
                                        idx = (i*width*A)+(j*A)+k
                                        scores1[idy] = scores[idx]
                                        proposals1[idy] = proposals[idx]
                                        print_msg(str(k) + '.' + str(j) + '.' + str(i) + ' Proposal ' + str(idy) + ' -> [' + str(int(8*scores1[idy])) + '] ' + str((16*proposals1[idy,:]).astype(int)),1)
                                        idy = idy+1
                prop, score = nms_hw(proposals1, scores1, num_stddev, nms_thresh, min_size, im_info[2], max_nms_topN, post_nms_topN)
                batch_inds = np.zeros((prop.shape[0], 1), dtype=np.float32)
                blob = np.hstack((batch_inds, prop.astype(np.float32, copy=False)))                             
        else:
                order = scores.ravel().argsort()[::-1]
                if pre_nms_topN > 0:
                        order = order[:pre_nms_topN]
                proposals = proposals[order, :]
                scores = scores[order]
                keep = nms(np.hstack((proposals, scores)), nms_thresh)
                if post_nms_topN > 0:
                        keep = keep[:post_nms_topN]
                proposals = proposals[keep, :]
                scores = scores[keep]
                print 'Number of proposals : ' + str(len(keep))
                batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
                blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        return blob

#def _filter_boxes(boxes, min_size):
#        """Remove all boxes with any side smaller than min_size."""
#        ws = boxes[:, 2] - boxes[:, 0] + 1
#        hs = boxes[:, 3] - boxes[:, 1] + 1
#        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
#        return keep
#
#def _filter_irregular_boxes(boxes, min_ratio = 0.2, max_ratio = 5):
#        """Remove all boxes with any side smaller than min_size."""
#        ws = boxes[:, 2] - boxes[:, 0] + 1
#        hs = boxes[:, 3] - boxes[:, 1] + 1
#        rs = ws / hs
#        keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
#        return keep
