"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
"""

import numpy as np
from .sqrootLookUp import sqrootLookUp
from utils.refModel_log import print_msg

def _estimate_parameters(scores, num_stddev):
        """ Hardware Mean and standard deviation estimation."""    
        num_proposals = len(scores)

        print_msg('Estimating threshold.',2)
        print_msg('Number of total proposals : ' + str(num_proposals),2)
        
        score_int = scores.astype(int)
        score_squared = score_int**2
        score_sum = np.sum(score_int)
        score_squared_sum = np.sum(score_squared)

        # DO some Fixed point stuff here
        # -> Use 6 frac bits
        if (num_proposals > 196608): 
                by_n_approx = 1.0*(2**18)/num_proposals
                if (score_sum < 0): mean = -1.0*int(-score_sum/(2**12))
                else:               mean =  1.0*int( score_sum/(2**12))
                sq_sum_by_n = 1.0*int(score_squared_sum/(2**12))
        elif (num_proposals > 98304):
                by_n_approx = 1.0*(2**17)/num_proposals
                if (score_sum < 0): mean = -1.0*int(-score_sum/(2**11))
                else:               mean =  1.0*int( score_sum/(2**11))
                sq_sum_by_n = 1.0*int(score_squared_sum/(2**11))
        elif (num_proposals > 49152): 
                by_n_approx = 1.0*(2**16)/num_proposals
                if (score_sum < 0): mean = -1.0*int(-score_sum/(2**10))
                else:               mean =  1.0*int( score_sum/(2**10))
                sq_sum_by_n = 1.0*int(score_squared_sum/(2**10))
        else: 
                by_n_approx = 1.0*(2**15)/num_proposals
                if (score_sum < 0): mean = -1.0*int(-score_sum/(2**9))
                else:               mean =  1.0*int( score_sum/(2**9))
                sq_sum_by_n = 1.0*int(score_squared_sum/(2**9))

        #print 'mean_shifted = {}'.format(str(mean))
        #print 'sq_sum_shifted = {}'.format(str(sq_sum_by_n))
        by_n_approx = 1.0*int((2**6)*by_n_approx)
        mean = (by_n_approx*mean)/(2**12)
        sq_sum_by_n = (by_n_approx*sq_sum_by_n)/(2**12)
        variance = sq_sum_by_n - 1.0*int((2**6)*(mean**2))/(2**6)
        stddev = sqrootLookUp(variance)

        th =  mean + num_stddev*stddev
        
        print_msg('num_stddev : ' + str(num_stddev),2)
        print_msg('alpha value used in refModel: ' + str(by_n_approx) + ' ,CSR Value : ' + str(int(by_n_approx*64)),2)
        print_msg('score_sum : ' + str(score_sum),2)
        print_msg('score_squared_sum : ' + str(score_squared_sum),2)
        print_msg('by_n_approx : ' + str(by_n_approx),2)
        print_msg('mean : ' + str(mean),2)
        print_msg('sq_sum_by_n : ' + str(sq_sum_by_n),2)
        print_msg('variance : ' + str(variance),2)
        print_msg('stddev : ' + str(stddev),2)
        print_msg('threshold : ' + str(th),2)

        return th

def _get_iou(box, box_register, nms_thresh):
        """ Get the intersection over union of two boxes."""

        xx1 = np.maximum(box[0], box_register[0])
        yy1 = np.maximum(box[1], box_register[1])
        xx2 = np.minimum(box[2], box_register[2])
        yy2 = np.minimum(box[3], box_register[3])
        #KM:  Drop Frac bits for all coordinates
        xx1 = xx1.astype(int)
        xx2 = xx2.astype(int)
        yy1 = yy1.astype(int)
        yy2 = yy2.astype(int)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter  = w*h + 1 # iou_area
        
        area1 = (box.astype(int)[2] - box.astype(int)[0] + 1)*(box.astype(int)[3] - box.astype(int)[1] + 1) + 1
        area2 = ( box_register.astype(int)[2] - box_register.astype(int)[0] + 1)*( box_register.astype(int)[3] - box_register.astype(int)[1] + 1) + 1
        
        union_area = area1 + area2 - inter
        #KM:  Scale nms_thresh and round to match CSR register programming used by HW
        nms_thresh_hw = int(nms_thresh*(2**7)+0.5)
        iou_th_area = (nms_thresh_hw * union_area)/(2**7)
                
        #check for where iou is greater then threshold
        return inter > iou_th_area      

def _filter_boxes(boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

def nms_hw(proposals, scores, num_stddev, nms_thresh, min_size, scale, max_nms_topN, post_nms_topN):

        score_th = _estimate_parameters(scores, num_stddev)

        #KM:  Remove predicted boxes with either height or width < threshold
        #    (NOTE: convert min_size to input image scale stored in im_info[2])
        print_msg('scale: ' + str(scale),0)  # im_info[2]
        print_msg('min_size: ' + str(min_size),0)
        print_msg('max_nms_topN: ' + str(max_nms_topN),1)
        print_msg('post_nms_topN: ' + str(post_nms_topN),1)
        keep = _filter_boxes(proposals, min_size * scale)

        proposals = proposals[keep, :]
        scores = scores[keep]
        
        print_msg('Num total Proposals before NMS : ' + str(proposals.shape),2)

        keep = np.where(scores >= score_th)[0]

        proposals = proposals[keep, :]
        scores = scores[keep]

        print_msg('Threshold for proposal selection : ' + str(score_th),2)
        print_msg('Number of proposals after thresholding : ' + str(len(keep)),2)
        
        box_register = np.ndarray([0,4])                # Hardware linked list 
        score_register = np.ndarray([0])                # Hardware linked list
        num_proposals = len(keep)

        if (num_proposals >= 1):
                box_register = np.insert(box_register, 0, proposals[0,:],0)     # Insert the first one
                score_register = np.insert(score_register, 0, scores[0],0)      # Insert the first one
        
                for idx in range(0, num_proposals):
                        print_msg('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',2)
                        print_msg('Proposal ID: ' + str(idx) + ' -> ' + str(int(8*scores[idx])) + str((16*proposals[idx,:]).astype(int)),2)
                        print_msg('Number of Proposals in Register : ' + str(len(score_register)),2)
                        if (idx < 1):
                                continue

                        insert_idx = 0
                        should_delete = False           # Flag to see if deletion is needed
                        should_ignore = False           # Flag to see if proposal should be ignored     

                        for j in range(len(score_register)):
                                #KM: Use > instead of >= so overlap comparison will determine insertion point if scores are equal
                                score_is_more = scores[idx] > score_register[j]        
                                iou_is_more = _get_iou(proposals[idx,:], box_register[j,:], nms_thresh)                                         
                                if (not score_is_more) and iou_is_more:         
                                        print_msg('Ignoring Proposal | score_is_more : ' + str(score_is_more) + ' iou_is_more : ' + str(iou_is_more),2)
                                        should_ignore = True 
                                        break
                                elif score_is_more:
                                        print_msg('Inserting Point found : ' +str(insert_idx),2)
                                        should_delete = True
                                        break
                                insert_idx += 1 

                        # Insert at the end if insertion point not found
                        if not should_ignore:   
                                print_msg('Inserting Proposal @ ' + str(insert_idx),2)
                                box_register = np.insert(box_register, insert_idx, proposals[idx,:],0)  
                                score_register = np.insert(score_register, insert_idx, scores[idx],0)                   

                        # Trim the register array to the max size of 400 (Hardware Limitation)
                        keepLength = min(len(score_register), max_nms_topN)
                        score_register = score_register[:keepLength]
                        box_register = box_register[:keepLength,:]
                
                        # If we inserted the new proposal, we should delete overlapping ones
                        if should_delete: 
                                k = insert_idx+1
                                end = len(score_register)
                                while k < end: 
                                        iou_is_more = _get_iou(proposals[idx,:], box_register[k,:], nms_thresh)                                         
                                        if iou_is_more:
                                                print_msg('Deleting Proposal @ ' + str(k) + ' -> ' + str(int(8*score_register[k])) + str((16*box_register[k,:]).astype(int)),2)
                                                box_register = np.delete(box_register, k, 0)    
                                                score_register = np.delete(score_register, k, 0)
                                                end -= 1
                                                k -= 1
                                        k += 1

        # Send out the top 300 proposals
        keepLength = min(len(score_register), post_nms_topN)
        score_register = score_register[:keepLength]
        box_register = box_register[:keepLength,:]
        print_msg('Final Number of Boxes after HW NMS : '+str(keepLength),2)
        for idx in range(0,keepLength):
                #print_msg(str(idx) + ' -> ' + str(score_register[idx]) + ' ' + str(box_register[idx,:]),1)
                print_msg(str(idx) + ' -> ' + str(int(8*score_register[idx])) + ' ' + str((16*box_register[idx,:]).astype(int)),1)

        return box_register, score_register
