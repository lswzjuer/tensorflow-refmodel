import numpy as np


def calculate_intersection_and_areas(bboxes1, bboxes2):
    bc_bboxes1 = bboxes1[:, np.newaxis]
    bc_bboxes2 = bboxes2[np.newaxis]
    inter_upleft = np.maximum(bc_bboxes1[:, :, :2], bc_bboxes2[:, :, :2])
    inter_botright = np.minimum(bc_bboxes1[:, :, 2:4], bc_bboxes2[:, :, 2:4])
    inter_wh = np.maximum(inter_botright - inter_upleft, 0)
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    return inter, area1, area2


def intersection_over_min_area(bboxes1, bboxes2):
    inter, area1, area2 = calculate_intersection_and_areas(bboxes1, bboxes2)
    min_area = np.minimum(area1[:, np.newaxis], area2[np.newaxis])
    zero_idx = (min_area == 0)
    min_area[zero_idx] = 1
    iou = inter / min_area
    if np.any(zero_idx):
        iou[zero_idx] = 0
    return iou


def non_max_suppression(bboxes, scores, nms_params):
    if bboxes.shape[0] == 0:
        return []

    overlap_thresh = nms_params.get("thr", 0.7)
    top_k = nms_params.get("top_k", 125)
    mode = nms_params.get("mode", "max")

    picks = []
    idxs = np.argsort(scores)
    if mode == "min":
        idxs = idxs[::-1]

    while len(idxs) > 0 and len(picks) < top_k:
        last = len(idxs) - 1
        pick = idxs[last]
        picks.append(pick)
        overlap = intersection_over_min_area(bboxes[[pick]], bboxes[idxs[:last]])
        above_thresh = np.where(overlap > overlap_thresh)[1]
        idxs = np.delete(idxs, np.concatenate(([last], above_thresh)))
    return picks


def rescale_bboxes(bboxes, input_shape, output_shape):
    w_fact = float(output_shape[1]) / float(input_shape[1])
    h_fact = float(output_shape[0]) / float(input_shape[0])
    bboxes[:, 0] *= w_fact
    bboxes[:, 2] *= w_fact
    bboxes[:, 1] *= h_fact
    bboxes[:, 3] *= h_fact
    return bboxes


def get_bboxes(net_output, anchors, nms_params):
    """
    Decode the network output as bounding boxes in the format (left_x, top_y, width, height) w.r.t. the original image
    and the respective score for the pedestrian class.
    -------

    """
    score_threshold = 0.001  # Background score threshold, higher score rejects more background boxes
    cls_idx = [4, 5]  # Background_idx, Pedestrian_idx

    # Extract scores
    scores = net_output[:, cls_idx[1]]

    # Convert net_output to bounding boxes in CCWH encoding [cx,cy, w,h]
    bboxes = np.zeros((len(net_output), 4))
    bboxes[:, 0:2] = net_output[:, 0:2] * anchors[:, 2:4] + anchors[:, 0:2]
    bboxes[:, 2:4] = np.exp(net_output[:, 2:4]) * anchors[:, 2:4]

    # Convert bounding boxes into IMAGE encoding [x1,y1, x2,y2]
    bboxes[:, 0:2] = bboxes[:, 0:2] - bboxes[:, 2:4] * 0.5
    bboxes[:, 2:4] = bboxes[:, 0:2] + bboxes[:, 2:4]

    # Perform score thresholding
    idx = scores >= score_threshold
    bboxes = bboxes[idx, 0:4]
    scores = scores[idx]

    # Perform NMS
    nms_idx = non_max_suppression(bboxes, scores, nms_params)

    # Convert bboxes to LTWH (left_x, top_y, w, h)
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    bboxes_nms = bboxes[nms_idx]
    scores = scores[nms_idx]
    return bboxes_nms, scores


def load_anchors(anchor_file):
    return np.load(anchor_file)
