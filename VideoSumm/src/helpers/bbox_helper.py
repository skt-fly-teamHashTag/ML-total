from itertools import groupby
from operator import itemgetter
from typing import Tuple

import torch 
import numpy as np


def lr2cw(bbox_lr: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from left-right (LR) to center-width (CW) format.

    :param bbox_lr: LR bounding boxes. Sized [N, 2].
    :return: CW bounding boxes. Sized [N, 2].
    """
    bbox_lr = np.asarray(bbox_lr, dtype=np.float32).reshape((-1, 2))
    center = (bbox_lr[:, 0] + bbox_lr[:, 1]) / 2
    width = bbox_lr[:, 1] - bbox_lr[:, 0]
    bbox_cw = np.vstack((center, width)).T
    return bbox_cw


def cw2lr(bbox_cw: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center-width (CW) to left-right (LR) format.

    :param bbox_cw: CW bounding boxes. Sized [N, 2].
    :return: LR bounding boxes. Sized [N, 2].
    """
    bbox_cw = np.asarray(bbox_cw, dtype=np.float32).reshape((-1, 2))
    left = bbox_cw[:, 0] - bbox_cw[:, 1] / 2
    right = bbox_cw[:, 0] + bbox_cw[:, 1] / 2
    bbox_lr = np.vstack((left, right)).T
    return bbox_lr



def seq2bbox(sequence: np.ndarray) -> np.ndarray:
    """Generate CW bbox from binary sequence mask
    : seq 배열의 1인 원소의 연속된 프레임 번호를 박스로 묶음 
    """
    
    sequence = np.asarray(sequence, dtype=np.bool)
    selected_indices, = np.where(sequence == 1) #1인 원소의 인덱스 리턴 

    bboxes_lr = []
    for k, g in groupby(enumerate(selected_indices), lambda x: x[0] - x[1]):
        segment = list(map(itemgetter(1), g)) #selected_indices의 원소(sequence ==1인 인덱스)가 담김, 
                                              #groupby 함수를 통해 1이 연속된 프레임의 인덱스가 segment로 담긴다 !
        start_frame, end_frame = segment[0], segment[-1] + 1
        bboxes_lr.append([start_frame, end_frame])

    bboxes_lr = np.asarray(bboxes_lr, dtype=np.int32)
    return bboxes_lr


def iou_lr(anchor_bbox: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
    """Compute iou between multiple LR bbox pairs.

    :param anchor_bbox: LR anchor bboxes. Sized [N, 2].
    :param target_bbox: LR target bboxes. Sized [N, 2].
    :return: IoU between each bbox pair. Sized [N].
    """
    anchor_left, anchor_right = anchor_bbox[:, 0], anchor_bbox[:, 1]
    target_left, target_right = target_bbox[:, 0], target_bbox[:, 1]

    inter_left = np.maximum(anchor_left, target_left)
    inter_right = np.minimum(anchor_right, target_right)
    union_left = np.minimum(anchor_left, target_left)
    union_right = np.maximum(anchor_right, target_right)

    intersect = inter_right - inter_left
    intersect[intersect < 0] = 0
    union = union_right - union_left
    union[union <= 0] = 1e-6

    iou = intersect / union
    return iou


def iou_cw(anchor_bbox: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
    """Compute iou between multiple CW bbox pairs. See ``iou_lr``"""
    anchor_bbox_lr = cw2lr(anchor_bbox)
    target_bbox_lr = cw2lr(target_bbox)
    return iou_lr(anchor_bbox_lr, target_bbox_lr)


def nms(scores: np.ndarray,
        bboxes: np.ndarray,
        thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Non-Maximum Suppression (
    ) algorithm on 1D bbox.

    :param scores: List of confidence scores for bboxes. Sized [N].
    :param bboxes: List of LR bboxes. Sized [N, 2].
    :param thresh: IoU threshold. Overlapped bboxes with IoU higher than this
        threshold will be filtered.
    :return: Remaining bboxes and its scores.
    """
    valid_idx = bboxes[:, 0] < bboxes[:, 1]
    scores = scores[valid_idx]
    bboxes = bboxes[valid_idx]

    arg_desc = scores.argsort()[::-1]

    scores_remain = scores[arg_desc]
    bboxes_remain = bboxes[arg_desc]

    keep_bboxes = []
    keep_scores = []

    while bboxes_remain.size > 0:
        bbox = bboxes_remain[0]
        score = scores_remain[0]
        keep_bboxes.append(bbox)
        keep_scores.append(score)

        iou = iou_lr(bboxes_remain, np.expand_dims(bbox, axis=0))

        keep_indices = (iou < thresh)
        bboxes_remain = bboxes_remain[keep_indices]
        scores_remain = scores_remain[keep_indices]

    keep_bboxes = np.asarray(keep_bboxes, dtype=bboxes.dtype)
    keep_scores = np.asarray(keep_scores, dtype=scores.dtype)

    return keep_scores, keep_bboxes


## iou가 일정 값 이상일 때, 0으로 억제하는 것이 아닌 score를 감소(가우시안 weight 곱해줌 )
def soft_nms(scores: np.ndarray,
            bboxes: np.ndarray,
            sigma: float, # gaussian weight decay (default= 0.5)
            thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Non-Maximum Suppression (
    ) algorithm on 1D bbox.

    :param scores: List of confidence scores for bboxes. Sized [N].
    :param bboxes: List of LR bboxes. Sized [N, 2].
    :param thresh: IoU threshold. Overlapped bboxes with IoU higher than this
        threshold will be filtered.
    :return: Remaining bboxes and its scores.
    """
    N = bboxes.shape[0]

    valid_idx = bboxes[:, 0] < bboxes[:, 1] 
    scores = scores[valid_idx] 
    bboxes = bboxes[valid_idx] 

    x1 = bboxes[:, 0]
    x2 = bboxes[:, 1]
    areas = (x2 - x1 + 1)

    for i in range(N):
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                bboxes[i], bboxes[maxpos.item() + i + 1] = bboxes[maxpos.item() + i + 1].clone(), bboxes[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # iou calculate
        iou = iou_lr(bboxes, np.expand_dims(bboxes[i], axis=0))

        # Gaussian dacay 
        weight = torch.exp(-(iou * iou) / sigma) 

        keep_indices = (iou < thresh)
        scores[keep_indices] *= weight


    return scores, bboxes



'''
def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[x1, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [x1, x2]
    x1 = dets[:, 0]
    x2 = dets[:, 1]

    scores = box_scores
    areas = (x2 - x1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter)) #iou 

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:] 

    # select the boxes and keep the corresponding indexes 
    keep = dets[:, 4][scores > thresh].int()

    return keep
'''
