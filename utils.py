import torch
import math

class BoxCoder:
    """
    Given given anchor bbox in [xc, yc, w, h] and targets in xc, yc, w, h
    convert to and from the forms for regression
    """
    def __init__(self, bbox_exp_clip = math.log(100)):
        self.bbox_exp_clip = bbox_exp_clip


    def encode(self, anchors, gt_bboxes):
        """
        anchors and gt_bboxes need to be in the format of xc, yc, w, h
        """

        gt_bboxes[..., 0] = (gt_bboxes[..., 0] - anchors[..., 0]) / anchors[..., 2]
        gt_bboxes[..., 1] = (gt_bboxes[..., 1] - anchors[..., 1]) / anchors[..., 3]
        gt_bboxes[..., 2] = torch.log(gt_bboxes[..., 2] / anchors[..., 2])
        gt_bboxes[..., 3] = torch.log(gt_bboxes[..., 3] / anchors[..., 3])

        return gt_bboxes

    def decode(self, anchors, preds):
        """
        convert pred back to the xc, yc, w, h format
        """

        preds[..., 0] = preds[..., 0] * anchors[..., 2] + anchors[..., 0]
        preds[..., 1] = preds[..., 1] * anchors[..., 3] + anchors[..., 1]
        # clamp pred[..., [2, 3]] to prevent explosion
        preds[..., 2] = torch.clamp(preds[..., 2], max = self.bbox_exp_clip)
        preds[..., 3] = torch.clamp(preds[..., 3], max = self.bbox_exp_clip)
        preds[..., 2] = torch.exp(preds[..., 2]) * anchors[..., 2]
        preds[..., 3] = torch.exp(preds[..., 3]) * anchors[..., 3]

        return preds