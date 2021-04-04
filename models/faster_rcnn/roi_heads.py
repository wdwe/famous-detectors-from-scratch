import torch
import torch.nn as nn
from utils import BoxCoder
from torchvision import ops

class ROIHead(nn.Module):

    def __init__(
            self,
            roi_output_size,
            spatial_scale,
            mlp,
            bbox_head,
            train_roi_per_img = 64,
            train_fg_iou = 0.5,
            train_ignore_iou = 0.1,
            bbox_coder = None
        ):

        self.spatial_scale = spatial_scale
        self.mlp = mlp
        self.bbox_head = bbox_head
        if bbox_coder is None:
            self.bbox_coder = BoxCoder()
        self.train_roi_per_img = train_roi_per_img
        self.train_fg_iou = train_fg_iou
        self.train_ignore_iou = train_ignore_iou

    def assign_anchors_to_targets(self, anchors, targets):
        pass


    def forward(self, features, bboxes, targets = None):
        """
        features: tensors in shape [batch_size, c, h, w] from rcnn feature extractor
        bboxes: a list of batch_size bboxes of shape [n, 4] in format of xyxy
        """
        roi_pool_feats = ops.roi_pool(features, bboxes, self.spatial_scale)
        class_logits, bbox_regress = self.bbox_head(roi_pool_feats)
        if self.training:
            assert targets is not None, "Targets is None in ROIHead forward in training mode."
            

        


        