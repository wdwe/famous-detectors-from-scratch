import torch
import torch.nn as nn

class RPN(nn.Module):
    def __init__(
        self,
        rpn_head,
        anchor
    ):
        super().__init__()
        self.rpn_head = rpn_head
        self.anchor = anchor

    def _compute_loss(self, objness, bbox_regress, anchors, targets):
        batch_size = len(objness)
        for i in range(batch_size):




    def forward(self, features, targets = None):
        # objness is of shape [batchsize, 1, feat_h, feat_w]
        # bbox_regress is of shape [batchsize, 4*k, feat_h, feat_w]
        # let the 1st dim of bbox_regress be cx{1}_offset, cy{1}_offset, w{1}_offset, h{1}_offset, ..., w{k}_offset, h{k}_offset
        objness, bbox_regress = self.rpn_head(features)
        # anchors is in the format of [batchsize, k, 4, feat_h, feat_w]
        anchors = self.anchor(features)
        if self.training:
            loss = self._compute_loss(objness, bbox_regress, anchors, targets)
        
            
        return objness, bbox_regress, anchors



class RPNHead(nn.Module):
    def __init__(self, inplanes, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size = 3, stride = 1, padding = 1)
        self.cls_logits = nn.Conv2d(inplanes, num_anchors, kernel_size = 1, stride = 1, padding = 0)
        self.bbox_regress = nn.Conv2d(inplanes, 4 * num_anchors, kernel_size = 1, stride = 1, padding = 0)
    
    def forward(self, features):
        features = self.conv(features)
        logits = self.cls_logits(features)
        bbox_regress = self.bbox_regress(features)

        return logits, bbox_regress