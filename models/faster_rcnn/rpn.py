import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class RPN(nn.Module):
    def __init__(
        self,
        rpn_head,
        anchor,
        train_fg_iou = 0.7,
        train_bg_iou = 0.3,
        train_pos_samples = 128,
        train_neg_samples = 128
    ):
        super().__init__()
        self.rpn_head = rpn_head
        self.anchor = anchor
        self.train_fg_iou = train_fg_iou
        self.train_bg_iou = train_bg_iou
        self.train_pos_samples = train_pos_samples
        self.train_neg_samples = train_neg_samples
        


    def _compute_loss(self, objness, bbox_regress, anchors, labels, matched_bboxes):
        # note anchors are in the format of cxcywh, matched_bboxes in the format of xyxy
        batch_size = len(objness)
        offset_pos_idc = []
        offset_train_idc = []
        
        for i in range(batch_size):
            labels_per_img = labels[i]
            all_pos_idc = torch.flatten((labels_per_img == 1).nonzero())
            all_neg_idc = torch.flatten((labels_per_img == 0).nonzero())
            if len(all_pos_idc) <= self.train_pos_samples:
                pos_idc = all_pos_idc
                extra_neg = self.train_pos_samples - len(all_pos_idc)
            else:
                idc = torch.randperm((len(all_pos_idc)))[:self.train_pos_samples]
                pos_idc = all_pos_idc[idc]
                extra_neg = 0
            idc = torch.randperm((len(all_neg_idc)))[:self.train_neg_samples + extra_neg]
            neg_idc = all_neg_idc[idc]
            train_idc = torch.cat((pos_idc, neg_idc))
            # as the operations are performed in batch
            # offset the idc by i * num_anchors_per_image
            pos_idc = pos_idc + i * objness.shape[1]
            train_idc = train_idc + i * objness.shape[1]
            offset_pos_idc.append(pos_idc)
            offset_train_idc.append(train_idc)
        
        # flatten objness, bbox_regress, anchors and concat labels, match_bboxes, offset_pos_idc, offset_train_idc
        # to perform batched loss computation
        objness = torch.flatten(objness)
        bbox_regress = bbox_regress.view(-1, 4)
        anchors = anchors.view(-1, 4)
        labels = torch.cat(labels)
        matched_bboxes = torch.cat(matched_bboxes)
        offset_pos_idc = torch.cat(offset_pos_idc)
        offset_train_idc = torch.cat(offset_train_idc)

        # compute loss now
        objness_preds = objness[offset_train_idc]
        objness_labels = labels[offset_train_idc]
        objness_loss = F.binary_cross_entropy_with_logits(objness_preds, objness_labels)


            





    def assign_anchors_to_targets(self, anchors, targets):
        """
        Given the anchors (tensor) and targets(list of dict for the image batch) return a list of
        labels of shape number of anchors, and a tensor (matched_bboxes) of the same shape of anchors where
        matched_bboxes[i][j] is the gt_bbox the anchor should regress to if labels[i][j] == 1. Else matched_bboxes
        is [0, 0, 0, 0] which will be ignored when computing loss anyway.
        The anchor box has a corresponding gt_box if its highest iou with the gt_boxes is more than self.train_fg_iou
        or with this anchor box, a gt_box attains the highest iou.
        For the remaining anchor boxes:
        The anchor box has a negative label if the highest iou is less than self.train_bg_iou. The label is -1 if its
        highest iou is inbetween.
        """
        labels = []
        matched_bboxes = []
        for anchors_per_img, targets_per_img in zip(anchors, targets):
            device = anchors_per_img.device
            target_bboxes = targets_per_img["bbox"].to(device)
            print(anchors_per_img.shape)
            if len(target_bboxes) == 0:
                labels_per_img = torch.zeros((len(anchors_per_img), ), dtype = torch.long, device = device)
                matched_bboxes_per_img = torch.zeros_like(anchors_per_img)
            else:
                anchors_per_img = ops.box_convert(anchors_per_img, "cxcywh", "xyxy")
                # anchor_target_bbox_iou is of shape N, M where N, M is the pairwise IoU
                anchor_target_bbox_iou = ops.box_iou(anchors_per_img, target_bboxes)
                labels_per_img = -torch.ones((len(anchors_per_img), ), dtype = torch.long, device=device)
                matched_bboxes_per_img = torch.zeros_like(anchors_per_img, device = device)
                
                # for each gt box, find the anchor boxes with the maximum iou and assign the gt box to the anchor
                anchor_idc = torch.argmax(anchor_target_bbox_iou, dim = 0)
                labels_per_img[anchor_idc] = 1
                matched_bboxes_per_img[anchor_idc] = target_bboxes

                # for each anchor get the gt box with the highest iou
                anchor_max_iou = torch.max(anchor_target_bbox_iou, dim = 1)
                max_iou_gt_box, anchor_max_iou_val = anchor_max_iou.indices, anchor_max_iou.values

                # if the highest iou is more than self.train_fg_iou
                # assign the gt bboxes to the target bboxes
                anchor_idc = torch.flatten(torch.nonzero(anchor_max_iou_val > self.train_fg_iou))
                labels_per_img[anchor_idc] = 1
                matched_bboxes_per_img[anchor_idc] = target_bboxes[max_iou_gt_box[anchor_idc]]

                # if the highest iou is less than self.train_bg_iou
                # set the labels to be 0
                remaining_idc_less = (anchor_max_iou_val < self.train_bg_iou) * (labels_per_img < 0)
                anchor_idc = torch.flatten(torch.nonzero(remaining_idc_less))
                labels_per_img[anchor_idc] = 0

            labels.append(labels_per_img)
            matched_bboxes.append(matched_bboxes_per_img)
        return labels, matched_bboxes
                

    def filter_cross_boundary_anchors(self, labels, anchors, image_size):
        """
        set label to -1 if its corresponding anchors cross the image boundary
        """
        labels = []
        H, W = image_size
        for i, labels_per_image in enumerate(labels):
            anchors_per_img = anchors[i]
            anchors_per_img = ops.box_convert(anchors_per_img, "cxcywh", "xyxy")
            ignore_bool = ((anchors_per_img[:, 0] < 0) + (anchors_per_img[:, 1] < 0) +\
                 (anchors_per_img[:, 2] > W) + (anchors_per_img[:, 3] > H)) > 0
            ignore_idc = torch.nonzero(ignore_bool, as_tuple=True)[0]
            labels_per_image[ignore_idc] = -1
            labels.append(labels_per_image)



    def forward(self, features, targets = None, image_size = (640, 640)):
        # objness is of shape [batchsize, k, feat_h, feat_w]
        # bbox_regress is of shape [batchsize, 4*k, feat_h, feat_w]
        # let the 1st dim of bbox_regress be cx{1}_offset, cy{1}_offset, w{1}_offset, h{1}_offset, ..., w{k}_offset, h{k}_offset
        objness, bbox_regress = self.rpn_head(features)
        # objness is of shape [batchsize, num_anchors, h, w]
        # flatten the objness to [batchsize, k*feat_h*feat_w, 1]
        objness = objness.view((objness.shape[0], -1, 1))
        # bbox_regress is of the shape [batchsize, 4 * num_anchors, feat_h, feat_w]
        shape = bbox_regress.shape
        bbox_regress = bbox_regress.view((shape[0], shape[1]/4, 4, shape[2], shape[3]))
        bbox_regress = bbox_regress.permute(0, 2, 3, 1)
        bbox_regress = bbox_regress.view((shape[0], -1, 4))
        # anchors is in the format of [batchsize, k, feat_h, feat_w, 4]
        anchors = self.anchor(features)
        # flatten the anchors
        anchors = anchors.view((anchors.shape[0], -1, 4))
        if self.training:
            assert targets is not None, "targets is None in RPN forward()"
            labels, matched_bboxes = self.assign_anchors_to_targets(anchors, targets)
            labels = self.filter_cross_boundary_anchors(labels, anchors, image_size)
            loss = self._compute_loss(objness, bbox_regress, anchors, labels, matched_bboxes)
        
            
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


if __name__ == "__main__":
    from datasets import CocoDetDataset

    dataset = CocoDetDataset("../coco_data/", split = "val")