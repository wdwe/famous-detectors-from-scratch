import torch
import torch.nn as nn
from itertools import product


class Anchor(nn.Module):
    def __init__(
        self,
        scales,
        ratios,
        stride,
    ):
        super().__init__()
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.base_anchors = self.gen_base_anchor(self.scales, self.ratios)
        self._cache = {}

    def gen_base_anchor(self, scales, ratios):
        anchor_dims = []
        for s, r in product(scales, ratios):
            w = s * r
            h = s / r
            anchor_dims.append([0, 0, w, h])
        return torch.tensor(anchor_dims)

    def gen_grid(self, feat_h, feat_w, stride):
        """
        Args:
            feat_h (int): height of the feature map.
            feat_w (int): width of the feature map.
            stride (int): the receptive field on the original image for each pixel on the feature map

        Returns:
            grid (tensor): tensor of shape (feat_h, feat_w, 2) where the last dim contains the center
                location of the anchor on the original image.
        """

        y_ = torch.arange(feat_h) * stride
        x_ = torch.arange(feat_w) * stride
        yy, xx = torch.meshgrid(y_, x_)
        grid = torch.stack((xx, yy), dim = -1)
        grid = grid + stride / 2
        return grid

    def get_anchors(self, features, stride):
        feat_h, feat_w = tuple(features.shape)[-2:]
        device = features.device
        anchor_key = (feat_h, feat_w, stride, device)
        if anchor_key not in self._cache:
            grid = self.gen_grid(feat_h, feat_w, stride)
            pad = torch.zeros_like(grid)
            grid = torch.cat((grid, pad), dim = -1)
            anchors = grid[..., None, :] + self.base_anchors
            
            anchors = anchors.to(features.device)
            anchors = anchors.permute((2, 3, 0, 1))
            self._cache[anchor_key] = anchors
        return self._cache[anchor_key]

    def forward(self, features):
        batch_size = features.shape[0]
        anchors = self.get_anchors(features, self.stride)
        anchors = torch.stack([anchors] * batch_size, dim = 0)
        return anchors
