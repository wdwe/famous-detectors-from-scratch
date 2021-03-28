import torch
import torch.nn as nn
import torchvision


class FasterRCNN_vgg16_bn(nn.Module):
    stride = 16
    feature_map_planes = 512
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16_bn(pretrained = True)
        self.backbone = nn.ModuleList(*list(vgg16.features.children())[:-1])

    def extract_features(self, images):
        for m in self.backbone:
            images = m(images)
        return images

    def forward(self, meta_img):
        images = meta_img.images
        target = meta_img.targets
        features = self.backbone(images)
        
        return features

