import torch
import torch.nn as nn
import torchvision


class FasterRCNN_res50(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(pretrained = True)
        self.backbone = nn.ModuleList(
            [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4]
        )

    def extract_features(self, images):
        for m in self.backbone:
            images = m(images)
        return images

    def forward(self, meta_img):
        images = meta_img.images
        target = meta_img.targets
        features = self.extract_features(images)
        
        return features

