import torch
from torchvision.transforms import functional as F

class MetaImages:
    """
    This clas helps with resizing both images and targets (bbox).
    It fills the images to the area defined by size while retain the aspect ratio.
    The unfilled area is padded with zeros.

    Args:
        images: a list of Tensor
        targets: a list of None or dict where targets[i]["bbox"] should be a Tensor of shape [batchsize, 4] in (xmin, ymin, xmax, ymax) format

    """
    def __init__(self, images, targets = None, size = (640, 640)):
        self.size = size
        self.orig_sizes = []
        for img in images:
            self.orig_sizes.append(img.shape)
        if targets is None:
            targets = [None] * len(images)
        self._transform(images, targets)
        if not self.targets:
            self.targets = None

    def _transform(self, images, targets):
        self.images = []
        self.scales = []
        self.targets = []
        for i, image in enumerate(images):
            image, scale = self._trans_img(image)
            self.scales.append(scale)
            self.images.append(image)
            if targets is not None:
                target = self._trans_target(targets[i], scale)
                self.targets.append(target)
        self.images = torch.stack(self.images)

    def _trans_img(self, image):
        _, h, w = image.shape
        # compute size
        H, W = self.size
        scale = int(min(H/h, W/w))
        h *= scale
        w *= scale
        size = (h, w)
        image =F.resize(image, size)
        # pad to right or left
        image = F.pad(image, (0, 0, W - w, H - h))

        return image, scale

    def _trans_target(self, target, scale):
        target["bbox"] *= scale
        return target

    def to(self, device):
        self.images = self.images.to(device)
        if self.targets is not None:
            self.targets = [
                {k: v.to(device) for k, v in target.items()} for target in self.targets
            ]

    def cuda(self, num = 0):
        device = torch.device(f"cuda: {num}")
        self.to(device)