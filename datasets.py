from pycocotools.coco import COCO
import torchvision.transforms as T
import os
import os.path as osp
from PIL import Image
import torch

class CocoDetDataset:
    def __init__(self, root, split = "train", transforms = None, img_transforms = None, target_transforms = None):
        assert split in ("train", "val"), "split can only be train or val."
        self.root = root
        self.img_root = osp.join(self.root, f"{split}2017")
        self.transforms = transforms
        self.img_transforms = img_transforms
        self.target_transforms = target_transforms
        ann_file = f"{self.root}/annotations/instances_{split}2017.json"
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _collate_coco_anns(self, anns):
        target = {}
        bboxes = []
        labels = []
        for ann in anns:
            bboxes.append(ann["bbox"])
            labels.append(ann["category_id"])
        bboxes = torch.tensor(bboxes)
        labels = torch.tensor(labels)
        # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax)
        if len(labels) > 0:
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        else:
            print(labels)
            print(bboxes)
        target["bbox"] = bboxes
        target["labels"] = labels
        return target

    def __getitem__(self, idx):
        
        img_id = self.ids[idx]
        img_file = self.coco.imgs[img_id]["file_name"]
        img_file = osp.join(self.img_root, img_file)
        img = Image.open(img_file).convert("RGB")

        ann_id = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_id)
        target = self._collate_coco_anns(anns)

        if self.img_transforms is not None:
            img = self.img_transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if not isinstance(img, torch.Tensor):
            img = T.functional.to_tensor(img)

        return img, target, img_id

    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    dataset = CocoDetDataset("../datasets/coco2017", "val")
    min_label = 20
    for i in range(len(dataset)):
        _, target, _ = dataset[i]
        if len(target["labels"]) > 0 and min(target["labels"]) < min_label:
            min_label = min(target["labels"])
    print(min_label)