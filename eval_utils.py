from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import json
import os
import os.path as osp
import utils
from tqdm import tqdm

class CocoDetEval:
    def __init__(self, ann_file):
        self.coco_gt = COCO(ann_file)
        self.coco_det = None
        self.coco_eval = None
        self.ann_type = "bbox"
    
    @torch.no_grad()
    def evaluate_model(self, model, dataloader, save_dir = None, save_name = "results.json"):
        results = []
        model.eval()
        device = next(model.parameters()).device
        for images, _, img_ids in tqdm(dataloader):
            images = [image.to(device) for image in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                img_id = img_ids[i]
                bboxes, labels, scores = output["boxes"], output["labels"], output["scores"]
                # convert to [x, y, width, height]
                bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
                bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
                for bbox, label, score in zip(bboxes, labels, scores):
                    bbox = bbox.cpu().numpy().tolist()
                    result = {
                        "image_id": img_id,
                        "bbox": bbox,
                        "category_id": label.item(),
                        "score": score.item()
                    }
                    results.append(result)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        save_file = osp.join(save_dir, save_name)
        with open(save_file, "w") as fi:
            json.dump(results, fi)

        return self.evaluate_results(results)


    def evaluate_results(self, results):

        self.coco_det = self.coco_gt.loadRes(results)
        self.coco_eval = COCOeval(self.coco_gt, self.coco_det, self.ann_type)
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        eval_results = self.coco_eval.stats

        return eval_results


if __name__ == "__main__":
    from datasets import CocoDetDataset
    import torchvision
    import data_utils
    from torch.utils.data import DataLoader
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    dataset = CocoDetDataset("../coco_data/", split = "val")
    dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, collate_fn = data_utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    model.cuda()
    coco_eval = CocoDetEval("../coco_data/annotations/instances_val2017.json")
    # coco_eval.evaluate_model(model, dataloader, "tests", "try.json")
    # print(coco_eval.evaluate_results("./tests/try.json"))