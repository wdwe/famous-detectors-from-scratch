


if __name__ == "__main__":
    from datasets import CocoDetDataset
    from models.faster_rcnn.faster_rcnn import FasterRCNN_res50
    from models.faster_rcnn.rpn import RPN, RPNHead
    from models.utils import Anchor
    import torchvision
    import utils
    from torch.utils.data import DataLoader
    from data_utiils import MetaImages, collate_fn
    BATCH_SIZE = 2
    NUM_WORKERS = 4

    dataset = CocoDetDataset("../coco_data/coco2017/", split = "val")
    dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, collate_fn = collate_fn)
    model = FasterRCNN_res50()
    scales = [128, 256, 512]
    ratios = [1, 2, 0.5]
    anchor = Anchor(scales, ratios, model.stride)
    rpn_head = RPNHead(model.feature_map_planes, len(scales)*len(ratios))
    rpn = RPN(rpn_head, anchor)
    rpn.cuda()
    model.cuda()
    for images, targets, ids in dataloader:
        meta_img = MetaImages(images, targets)
        meta_img.cuda()
        features = model(meta_img)
        objness, bbox_regress, anchors = rpn(features)
        print(objness.shape)
        print(bbox_regress.shape)
        print(anchors.shape)
        print(anchors[:, 0, 0])
        break