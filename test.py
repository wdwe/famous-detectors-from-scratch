


if __name__ == "__main__":
    # import torch
    # torch.set_printoptions(edgeitems=100000000)
    from datasets import CocoDetDataset
    from models.faster_rcnn.faster_rcnn import FasterRCNN_vgg16_bn
    from models.faster_rcnn.rpn import RPN, RPNHead
    from models.utils import Anchor
    import torchvision
    import utils
    from torch.utils.data import DataLoader
    from data_utils import MetaImages, collate_fn
    BATCH_SIZE = 2
    NUM_WORKERS = 4

    dataset = CocoDetDataset("../datasets/coco2017", split = "val")
    dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, collate_fn = collate_fn)
    model = FasterRCNN_vgg16_bn()
    scales = [128, 256, 512]
    ratios = [1, 2, 0.5]
    anchor = Anchor(scales, ratios, model.stride)
    rpn_head = RPNHead(model.feature_map_planes, len(scales)*len(ratios))
    rpn = RPN(rpn_head, anchor, train_fg_iou = 0.7)
    # rpn.cuda()
    # model.cuda()
    for i, (images, targets, ids) in enumerate(dataloader):
        meta_img = MetaImages(images, targets)
        # meta_img.cuda()
        features = model(meta_img)
        bboxes, loss = rpn(features, targets, (640, 640))
        print(loss)
        for bbox in bboxes:
            print(bbox.shape)
        
