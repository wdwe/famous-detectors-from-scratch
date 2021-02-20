


if __name__ == "__main__":
    from datasets import CocoDetDataset
    from faster_rcnn.faster_rcnn import FasterRCNN_res50
    import torchvision
    import utils
    from torch.utils.data import DataLoader
    from data_utiils import MetaImages
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    dataset = CocoDetDataset("../datasets/coco2017/", split = "val")
    dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, collate_fn = utils.collate_fn)
    for images, targets, ids in dataloader:
        meta_img = MetaImages(images, targets)
        break