def collate_fn(batch):
    # split batch from [(img_1, target_1), ..., (img_n, target_n)]]
    # to
    # ((img_1, ..., img_n), (target_1, ..., target_n))
    return tuple(zip(*batch))