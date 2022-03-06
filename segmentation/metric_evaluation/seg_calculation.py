import pickle
import numpy as np

from .post_process import create_instance_masks
from .metrics import seg


def calculate_seg(image_path, masks_pickle_path, params=None, thrs=0.5, margin=0):
    with open(image_path, 'rb') as file:
        img_label_pairs_fluo = pickle.load(file)

    with open(masks_pickle_path, 'rb') as file:
        res_masks = pickle.load(file)

    if isinstance(thrs, float):
        thrs = [thrs]

    seg_cur = []
    for thr in thrs:
        instance_masks = create_instance_masks(res_masks, params, thr)
        mean_seg = 0
        for pair, instance_mask in zip(img_label_pairs_fluo, instance_masks):
            _, mask = pair
            if margin > 0:
                mask = mask[margin:-margin, margin:-margin]
                instance_mask = instance_mask[margin:-margin, margin:-margin]
            seg_value = np.mean(seg(mask, instance_mask)[0])
            mean_seg += seg_value
        seg_cur.append(mean_seg / len(instance_masks))

    if len(seg_cur) == 1:
        return seg_cur[0]
    return seg_cur
