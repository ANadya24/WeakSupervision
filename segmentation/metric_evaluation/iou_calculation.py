import pickle
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from .metrics import iou


def calculate_iou(image_path, masks_pickle_path, params=None, margin=0):
    with open(image_path, 'rb') as file:
        img_label_pairs_fluo = pickle.load(file)

    iou_cur = []
    with open(masks_pickle_path, 'rb') as file:
        res_masks = pickle.load(file)

    for thr in np.arange(0.1, 0.9, 0.05):
        mean_iou = 0
        for pair, res_mask in zip(img_label_pairs_fluo, res_masks):
            _, mask = pair
            if margin > 0:
                mask = mask[margin:-margin, margin:-margin]
                res_mask = res_mask[margin:-margin, margin:-margin]

            mask = mask > 0

            if params is None:
                mean_iou += iou(mask, res_mask[:, :, 0] > thr)
            else:
                mean_iou += iou(mask, remove_small_objects(
                    remove_small_holes(res_mask[:, :, 0] > thr, params['small_holes_param']),
                    min_size=params['small_objects_param']))

        mean_iou /= len(res_masks)
        iou_cur.append(mean_iou)
    return iou_cur, np.arange(0.1, 0.9, 0.05)
