import numpy as np
from typing import List, Tuple, Any, Dict


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Вычисление метрики IoU между двумя масками.
    :param mask1: маска бинарная
    :param mask2: маска бинарная
    :return: значение метрики в диапаазоне [0,1]
    """
    mask1 = mask1.astype('uint8')
    mask2 = mask2.astype('uint8')
    inter = (mask1 * mask2).sum()
    union = np.clip(mask1 + mask2 - mask1 * mask2, 0, 1).sum()
    #     union = np.clip(mask1, 0, 1).sum()

    return inter / union


def seg(gt_mask: np.ndarray, seg_mask: np.ndarray, eps: float = 1e-7)\
        -> Tuple[List[int], Dict[Any, int]]:
    """
    Вычисление метрики SEG между двумя масками.

    :param gt_mask: маска в формате uint16,
    где каждому целому значению соответсует свзяная компонента
    :param seg_mask: маска в формате uint16,
    где каждому целому значению соответсует свзяная компонента
    :return: значение метрики в диапазоне [0, 1]
    """
    frame_segs = []
    out = {}
    for obj_num in np.unique(gt_mask):
        if obj_num == 0:
            continue
        obj_gt = gt_mask == obj_num
        mean_seg = 0
        total = 0
        for obj_num2 in np.unique(seg_mask):
            if obj_num2 == 0:
                continue
            obj_seg = seg_mask == obj_num2
            inter = (obj_gt*obj_seg).sum()
            if inter > 0.5 * obj_gt.sum():
                mean_seg += iou(obj_gt, obj_seg)
                total += 1
        mean_seg /= (total + eps)
        out[obj_num] = mean_seg
        frame_segs.append(mean_seg)
    return frame_segs, out
