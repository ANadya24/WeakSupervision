import numpy as np
import optuna
import os
import json
import pickle
import cv2
from skimage.morphology import erosion, dilation, disk
from segmentation.metric_evaluation.metrics import seg
from segmentation.metric_evaluation.post_process import postprocess_watershed


type_seg = 'DEF/GT/1000/1501_miss/01/'
dataset = 'Fluo-N2DL-HeLa'
image_path = f'../out_masks/{dataset}/01_gt_images.pkl'
n_trials = 500
margin = 50

if type_seg.find('DEF') >= 0:
    mask_path = f'../out_masks/{dataset}/{type_seg}/out.pkl'
    optuna_dir = f'../optuna/{dataset}/{type_seg}/'

else:
    mask_path = f'../out_masks/{dataset}/{type_seg}/out.pkl'
    optuna_dir = f'../optuna/{dataset}/{type_seg}/'

os.makedirs(optuna_dir, exist_ok=True)

with open(image_path, 'rb') as file:
    img_label_pairs_gt = pickle.load(file)

with open(mask_path, 'rb') as file:
    res_masks = pickle.load(file)


def add_items_from_mask(marker, mask):
    n_comps, comps = cv2.connectedComponents(mask)
    for label in range(1, n_comps):
        label_mask = comps == label
        if ((marker>0)*label_mask).sum() < 100:
            tmp = label_mask.astype('float32')
            tmp = dilation(tmp, disk(5))
            tmp[tmp > 0] = 0.99
            marker += np.max(np.stack([tmp, marker], -1), -1)
    return marker


def add_instances_from_mask(instance_mask, mask):
    num_instance = instance_mask.max() + 1
    n_comps, comps = cv2.connectedComponents(mask)
    for label in range(1, n_comps):
        label_mask = comps == label
        if ((instance_mask>0)*label_mask).sum() < 100:
            tmp = label_mask.astype('float')
            tmp = dilation(tmp, disk(5)).astype('uint16')

            tmp[tmp > 0] = num_instance
            num_instance += 1
            instance_mask += tmp.astype(instance_mask.dtype)
    return instance_mask


def objective(trial):
    params = {'invert_markers': False,
              'bad_markers': trial.suggest_categorical('bad_markers', [True, False]),
              'thr': trial.suggest_float('thr', 0., 1.),
              'num_erosion': trial.suggest_int('num_erosion', 1, 20),
              'marker_erosion_seed': None,
              'footprint': trial.suggest_categorical('footprint', np.arange(40, 200).tolist() + [None]),
              'small_holes_param': trial.suggest_int('small_holes_param', 50, 700),
              'small_objects_param': trial.suggest_int('small_objects_param', 200, 800),
              'marker_threshold': trial.suggest_float('marker_threshold', 0., 1.),
              'min_size_label': trial.suggest_int('min_size_label', 1, 500)}

    mean_ious = []
    for pair, res_mask in zip(img_label_pairs_gt, res_masks):
        _, mask = pair
        if margin > 0:
            mask = mask[margin:-margin, margin:-margin]
            res_mask = res_mask[margin:-margin, margin:-margin]
        markers = res_mask[:, :, 1]
        res_mask = res_mask[:, :, 0]
        markers = add_items_from_mask(markers, np.uint8(res_mask > params["thr"]))
        instance_mask = postprocess_watershed(res_mask, markers, **params)
        instance_mask = add_instances_from_mask(instance_mask, np.uint8(res_mask > params["thr"]))
        res_masks.append(instance_mask)
        mean_ious.extend(seg(mask, instance_mask)[0])

    mean_iou = np.mean(mean_ious)
    return mean_iou


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, n_jobs=4)

print(study.best_params)
with open(optuna_dir + 'best_params.json', 'w') as file:
    json.dump(study.best_params, file)

with open(optuna_dir + 'study.pkl', 'wb') as file:
    pickle.dump(study, file)