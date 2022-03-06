import torch
from torchvision.transforms import ToTensor
from torch.utils import data

from albumentations import Crop, RandomResizedCrop, CenterCrop, MaskDropout, CropNonEmptyMaskIfExists
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    ElasticTransform, ToFloat,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,

)
from albumentations.augmentations.transforms import MultiplicativeNoise

import numpy as np

from segmentation.data_manipulation.utils import normalize


def train_aug(p=1, center_crop_size=480):
    return Compose([
        RandomRotate90(),
        Flip(-1),
        Transpose(),
        OneOf([
            Blur(blur_limit=7, p=0.5),
            MotionBlur(blur_limit=7, p=1),
        ], p=0.7),
        MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=1),
        MultiplicativeNoise(multiplier=(0.5, 1.5), elementwise=False, p=0.8),
        OneOf([
            ShiftScaleRotate(shift_limit=0.125, scale_limit=0.3, rotate_limit=90, p=0.5),
            ElasticTransform(alpha=2, sigma=50, alpha_affine=25, p=0.5),
        ], p=0.8),
        OneOf([
            OpticalDistortion(distort_limit=0.2, p=0.5),
            GridDistortion(distort_limit=0.2, p=0.5)
        ], p=0.75),
        RandomResizedCrop(center_crop_size, center_crop_size, always_apply=True)
    ], p=p)


def val_aug(p=0.5, center_crop_size=480):
    return Compose([
        RandomRotate90(),
        Flip(-1),
        Transpose(),
        CenterCrop(center_crop_size, center_crop_size, True)
    ], p=p)


def test_aug(p=1.0, center_crop_size=480):
    return Compose([
        CenterCrop(center_crop_size, center_crop_size, True),
        MotionBlur(blur_limit=9, p=1)
    ], p=p)


class SegmentationDataset(data.Dataset):
    def __init__(self, x_set, y_set_mask_and_markers, y_gt_markers,
                 n_classes, augmentations, center_crop_size=480,
                 neighbour_count=4, weights=None,
                 weight_type="deform",
                 map_weights=[1.3, 0.5, 1.5, 3.]):
        self.x, self.y = x_set, y_set_mask_and_markers
        self.n_classes = n_classes
        self.augment = augmentations
        self.gt_markers = y_gt_markers
        self.tensor = ToTensor()
        self.neighbour_count = neighbour_count
        self.center_crop_size = center_crop_size
        self.weight_type = weight_type
        self.weights = weights
        self.map_weights = map_weights

    #         self.crop = CropNonEmptyMaskIfExists(center_crop_size,center_crop_size,p=1.0)

    def __len__(self):
        return len(self.x)

    def _create_weight_map(self, mask, gt_marker):
        weight = mask.copy()
        weight[weight > 0] = 1.
        weight[weight != 1] = gt_marker
        return weight

    def _set_weights_by_map(self, weight):
        weight_map = np.ones(weight.shape).astype('float32')
        weight_map[weight == 0] = self.map_weights[0]
        weight_map[weight == 1] = self.map_weights[1]
        weight_map[weight == 2] = self.map_weights[2]
        weight_map[weight == 3] = self.map_weights[3]
        return weight_map

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        gt_marker = self.gt_markers[idx]

        if gt_marker == 0:
            gt_marker = 1 / self.neighbour_count
            
        if self.weight_type in ["deform", "custom"]:
            weight = self.weights[idx]
            if self.weight_type == "deform":
                weight[:,:,0] = self._set_weights_by_map(weight[:,:,0])
            elif weight is None:
                weight = np.zeros(y.shape[:2] + (1,), dtype=np.float32)
            y = np.concatenate([y, weight], 2)
        aug = self.augment(image=x, mask=y)
        if aug['mask'].sum() == 0:
            aug = val_aug(center_crop_size=self.center_crop_size)(image=x, mask=y)
            x = aug['image']
            y = aug['mask']
        else:
            x = aug['image']
            y = aug['mask']

        gt_marker = float(gt_marker)

        if self.weight_type in ["deform", "custom"]:
            borders_weight = y[:, :, -1]
            if self.weight_type == "deform":
                weight = y[:, :, -2]
                y = y[:, :, :-2]
            else:
                y = y[:, :, :-1]
                weight = self._create_weight_map(y[:, :, 0], gt_marker)
            weight += borders_weight
            weight = np.clip(weight, 0, borders_weight.max())
            weight = torch.Tensor(weight).float()[None]
        elif self.weight_type == "digit":
            weight = gt_marker
        else:
            weight = 1.
        x = normalize(x)
        x = torch.Tensor(x.transpose((2, 0, 1)))
        y = torch.Tensor(y.transpose((2, 0, 1))).long()
        return x, y, weight
