import cv2
import time
from skimage.morphology import (
    square, erosion,
    remove_small_holes
)
import imutils
from skimage.color import gray2rgb
from skimage import img_as_ubyte
from skimage import io
import os
import numpy as np
import segmentation_models_pytorch as sm
from torch.optim import lr_scheduler
from torch import nn
import torch
from torch.utils import data

from dataset import SegmentationDataset, train_aug, val_aug
from loss import (
    FocalLoss,
    BCEWeightsDigitLoss,
    BCEWeightsMapLoss,
)
from train import train_model
from registration.src.losses import dice_loss
from utils import preprocess_label

size_im = {'PhC-C2DH-U373': (520, 696), 'Fluo-N2DH-SIM+': (690, 628), 'Fluo-N2DL-HeLa': (600, 1000)}
center_crop_size = 256
batch_size = 8
torch.manual_seed(233565)
num_seq = 1
margin = 50
device = 'cuda:2'
save_prefix = "_markers"
use_weights = "deform"  # one of none, deform, custom, digit
num_erosion = 5
erosion_elem = square(3)
marker_type = 'erosion'  # erosion or circle
seg_type = 'GT'

for dataset in ['Fluo-N2DL-HeLa']:
    for neighbour_count in [1000]:
        print('Dataset is', dataset)
        print('neighbour_count is', neighbour_count)
        path = f'../data/{dataset}/{num_seq:02d}/'
        label_path = f'../data/{dataset}/{num_seq:02d}_DEF{save_prefix}/{seg_type}/FRAMES/{neighbour_count}/'
        weights_path = label_path.replace('FRAMES', 'WEIGHTS')
        gt_label_path = f'../data/{dataset}/{num_seq:02d}_GT/SEG/'
        result_path = f'./results/{dataset}/DEF/{seg_type}/{neighbour_count}/1201{save_prefix}/'
        os.makedirs(result_path, exist_ok=True)

        full_fluo_paths = [path]
        fluo_label_paths = [label_path]
        img_label_pairs_fluo = []
        n_count_seq = {}

        for im_fluo_path, mask_fluo_path in zip(full_fluo_paths, fluo_label_paths):
            seq_tag = im_fluo_path.split('/')[2]
            count = 0
            for r, d, f in os.walk(mask_fluo_path):
                for file in f:
                    if not file.endswith('.tif'):
                        continue
                    if file.startswith('man_seg'):  # check pairs
                        t = file.split('.tif')[0].split('man_seg')[1]
                        mask_path = os.path.join(label_path, file)
                        marker = 0
                        if os.path.exists(os.path.join(gt_label_path, file)):
                            marker = 1
                        t_full = int(t)
                        img_path = os.path.join(im_fluo_path, 't' + str(t_full).zfill(3) + '.tif')
                        img = io.imread(img_path)[margin:-margin, margin:-margin]
                        label = io.imread(mask_path)[margin:-margin, margin:-margin]
                        weight_path = os.path.join(weights_path, 'w' + str(t_full).zfill(3) + '.tif')
                        weight = io.imread(weight_path)[margin:-margin, margin:-margin]
                        if seg_type == 'GT':
                            if (label > 0).sum() / label.size < 0.009:
                                continue
                        if (label > 0).sum() / label.size < 0.02:
                            marker = 0
                        if use_weights == 'deform':
                            pair = (img, label, marker, weight)
                        else:
                            pair = (img, label, marker, None)
                        img_label_pairs_fluo.append(pair)
                        count = count + 1
            n_count_seq[seq_tag] = count

        print('Number of fluo data: {}'.format(len(img_label_pairs_fluo)))
        n_count_seq['conf'] = 1
        data_pairs = img_label_pairs_fluo

        # now prepare masks for watershed segmentation

        binary_cell_masks = []
        binary_cell_markers = []
        gt_labels = []
        weights = []
        stacks = []

        for img, label, marker, weight in data_pairs:

            if marker_type == 'erosion':
                markers = np.zeros(label.shape)
                for num in np.unique(label):
                    if num == 0:
                        continue
                    tmp = label == num
                    for i in range(num_erosion):
                        tmp = erosion(tmp, erosion_elem)
                    markers += tmp

            else:
                image = img_as_ubyte(binary_mask)
                cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                markers = np.zeros(image.shape)
                for c in cnts:
                    # compute the center of the contour
                    M = cv2.moments(c)
                    if M["m00"] == 0.:
                        continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(markers, (cX, cY), 3, (1, 1, 1), -1)

            binary_mask = remove_small_holes(preprocess_label(label), 10)
            binary_cell_masks.append(binary_mask)
            binary_cell_markers.append(markers)
            stacks.append(img)
            gt_labels.append(marker)
            if weight is not None:
                weights.append(weight)

        all_ids = range(0, len(binary_cell_masks))
        val_ids = np.random.choice(all_ids, int(0.1 * (len(binary_cell_masks))),
                                   replace=False)  # take 13 seq and 2 images from confocal
        train_ids = [i for i in all_ids if i not in val_ids]

        kh, kw = size_im[dataset]
        X_train = np.empty((len(train_ids), kh, kw, 3))
        y_train = np.empty((len(train_ids), kh, kw, 2))
        y_train_weights = np.empty((len(train_ids), kh, kw, 1))
        y_gt_train = np.empty((len(train_ids)), dtype='float32')

        X_val = np.empty((len(val_ids), kh, kw, 3))
        y_val = np.empty((len(val_ids), kh, kw, 2))
        y_val_weights = np.empty((len(val_ids), kh, kw, 1))
        y_gt_val = np.empty((len(train_ids)), dtype='float32')

        i_train = 0
        i_val = 0
        for i, (stack, mask, markers) in enumerate(zip(stacks, binary_cell_masks, binary_cell_markers)):
            if mask.sum() == 0:
                print('zero mask')
                continue

            if i in val_ids:
                X_val[i_val] = gray2rgb(stack)
                y_val[i_val, ..., 0] = mask
                y_val[i_val, ..., 1] = markers
                y_gt_val[i_val] = gt_labels[i]
                if len(weights) > 0:
                    y_val_weights[i_val, ..., 0] = weights[i]
                i_val = i_val + 1
            else:
                X_train[i_train] = gray2rgb(stack)
                y_train[i_train, ..., 0] = mask
                y_train[i_train, ..., 1] = markers
                y_gt_train[i_train] = gt_labels[i]
                if len(weights) > 0:
                    y_train_weights[i_train, ..., 0] = weights[i]
                i_train = i_train + 1

        train_dataset = SegmentationDataset(X_train, y_train[..., 0:2], y_gt_train, n_classes=2,
                                            augmentations=train_aug(center_crop_size=center_crop_size),
                                            center_crop_size=center_crop_size,
                                            neighbour_count=min(neighbour_count, 4),
                                            weights=y_train_weights,
                                            weight_type=use_weights,
                                            )
        val_dataset = SegmentationDataset(X_val, y_val[..., 0:2], y_gt_val, n_classes=2,
                                          augmentations=val_aug(center_crop_size=center_crop_size),
                                          center_crop_size=center_crop_size,
                                          neighbour_count=min(neighbour_count, 4),
                                          weights=y_val_weights,
                                          weight_type=use_weights
                                          )

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                       pin_memory=True, drop_last=False)

        val_loader = data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0,
                                     pin_memory=False, drop_last=False)

        loaders = {'train': train_loader, 'val': val_loader}

        BACKBONE = 'resnet34'
        model = sm.Unet(BACKBONE, encoder_weights='imagenet',
                        classes=2, activation='sigmoid')
        model = nn.DataParallel(model, device_ids=[int(device.split('cuda:')[1])])
        model.to(device)

        print('Start training...')

        criterion = {'ce_loss': FocalLoss(gamma=2), 'dice_loss': dice_loss}

        optimizer = {'model': torch.optim.Adam(model.parameters(), lr=0.00001)}

        scheduler = lr_scheduler.StepLR(optimizer['model'], step_size=30, gamma=0.3)

        start_time = time.time()

        # Stage1
        model = train_model(model, criterion, optimizer['model'], scheduler, loaders,
                            result_path, device, num_epochs=80, prefix='stage1.')

        criterion = {}
        if use_weights in ["deform", "custom"]:
            criterion['weights_loss'] = BCEWeightsMapLoss()
        else:
            criterion['weights_loss'] = BCEWeightsDigitLoss()
        # criterion['ce_loss'] = nn.BCELoss()
        criterion['dice_loss'] = dice_loss

        optimizer = {'model': torch.optim.Adam(model.parameters(), lr=0.0001)}

        scheduler = lr_scheduler.StepLR(optimizer['model'], step_size=30, gamma=0.3)

        start_time = time.time()

        # Stage2
        model = train_model(model, criterion, optimizer['model'], scheduler, loaders,
                            result_path, device, num_epochs=160, prefix='stage2.')

        criterion = {}

        if use_weights in ["deform", "custom"]:
            criterion['weights_loss'] = BCEWeightsMapLoss()
        else:
            criterion['weights_loss'] = BCEWeightsDigitLoss()
        criterion['dice_loss'] = dice_loss
        optimizer = {'model': torch.optim.Adam(model.parameters(), lr=0.0001)}

        scheduler = lr_scheduler.StepLR(optimizer['model'], step_size=100, gamma=0.3)

        start_time = time.time()

        # Stage3
        model = train_model(model, criterion, optimizer['model'], scheduler, loaders,
                            result_path, device, num_epochs=500, prefix='stage3.')

        criterion = {'ce_loss': nn.BCELoss(), 'dice_loss': dice_loss}

        # Stage4
        model = train_model(model, criterion, optimizer['model'], scheduler, loaders,
                            result_path, device, num_epochs=100, prefix='stage4.')
