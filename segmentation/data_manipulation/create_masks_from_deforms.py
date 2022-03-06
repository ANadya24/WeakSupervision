import json
import sys
import os
import torch
from skimage import io
from collections import defaultdict
import numpy as np
import warnings

from deform_utils import get_frame_mask_affine_def, on_border
from utils import pad_to_shape

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        config = json.load(file)
    dataset = config["dataset"]

    sys.path.append(config["model_path"])
    from src.model import RegNet

    checkpoint_path = os.path.join(config["model_path"], config["checkpoint_name"])
    im_size = (config["image_size"], config["image_size"])
    model = RegNet(2, image_size=config["image_size"], device=config["device"])
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.to(config["device"])
    model.eval()
    print("Model loaded successfully!")

    data_path = config["data_path"]
    seg_type = config["seg_type"]
    num_seq = config["num_seq"]
    r = config["radius"]
    mode = config["mode"]
    bad_cells_limit = config["bad_cells_limit"]
    save_prefix = config["save_prefix"]
    margin = config["margin"]

    image_path = f'{data_path}/{dataset}/{num_seq:02d}/'

    path_crop_images = f'{data_path}/{dataset}/{num_seq:02d}_CROP/'
    path_crop_masks = f'{data_path}/{dataset}/{num_seq:02d}_CROP_SEG/'
    path_tracks = f'{data_path}/{dataset}/{num_seq:02d}_GT/TRA/'
    tra_gt_path = f'{data_path}/{dataset}/{num_seq:02d}_GT/TRA/man_track.txt'

    neighbours_param = config["neighbours_param"]
    opath_crop_masks = f'{data_path}/{dataset}/{num_seq:02d}_DEF{save_prefix}/{seg_type}/CROP/{neighbours_param}/'
    seg_path = f'{data_path}/{dataset}/{num_seq:02d}_GT/SEG/'
    if seg_type == "GT":
        st_seg_path = seg_path
    else:
        st_seg_path = f'{data_path}/{dataset}/{num_seq:02d}_ST/SEG/'
    oseg_path = f'{data_path}/{dataset}/{num_seq:02d}_DEF{save_prefix}/{seg_type}/FRAMES/{neighbours_param}/'
    oweights_path = oseg_path.replace('FRAMES', 'WEIGHTS')

    os.makedirs(oseg_path, exist_ok=True)
    os.makedirs(oweights_path, exist_ok=True)
    os.makedirs(opath_crop_masks, exist_ok=True)
    debug = config["debug"]
    r = config["radius"]

    with open(tra_gt_path, 'r') as file:
        tracks = [l.strip() for l in file.readlines()]

    segs = defaultdict(list)

    for i, track in enumerate(tracks):
        track_id, start_frame, end_frame, _ = list(map(int, track.split(' ')))
        for j in range(start_frame, end_frame + 1):
            segs[j].append(track_id)

    for frame in segs:
        skip = False
        name = path_tracks + f'man_track{frame:03d}.tif'
        track_img = io.imread(name)

        seg_name = seg_path + f'man_seg{frame:03d}.tif'

        if not os.path.exists(seg_name):
            seg = np.zeros_like(track_img)
            label_num = 1
        else:
            seg_name = st_seg_path + f'man_seg{frame:03d}.tif'
            seg = io.imread(seg_name)
            label_num = seg.max() + 1

        weights = seg.copy()
        weights[weights > 0] = 3

        bad_cells = 0
        for track_id in segs[frame]:
            opath = os.path.join(opath_crop_masks, f'track_{track_id}')
            os.makedirs(opath, exist_ok=True)

            mask = track_img == track_id

            if (seg * mask).sum() == 0:
                weights_mul = 2
                # нет пересечений с GT маской
                masks_path = os.path.join(path_crop_masks, f'track_{track_id}')
                diff_frames_masks = list(filter(lambda crop_name:
                                                crop_name.find('tif') > 0, os.listdir(masks_path)))
                if len(diff_frames_masks) == 0:
                    weights_mul = 1
                    if on_border(mask, margin):
                        continue
                    if mode == "fill":
                        mask_crop = track_img == track_id
                        y, x = np.where(mask)
                        cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
                        cx, cy = list(map(int, [cx, cy]))
                        h, w = track_img.shape[:2]
                        st_y, en_y, st_x, en_x = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
                        mask_crop = mask_crop[st_y:en_y, st_x:en_x]
                    else:
                        continue
                else:
                    neighbours = [int(d.split('crop')[-1].split('.tif')[0]) for d in diff_frames_masks]
                    idx = np.argmin(abs(np.array(neighbours) - frame))

                    if abs(frame - neighbours[idx]) > neighbours_param:
                        # если расстрояние больше чем окрестность, из которой берем маски
                        print('расстояние больше чем окрестность, из которой берем маски')
                        skip = True
                        input()
                        break

                    mask_crop = get_frame_mask_affine_def(frame, neighbours[idx], track_id,
                                                          path_crop_images, path_crop_masks, model,
                                                          im_size, device=config['device'], debug=False)

                if mask_crop.sum() == 0:
                    weights_mul = 1
                    print('кривое совмещение')
                    bad_cells += 1
                    if mode == "fill":
                        mask_crop = track_img == track_id
                        y, x = np.where(mask)
                        cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
                        cx, cy = list(map(int, [cx, cy]))
                        h, w = track_img.shape[:2]
                        st_y, en_y, st_x, en_x = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
                        mask_crop = mask_crop[st_y:en_y, st_x:en_x]
                    else:
                        continue
            else:
                weights_mul = 3
                y, x = np.where(mask)
                cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
                cx, cy = list(map(int, [cx, cy]))
                h, w = track_img.shape[:2]
                st_y, en_y, st_x, en_x = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
                mask_crop = seg[st_y:en_y, st_x:en_x]
                mask_crop = mask_crop == seg[cy, cx]
                pad_st_y = -1 * min(cy - r, 0)
                pad_st_x = -1 * min(cx - r, 0)
                pad_en_y = max(cy + r + 1 - h, 0)
                pad_en_x = max(cx + r + 1 - w, 0)
                mask_crop = np.pad(mask_crop, np.array([pad_st_y, pad_en_y, pad_st_x, pad_en_x]).reshape(2, 2))

            mask_crop = mask_crop.astype('uint16')

            io.imsave(os.path.join(opath, f'crop{frame}.tif'), mask_crop)

            y, x = np.where(mask)
            cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
            cx, cy = list(map(int, [cx, cy]))
            h, w = track_img.shape[:2]
            st_y, en_y, st_x, en_x = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)

            pad_st_y = -1 * min(cy - r, 0)
            pad_st_x = -1 * min(cx - r, 0)
            pad_en_y = -1 * max(cy + r + 1 - h, 0)
            pad_en_x = -1 * max(cx + r + 1 - w, 0)
            if pad_en_x == 0:
                pad_en_x = w
            if pad_en_y == 0:
                pad_en_y = h

            arr = seg[st_y: en_y, st_x: en_x]
            mask_crop = mask_crop[pad_st_y:pad_en_y, pad_st_x:pad_en_x]
            mask_crop *= label_num
            label_num += 1
            init_shape = mask_crop.shape
            mask_crop = pad_to_shape(mask_crop, arr.shape[:2])
            mask_crop[arr > 0] = 0
            arr = arr + mask_crop
            seg[st_y: en_y, st_x: en_x] = arr
            weights[st_y: en_y, st_x: en_x] = np.max(np.stack([(arr > 0).astype('int') * weights_mul,
                                                     weights[st_y: en_y, st_x: en_x]], -1), -1)

        if skip:
            continue

        seg = seg.astype('uint16')
        if seg.max() != 0 and bad_cells <= bad_cells_limit:
            print('saving...', oseg_path + f'man_seg{frame:03d}.tif')
            io.imsave(oseg_path + f'man_seg{frame:03d}.tif', seg)
            io.imsave(oweights_path + f'w{frame:03d}.tif', weights)
