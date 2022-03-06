import os
from collections import defaultdict
from skimage import io
import numpy as np
from glob import glob
from utils import (
    extract_crop_by_mask,
    extract_crop_by_size,
    get_radius
)

num_seq = 2
seg_suffix = ''
image_suffix = '_hist'
dataset = 'Fluo-C2DL-Huh7'  #'PhC-C2DH-U373'#'Fluo-N2DH-SIM+'
# path_images = f'/home/n.anoshina/CTC/segmentation/data/{dataset}/{num_seq:02d}/'
# path_tracks = f'/home/n.anoshina/CTC/segmentation/data/{dataset}/{num_seq:02d}_GT/TRA/'
# path_seg = f'/home/n.anoshina/CTC/segmentation/data/{dataset}/{num_seq:02d}_GT/SEG/'
# tra_gt_path = f'/home/n.anoshina/CTC/segmentation/data/{dataset}/{num_seq:02d}_GT/TRA/man_track.txt'
# crop_path = f'/home/n.anoshina/CTC/segmentation/data/{dataset}/{num_seq:02d}_CROP/'
path_images = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}{image_suffix}/'
path_tracks = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/TRA/'
path_seg = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/SEG/'
tra_gt_path = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/TRA/man_track.txt'
crop_path = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_CROP{seg_suffix}/'

##FOR MASKS UNCOMMENT
# crop_path_masks = f'/home/n.anoshina/CTC/segmentation/data/{dataset}/{num_seq:02d}_CROP_SEG/'
crop_path_masks = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_CROP_SEG{seg_suffix}/'
create_masks_crops = True
if create_masks_crops:
    os.makedirs(crop_path_masks, exist_ok=True)

expand_r = 0.1
pad = False
global_radius = {'Fluo-N2DH-GOWT1': 50, 'Fluo-N2DL-HeLa': 25, 'DIC-C2DH-HeLa': 100,
                 'Fluo-C2DL-Huh7': 80}

with open(tra_gt_path, 'r') as file:
    tracks = [l.strip() for l in file.readlines()]

frame_map = defaultdict(list)
tracks_map = defaultdict(list)
seg_map = defaultdict(list)
radiuses = defaultdict(int)

for i, track in enumerate(tracks):
    track_id, start_frame, end_frame, _ = list(map(int, track.split(' ')))
    tracks_map[track_id].extend(np.arange(start_frame, end_frame + 1).tolist())
    for j in range(start_frame, end_frame + 1):
        frame_map[j].append(track_id)

for seg_name in glob(path_seg + 'man_seg*.tif'):
    seg_img = io.imread(seg_name)
    idx_frame = int(seg_name.split('man_seg')[-1].split('.')[0])
    for track_id in frame_map[idx_frame]:
        track_name = path_tracks + f'man_track{idx_frame:03d}.tif'
        track_img = io.imread(track_name)

        mask = track_img == track_id
        if (seg_img * mask).sum() == 0:
            continue
        else:
            seg_map[idx_frame].append(track_id)
            if create_masks_crops:
                opath = os.path.join(crop_path_masks, f'track_{track_id}')
                os.makedirs(opath, exist_ok=True)
        del track_img
        del mask
    del seg_img

for frame in seg_map:
    seg_name = path_seg + f'man_seg{frame:03d}.tif'
    seg_img = io.imread(seg_name)

    cell_name = path_images + f't{frame:03d}.tif'
    cell_img = io.imread(cell_name)

    track_name = path_tracks + f'man_track{frame:03d}.tif'
    track_img = io.imread(track_name)
    for track_id in seg_map[frame]:
        r = get_radius(track_id, track_img, seg_img, expand_r=expand_r)
        # assert abs(max(mask_crop.shape) - 2*r-1) <= 1, f'{mask_crop.shape} and {2*r + 1}'

        if track_id in radiuses:
            radiuses[track_id] = max(radiuses[track_id], r)
        else:
            radiuses[track_id] = r

for i, track in enumerate(tracks):
    track_id, start_frame, end_frame, _ = list(map(int, track.split(' ')))
    if track_id not in radiuses:
        print('ALERT', track_id)
        r = global_radius[dataset]
        continue

    opath = os.path.join(crop_path, f'track_{track_id}')
    os.makedirs(opath, exist_ok=True)

    for j in range(start_frame, end_frame + 1):
        r = radiuses[track_id]
        name = path_tracks + f'man_track{j:03d}.tif'
        track_img = io.imread(name)

        cell_name = path_images + f't{j:03d}.tif'
        cell_img = io.imread(cell_name, 2)

        if track_id in seg_map[j]:
            seg_name = path_seg + f'man_seg{j:03d}.tif'
            seg_img = io.imread(seg_name)
            crop, mask_crop, _ = extract_crop_by_mask(track_id, track_img,
                                                     seg_img, cell_img, r=r)
            
            if create_masks_crops:
                opath_mask = os.path.join(crop_path_masks, f'track_{track_id}')
                io.imsave(opath_mask + f'/crop{j}.tif', mask_crop)
        else:
            crop = extract_crop_by_size(track_id, track_img, r, cell_img, pad=pad)

        if crop.sum() == 0 or crop[r - 2:r + 2, r - 2:r + 2].sum() == 0:
            continue

        io.imsave(opath + f'/crop{j}.tif', crop)
