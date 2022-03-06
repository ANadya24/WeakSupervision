from skimage import io
import os
import numpy as np

#MASK CROPS
num_seq = 2
dataset = 'Fluo-N2DH-GOWT1' #'Fluo-N2DL-HeLa'  # 'Fluo-N2DH-GOWT1'#'DIC-C2DH-HeLa'
path_images = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/SEG/'
path_tracks = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/TRA/'
tra_gt_path = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/TRA/man_track.txt'
crop_path = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_CROP_SEG/'

radius = {'Fluo-N2DH-GOWT1': 50, 'Fluo-N2DL-HeLa': 25, 'DIC-C2DH-HeLa': 100,
          'Fluo-C2DL-Huh7': 80}

# IMAGE CROPS
# num_seq = 2
# dataset = 'Fluo-N2DH-GOWT1' # 'Fluo-N2DL-HeLa'
# path_images = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}/'
# path_tracks = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/TRA/'
# tra_gt_path = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_GT/TRA/man_track.txt'
# crop_path = f'/srv/fast1/n.anoshina/data/{dataset}/{num_seq:02d}_CROP/'

pad_2_square = False
r = radius[dataset]

with open(tra_gt_path, 'r') as file:
    tracks = [l.strip() for l in file.readlines()]

for i, track in enumerate(tracks):
    track_id, start_frame, end_frame, _ = list(map(int, track.split(' ')))
    opath = os.path.join(crop_path, f'track_{track_id}')
    os.makedirs(opath, exist_ok=True)
    for j in range(start_frame, end_frame + 1):
        name = path_tracks + f'man_track{j:03d}.tif'
        track_img = io.imread(name)
        if path_images.find('SEG') > 0:
            name = path_images + f'man_seg{j:03d}.tif'
        else:
            name = path_images + f't{j:03d}.tif'

        if not os.path.exists(name):
            continue

        cell_img = io.imread(name, 2)

        mask = track_img == track_id

        y, x = np.where(mask)
        cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
        cx, cy = list(map(int, [cx, cy]))
        h, w = track_img.shape[:2]
        st_y, en_y, st_x, en_x = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
        crop = cell_img[st_y:en_y, st_x:en_x]
        pad_st_y = -1 * min(cy - r, 0)
        pad_st_x = -1 * min(cx - r, 0)
        pad_en_y = max(cy + r + 1 - h, 0)
        pad_en_x = max(cx + r + 1 - w, 0)
        if pad_2_square:
            crop = np.pad(crop, np.array([pad_st_y, pad_en_y, pad_st_x, pad_en_x]).reshape(2, 2))

        if crop.sum() == 0 or crop[r - 2:r + 2, r - 2:r + 2].sum() == 0:
            continue

        io.imsave(opath + f'/crop{j}.tif', crop)
