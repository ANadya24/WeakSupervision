import pickle
import os
import numpy as np
from skimage import io
from skimage.color import gray2rgb
from skimage.exposure import adjust_gamma

from segmentation.data_manipulation.utils import normalize
from draw_utils import draw_mask_image, draw_mask_color


dataset = ""
out_path = ""
seg_types = ""
images_path = ""
mask_paths = ""

with open(images_path, 'rb') as file:
    img_label_pairs_gt = pickle.load(file)

if isinstance(seg_types, str):
    seg_types = [seg_types]

if isinstance(mask_paths, str):
    mask_paths = [mask_paths]

assert len(seg_types) == len(mask_paths), "The amount of different modes must " \
                                           "be equal to the amount of mask arrays"
for mask_path, seg_type in zip(mask_paths, seg_types):
    with open(mask_path, 'rb') as file:
        out_masks = pickle.load(file)

        os.makedirs(out_path + f'{dataset}/{seg_type}/images/')
        for i, (pair, res_mask) in enumerate(zip(img_label_pairs_gt, out_masks)):
            img, _ = pair
            image = np.uint8(draw_mask_image(img, res_mask))
            mask = draw_mask_color(res_mask, True)
            io.imsave(out_path + f'{dataset}/{seg_type}/images/im_{i}.png', image)
            io.imsave(out_path + f'{dataset}/{seg_type}/images/mask_{i}.png', mask)

os.makedirs(out_path + f'{dataset}/images/')
os.makedirs(out_path + f'{dataset}/masks/')
for i, pair in enumerate(img_label_pairs_gt):
    img, mask = pair
    image = gray2rgb(np.uint8(adjust_gamma(normalize(img.copy()), 0.5) * 255))
    mask_int = np.uint8(mask.copy() > 0) * 255
    io.imsave(out_path + f'{dataset}/images/im_{i}.png', image)
    io.imsave(out_path + f'{dataset}/images/mask_{i}.png', mask_int)
    image = np.uint8(draw_mask_image(img, mask))
    mask = draw_mask_color(mask, True)
    io.imsave(out_path + f'{dataset}/masks/im_{i}.png', image)
    io.imsave(out_path + f'{dataset}/masks/mask_{i}.png', mask)


