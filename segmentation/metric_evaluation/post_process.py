from collections import Counter
import imutils
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import numpy as np

from skimage.segmentation import watershed
from skimage.morphology import erosion, dilation, \
    remove_small_holes, remove_small_objects
from skimage.color import label2rgb
import cv2
from skimage.feature import peak_local_max
from skimage import img_as_ubyte


def normalize(im: np.ndarray):
    im = im - im.min()
    im = im / im.max()
    return im


def remove_small_labels(img, min_size=150):
    sizes = []
    for label_num in np.unique(img):
        sizes.append((img == label_num).sum())

    img2 = img.copy()
    for i, label_num in enumerate(np.unique(img)):
        if label_num == 0:
            continue
        if sizes[i] < min_size:
            mask = img2 == label_num
            mask = dilation(mask, np.ones((31, 31))) ^ mask
            nearest_neighbour = Counter(img2[mask][img2[mask] != 0])
            if len(nearest_neighbour) == 0:
                continue
            nearest_neighbour = nearest_neighbour.most_common()[0][0]
            img2[img2 == label_num] = nearest_neighbour

    return img2


def postprocess_watershed(res_mask, markers, thr=0.5,
                          invert_markers=False,
                          bad_markers=False,
                          num_erosion=15,
                          marker_erosion_seed=np.ones((5, 5)),
                          footprint=np.ones((80, 80)),
                          small_holes_param=200,
                          small_objects_param=200,
                          min_size_label=1000,
                          marker_threshold=0.1,
                          debug=False):
    mask = res_mask > thr

    if invert_markers:
        markers = (1 - markers)

    if marker_erosion_seed is not None:
        if isinstance(marker_erosion_seed, int):
            marker_erosion_seed = np.ones((marker_erosion_seed, marker_erosion_seed))
        for i in range(3):
            markers = erosion(markers, marker_erosion_seed)
        markers = dilation(markers, marker_erosion_seed)
    markers = normalize(markers)

    if debug:
        print(markers.max(), markers.min())
        plt.imshow(markers)
        plt.pause(1)

    # distance = ndi.distance_transform_edt(res_mask)
    distance = (res_mask.max() - res_mask)
    #     markers = h_maxima(markers, 0.2)

    marker_mask = markers > marker_threshold
    if debug:
        plt.imshow(marker_mask)
        plt.pause(1)

    if bad_markers:
        if isinstance(footprint, int):
            footprint = np.ones((footprint, footprint))

        coords = peak_local_max(markers,
                                threshold_abs=np.median(markers),
                                footprint=footprint)
        marker_mask = np.zeros(distance.shape, dtype=bool)
        marker_mask[tuple(coords.T)] = True
        markers = remove_small_holes(markers >= np.median(markers), 10)

        for i in range(num_erosion):
            markers = erosion(markers, np.ones((5, 5)))

        markers = img_as_ubyte(markers)
        cnts = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        tmp = np.zeros(markers.shape)

        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] == 0.:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(tmp, (cX, cY), 3, (1, 1, 1), -1)

        marker_mask = np.logical_or(marker_mask, tmp > 0)

    if debug:
        plt.imshow(dilation(marker_mask, np.ones((10, 10))))
        plt.pause(1)
        plt.imshow(distance)
        plt.pause(1)
        print(marker_mask.dtype)

    markers, _ = ndi.label(marker_mask)
    mask = remove_small_objects(remove_small_holes(mask, small_holes_param),
                                    min_size=small_objects_param)
    instance_mask = watershed(distance, markers, mask=mask, connectivity=8).astype('uint16')
    instance_mask = remove_small_labels(instance_mask, min_size=min_size_label)

    if debug:
        print(np.unique(instance_mask))
        labels_show = label2rgb(instance_mask, bg_label=0)

        labels_show[dilation(marker_mask, np.ones((10, 10)))] = 0

        plt.figure(figsize=(15, 15))
        plt.imshow(labels_show)
        plt.pause(2)
        # plot_pair(res_mask, mask, 15)
        print(len(np.unique(instance_mask)), len(np.unique(mask)))

    return instance_mask


def create_instance_masks(masks, params, thr=None, debug=False):
    if "thr" in params and thr is None:
        thr = params.pop("thr")
    instance_masks = []
    for mask in masks:
        markers = mask[:, :, 1]
        mask = mask[:, :, 0]
        instance_mask = postprocess_watershed(mask, markers, thr, debug=debug, **params)
        instance_masks.append(instance_mask)
    return instance_masks
