from skimage.color import label2rgb
import cv2
from skimage.morphology import *
import imutils
from skimage import img_as_ubyte

import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import adjust_gamma

from segmentation.data_manipulation.utils import normalize


def unet_weight_map(y, wc=None, w0=10, sigma=5):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.
    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2)

        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(y)

    return w


def draw_mask_color(mask, draw_contours=False, draw_weights=True):
    """
    Нарисовать бинарную маску в цвете, по желанию указывая все контуры и веса
    """
    img = np.zeros(mask.shape + (3,), dtype='uint8')
    img[:, :, 0] += 255

    for label in np.unique(mask):
        if label == 0:
            continue
        cnt_im = img_as_ubyte((mask == label).astype('int'))
        cnts = cv2.findContours(cnt_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cv2.drawContours(img, cnts, 0, (0, 255, 0), -1)

    if draw_contours:
        for label in np.unique(mask):
            if label == 0:
                continue
            cnt_im = img_as_ubyte((mask == label).astype('int'))
            cnts = cv2.findContours(cnt_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cv2.drawContours(img, cnts, -1, (255, 255, 0), 1)

    if draw_weights:
        weights = unet_weight_map(mask)
        weights = (weights > 9.).astype('uint8')
        tmp = np.stack([weights * 0, weights * 0, weights * 255], -1).astype('uint8')
        img = img * (1 - weights[:, :, None]) + tmp
    return img


def draw_mask_image(img, mask, draw_contours=True):
    """
    Нарисовать маску с разноцветными instance поверх картинки
    """
    if len(img.shape) == 2:
        img = np.stack([img, img, img], -1)
        img = normalize(img)
        img = np.uint8(img * 255)
        img = adjust_gamma(img, 0.2)
    img = label2rgb(mask, image=img, bg_label=0, alpha=0.4, image_alpha=1.)
    img = np.uint8(img * 255.)
    if draw_contours:
        for label in np.unique(mask):
            if label == 0:
                continue
            cnt_im = img_as_ubyte((mask == label).astype('int'))
            cnts = cv2.findContours(cnt_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    return img