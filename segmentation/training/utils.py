from skimage.morphology import dilation, square
from scipy.ndimage import distance_transform_edt as distance
from skimage.measure import label
import numpy as np


def unet_weight_map(mask, wc=None, w0=10, sigma=5):
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

    labels = label(mask)
    # no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2)

        if wc:
            class_weights = np.zeros_like(mask)
            for k, v in wc.items():
                class_weights[mask == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(mask)

    return w


def preprocess_label(mask):
    label = mask.copy()
    dilated = dilation(label, square(3))
    outline_mask = dilated != label
    label[outline_mask] = 0  # nicely creates 1-pixel border between touching cells
    label[label > 0] = 1  # change to binary
    return label
