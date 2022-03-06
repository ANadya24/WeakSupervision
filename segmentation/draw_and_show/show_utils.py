from matplotlib import pyplot as plt
from skimage import io
import numpy as np


def plot_pair(img, mask, size):
    f = plt.figure(figsize=(size, 2 * size))
    ax_img = f.add_subplot(121)
    ax = f.add_subplot(122)
    cmap_1 = None
    if len(img.shape) < 3:
        cmap_1 = 'gray'
    elif img.shape[2] == 1:
        cmap_1 = 'gray'

    cmap_2 = None
    if len(mask.shape) < 3:
        cmap_2 = 'gray'
    elif mask.shape[2] == 1:
        cmap_2 = 'gray'
    ax_img.imshow(img, cmap=cmap_1)
    ax.imshow(mask, cmap=cmap_2)
    io.show()


def plot_pairs(list1, list2, size):
    for x, y in zip(list1, list2):
        plot_pair(x, y, size)


def plot_triplet(img, mask, img3, size):
    f = plt.figure(figsize=(size, 3 * size))
    ax_img = f.add_subplot(131)
    ax = f.add_subplot(132)
    ax_3 = f.add_subplot(133)
    cmap_1 = None
    if len(img.shape) < 3:
        cmap_1 = 'gray'
    elif img.shape[2] == 1:
        cmap_1 = 'gray'
    cmap_2 = None
    if len(mask.shape) < 3:
        cmap_2 = 'gray'
    elif mask.shape[2] == 1:
        cmap_2 = 'gray'
    cmap_3 = None
    if len(img3.shape) < 3:
        cmap_3 = 'gray'
    elif img3.shape[2] == 1:
        cmap_3 = 'gray'
    ax_img.imshow(img, cmap=cmap_1)
    ax.imshow(mask, cmap=cmap_2)
    ax_3.imshow(img3, cmap=cmap_3)
    io.show()


# def plot_rgb(rgb, size):
#     f = plt.figure(figsize=(size, 2 * size))
#     ax = f.add_subplot(111)
#     ax.imshow(rgb)
#     io.show()

def plot_rgb(r, g, b, size):
    rgb = np.stack([r, g, b], -1)
    f = plt.figure(figsize=(size, size))
    ax = f.add_subplot(111)
    ax.imshow(rgb)
    io.show()
