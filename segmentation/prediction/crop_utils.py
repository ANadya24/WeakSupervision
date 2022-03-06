import numpy as np


def imToCrop(image, size):
    """
    Разрезать картинку на кропы для более точного предсказания
    :param image: входное изображение
    :param sizeЖ размер кропа
    """
    h, w, _ = image.shape
    i_h = h // size[0] + 1
    i_w = w // size[1] + 1
    x = set()
    y = set()
    for i in range(i_h):
        if i == (i_h - 1):
            y.add((h - size[0], h))
        else:
            y.add((i * size[0], (i + 1) * size[0]))
    for j in range(i_w):
        if j == (i_w - 1):
            x.add((w - size[1], w))
        else:
            x.add((j * size[1], (j + 1) * size[1]))
    crop_images = np.empty((len(x) * len(y), size[0], size[1], size[2]), dtype=np.float)
    for i, (imin, imax) in enumerate(y):
        for j, (jmin, jmax) in enumerate(x):
            crop = image[imin:imax, jmin:jmax]
            crop_images[np.ravel_multi_index((i, j), (len(y), len(x)))] = crop

    return crop_images


def cropToIm(imcrops, imsize):
    """
    Собрать кропы в картинку
    :param imcrops: список кропов, составляющих полную картинку
    :param imsize: размер изображения, которое нужно собрать
    """
    h, w = imsize[:2]

    _, ch, cw, cc = imcrops.shape

    i_h = h // ch + 1
    i_w = w // cw + 1
    x = set()
    y = set()
    for i in range(i_h):
        if i == (i_h - 1):
            y.add((h - ch, h))
        else:
            y.add((i * ch, (i + 1) * ch))
    for j in range(i_w):
        if j == (i_w - 1):
            x.add((w - cw, w))
        else:
            x.add((j * cw, (j + 1) * cw))
    image = np.zeros((h, w, cc), dtype=np.float)

    for i, (imin, imax) in enumerate(y):
        for j, (jmin, jmax) in enumerate(x):
            crop = imcrops[np.ravel_multi_index((i, j), (len(y), len(x)))]
            crop[crop < image[imin:imax, jmin:jmax]] = image[imin:imax, jmin:jmax][
                crop < image[imin:imax, jmin:jmax]]
            image[imin:imax, jmin:jmax] = crop
    return image

