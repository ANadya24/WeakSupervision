import numpy as np


def is_border(crop, interval=2):
    """
    Проверка, что паддинг добавлен к кропу для квадратного размера

    :param crop:
    :param interval:
    :return:
    """
    left = crop[:, :interval].sum()
    right = crop[:, -interval:].sum()
    bottom = crop[:interval].sum()
    top = crop[-interval:].sum()
    if not left or not right or not bottom or not top:
        return True
    return False


def get_radius(track_id, track_img, seg_img, expand_r=0.1):
    """
    По маске клетки определяет размер кропа.

    :param track_id: номер трека
    :param track_img: картинка с треками клеток
    :param seg_img: картинка с масками клеток
    :param expand_r: значение расширения радиуса
    :return: радиус кропа
    """
    mask = track_img == track_id
    # Вырезаем кроп по маске
    y, x = np.where(mask)
    cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
    cx, cy = list(map(int, [cx, cy]))
    y, x = np.where(seg_img == seg_img[cy, cx])
    # m_cx, m_cy = np.ceil(x.mean()), np.ceil(y.mean())
    # m_cx, m_cy = list(map(int, [m_cx, m_cy]))
    m_cx, m_cy = cx, cy
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    r = max([m_cx - x_min, x_max - m_cx, m_cy - y_min, y_max - m_cy])
    r += int(expand_r * r)
    return r


def extract_crop_by_mask(track_id, track_img, seg_img, cell_img, expand_r=0.1, r=None, pad=True):
    """
    По маске клетки определяет размер кропа и вырезает из общей картинки.
    Центром кропа считается координаты трека клетки на картинке

    :param track_id: номер трека
    :param track_img: картинка с треками клеток
    :param seg_img: картинка с масками клеток
    :param cell_img: исходная картинка
    :param expand_r: значение расширения радиуса
    :param pad: Нужно ли добавлять паддинг до квадратного размера
    :return: кроп, маска кропа, радиус кропа
    """
    if r is None:
        r = get_radius(track_id, track_img, seg_img, expand_r)
    mask = track_img == track_id
    # Вырезаем кроп по маске
    y, x = np.where(mask)
    cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
    cx, cy = list(map(int, [cx, cy]))
    h, w = track_img.shape[:2]
    st_y, en_y, st_x, en_x = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
    mask_crop = seg_img[st_y:en_y, st_x:en_x]
    
    mask_crop = (mask_crop == seg_img[cy, cx]).astype('float32')

    crop = cell_img[st_y:en_y, st_x:en_x]

    if pad:
        pad_st_y = -1 * min(cy - r, 0)
        pad_st_x = -1 * min(cx - r, 0)
        pad_en_y = max(cy + r + 1 - h, 0)
        pad_en_x = max(cx + r + 1 - w, 0)
        mask_crop = np.pad(mask_crop, np.array([pad_st_y, pad_en_y, pad_st_x, pad_en_x]).reshape(2, 2))
        crop = np.pad(crop, np.array([pad_st_y, pad_en_y, pad_st_x, pad_en_x]).reshape(2, 2))

    return crop, mask_crop, r


def extract_crop_by_size(track_id, track_img, r, cell_img, pad=True):
    """
    Зная радиус и центр кропа вырезаем его из общей картинки

    :param track_id: номер трека
    :param track_img: общая картинка с треками клеток
    :param r: радиус кропа, который нужно вырезать
    :param cell_img: исходная кобщая картинка
    :param pad: Нужно ли добавлять паддинг до квадратного размера
    :return: кроп
    """
    mask = track_img == track_id
    # Вырезаем кроп по маске
    y, x = np.where(mask)
    cx, cy = np.ceil(x.mean()), np.ceil(y.mean())
    cx, cy = list(map(int, [cx, cy]))

    h, w = track_img.shape[:2]
    st_y, en_y, st_x, en_x = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
    crop = cell_img[st_y:en_y, st_x:en_x]
    if pad:
        pad_st_y = -1 * min(cy - r, 0)
        pad_st_x = -1 * min(cx - r, 0)
        pad_en_y = max(cy + r + 1 - h, 0)
        pad_en_x = max(cx + r + 1 - w, 0)
        crop = np.pad(crop, np.array([pad_st_y, pad_en_y, pad_st_x, pad_en_x]).reshape(2, 2))

    return crop


def pad(img:np.ndarray, padwidth: list):
    """
    Паддинг изображения нулями по высоте и ширине

    :param img: входное изображение
    :param padwidth: параметры паддинга сначала задаются 2
    значения для высоты, затем 2 значения для ширины
    :return: изображение с паддингом
    """
    h0, h1, w0, w1 = padwidth[:4]
    if len(padwidth) < 5:
        img = np.pad(img, ((h0, h1), (w0, w1)))
    else:
        img = np.pad(img, ((h0, h1), (w0, w1), (0, 0)))

    return img


def normalize(im: np.ndarray):
    """
    Нормировка значений изображения путем преобразования
    максимального значения в 1, а минимального в 0

    :param im: входное изображение
    :return: нормализованное изображение
    """
    im = im - im.min()
    im = im / im.max()
    return im


def pad_to_shape(img, shape):
    """
    Паддинг изображения равномерно с
    каждой стороны к нужному размеру

    :param img: входное изображение
    :param shape: размер, до которого нужно
    допаддить изображение
    :return:
    """
    h, w = img.shape[:2]
    h1, w1 = shape
    if h < h1 or w < w1:
        diff_h = max(h1-h, 0)
        diff_w = max(w1-w, 0)
        if len(img.shape) == 3:
            img = np.pad(img, np.array([0, diff_h, 0, diff_w, 0, 0]).reshape(3, 2))
        else:
            img = np.pad(img, np.array([0, diff_h, 0, diff_w]).reshape(2, 2))
    img = img[:h1, :w1]
    return img


def normalize_mean(img):
    mean, std = img.mean(), img.std()
    return (img-mean)/std
