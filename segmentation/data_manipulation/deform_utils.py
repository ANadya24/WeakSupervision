import numpy as np
from typing import Tuple
import cv2
from torch.nn import functional as F
import torch
import os
from skimage import io
from utils import normalize, normalize_mean, pad
from registration.src.model import RegNet
from segmentation.draw_and_show.show_utils import plot_pair


def get_def(fixed_image: np.ndarray, moving_image: np.ndarray,
            model: RegNet, im_size: Tuple[int, int] = (128, 128),
            device: str = 'cpu', smooth_gauss: bool = False):
    """
    Параметры совмещения для пары двух картинок

    :param fixed_image: картинка без маски
    :param moving_image: картинка с маской, которая будет совмещаться с image
    :param voxelmorph: сеть совмещения
    :param im_size: размер входа в нейронку
    :param device: устройтсво вычислений (GPU/CPU)
    :param smooth_gauss: флаг на сглаживание картинок и деформаци фильтром гаусса
    """
    ###Preprocessing
    moving_image = normalize(moving_image.copy())
    fixed_image = normalize(fixed_image.copy())
    if smooth_gauss:
        fixed_image = cv2.GaussianBlur(fixed_image, (5, 5), 1.2)
        moving_image = cv2.GaussianBlur(moving_image, (5, 5), 1.2)
    h, w = fixed_image.shape[:2]
    c = min(h, w)
    fixed_image = fixed_image[:c, :c]
    moving_image = moving_image[:c, :c]
    fixed_image = cv2.resize(fixed_image, im_size)
    moving_image = cv2.resize(moving_image, im_size)

    batch_fixed = torch.Tensor(fixed_image[None, None]).float()
    batch_moving = torch.Tensor(moving_image[None, None]).float()

    batch_fixed = batch_fixed.to(device)
    batch_moving = batch_moving.to(device)
    ###Inference
    img_save, deform, theta, _ = model(batch_moving, batch_fixed)

    ###Postprocessing
    deform = deform.detach().cpu().numpy().transpose((0, 2, 3, 1))[0]
    deform = cv2.resize(deform, (c, c))
    deform = pad(deform, [0, h - c, 0, w - c, 0, 0])
    if smooth_gauss:
        deform = cv2.GaussianBlur(deform, (31, 31), 10)
    deform = torch.Tensor(deform[None]).permute(0, 3, 1, 2)
    return theta, deform


# Вспомогательные функции для перевода матрицы аффинного
# преобразования из СК торча в стандартную СК
def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    N = get_N(W, H)
    return np.linalg.inv(N)


def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`

    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def cvt_ThetaToM(theta, w, h, return_inv=False):
    """convert theta matrix compatible with `torch.F.affine_grid` to affine warp matrix `M`
    compatible with `opencv.warpAffine`.
    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required
    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.
    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    theta_aug = np.concatenate([theta, np.zeros((1, 3))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv[:2, :]
    return M[:2, :]

def on_border(mask, margin):
    if margin == 0:
        return False
    """Проверка того, что объект на границе по маске."""
    left = mask[:, :margin].sum()
    right = mask[:, -margin:].sum()
    bottom = mask[:margin].sum()
    top = mask[-margin:].sum()
    if left or right or bottom or top:
        return True
    return False


def check_mask(out_mask: np.ndarray, mask_moving: np.ndarray):
    """
    Проверка маски на предмет нормальности и правильности совмещения
    """
    if out_mask.sum() > 0.7 * out_mask.size:
        return np.zeros_like(out_mask)
    n_comps, labelled, stats, _ = cv2.connectedComponentsWithStats(out_mask)
    sorted_idxs = np.argsort(stats[1:, -1])[::-1]
    if n_comps > 2:
        out_mask = np.array(labelled == sorted_idxs[0] + 1, dtype=out_mask.dtype)

    sq_now = float(out_mask.sum())
    sq_was = float(mask_moving.sum())
    if (sq_now / sq_was) > 2.:
        return np.zeros_like(out_mask)

    boundaries = out_mask[:, 0].sum() + out_mask[0, :].sum() + \
                 out_mask[-1].sum() + out_mask[:, -1].sum()

    if boundaries > sum(out_mask.shape) / 3:
        return np.zeros_like(out_mask)

    return out_mask


def get_frame_mask_affine_def(cur_frame: int, mask_frame: int, track_id: int,
                              data_path: str, mask_path: str, model: RegNet,
                              im_size: Tuple[int, int] = (128, 128), device: str = 'cpu',
                              debug: bool = False) -> np.ndarray:
    """
    Получение маски с помощью сетей совмещения следующим образом:
    Сначала проходимся последовательно от кадра с маской, до кадра,
    которму нужна маска и вычисляем матрицы аффинного преобразования, последовательно их складывая;
    затем, делаем применяем суммарное аффиное преобразование к moving и ищем деформацию между кадрами.

    :param cur_frame: текущий номер кадра, где отсутствует маска для этого трека
    :param mask_frame: номер кадра, где есть маска для этого трека
    :param track_id: номер трека
    :param data_path: путь к изображениям (кропы треков)
    :param mask_path: путь к маскам (кропы-маски)
    :param model: сеть совмещения
    :param im_size: размер входа сети
    :param device: устройство вычислений (GPU/CPU)
    """
    mpath = os.path.join(mask_path, f'track_{track_id}', f'crop{mask_frame}.tif')
    mask_moving = io.imread(mpath)
    h, w = mask_moving.shape[:2]
    # бинаризуем
    mask_moving = (mask_moving == mask_moving[h // 2, w // 2]).astype('float')
    # читаем кроп, который соответсует маске
    ipath = os.path.join(data_path, f'track_{track_id}', f'crop{mask_frame}.tif')
    img_moving = io.imread(ipath)

    # выстраиваем цепочку последовательных сравнений
    if mask_frame > cur_frame:
        frames = np.arange(mask_frame - 1, cur_frame - 1, -1)
    else:
        frames = np.arange(mask_frame + 1, cur_frame + 1)

    img_init = img_moving.copy()

    total_theta = None
    
    img_fixed = None

    for frame in frames:
        # читаем соседний кадр и совмещаем его с масочным
        ipath = os.path.join(data_path, f'track_{track_id}', f'crop{frame}.tif')
        if not os.path.exists(ipath):
            continue
        img_fixed = io.imread(ipath)
        theta, _ = get_def(img_fixed, img_moving,
                           model, im_size=im_size, device=device)

        img_moving = img_fixed.copy()
        theta = cvt_ThetaToM(theta.detach().cpu().numpy().squeeze(),
                             img_fixed.shape[1], img_fixed.shape[0])
        if total_theta is None:
            total_theta = theta
        else:
            total_theta = total_theta @ np.concatenate([theta, np.array([0, 0, 1]).reshape(1, 3)], axis=0)
            
    if img_fixed is None:
        return mask_moving * 0

    total_theta = cvt_MToTheta(total_theta, img_fixed.shape[1],
                               img_fixed.shape[0])
    if debug:
        plot_pair(img_fixed, img_init, 10)
        plot_pair(mask_moving, img_init, 10)

    moving_init_tensor = torch.Tensor(normalize(img_init.copy()).astype('float32')[None, None])

    grid = F.affine_grid(torch.tensor(total_theta).float()[None], moving_init_tensor.size())
    affine_moving_init_tensor = F.grid_sample(moving_init_tensor, grid).float()

    mask_moving_tensor = torch.Tensor(normalize(mask_moving.copy())[None, None])
    affine_mask_moving_tensor = F.grid_sample(mask_moving_tensor, grid).float()

    mask_moving_tensor = affine_mask_moving_tensor

    ipath = os.path.join(data_path, f'track_{track_id}', f'crop{cur_frame}.tif')
    img_fixed = io.imread(ipath)

    _, deform = get_def(normalize_mean(img_fixed),
                        normalize_mean(affine_moving_init_tensor.detach().cpu().numpy().squeeze()),
                        model, im_size, device=device)

    if debug:
        mask_tmp = mask_moving_tensor.detach().cpu().numpy().squeeze()
        plot_pair(mask_tmp, img_fixed, 10)

    mask_moving_tensor = mask_moving_tensor.to(device)
    deform = deform.to(device)

    out_mask = model.spatial_transform(mask_moving_tensor, deform)
    out_mask = out_mask.detach().cpu().numpy().squeeze()

    out_mask = cv2.GaussianBlur(out_mask, (9, 9), 0)  # was 9
    out_mask = out_mask > 0.5
    out_mask = out_mask.astype('uint8')

    out_mask = check_mask(out_mask, mask_moving)
    return out_mask
