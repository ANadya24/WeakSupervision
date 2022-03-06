import numpy as np
import torch
import torch.nn.functional as F

import pytorch_ssim


def cross_correlation(x, y, n, use_gpu=False):
    """
    Cross-correlation metric computation.
    """
    batch_size, channels, xdim, ydim = x.shape
    x_sq = torch.mul(x, x)
    y_sq = torch.mul(y, y)
    inputs_mul = torch.mul(x, y)
    sum_filter = torch.ones((1, channels, n, n))
    if use_gpu:
        sum_filter = sum_filter.cuda()
    x_sum = torch.conv2d(x, sum_filter, padding=1, stride=(1, 1))
    y_sum = torch.conv2d(y, sum_filter, padding=1, stride=(1, 1))
    x_sq_sum = torch.conv2d(x_sq, sum_filter, padding=1, stride=(1, 1))
    y_sq_sum = torch.conv2d(y_sq, sum_filter, padding=1, stride=(1, 1))
    mul_sum = torch.conv2d(inputs_mul, sum_filter, padding=1, stride=(1, 1))
    win_size = n ** 2
    u_1 = x_sum / win_size
    u_2 = y_sum / win_size
    cross = mul_sum - u_2 * x_sum - u_1 * y_sum + u_1 * u_2 * win_size
    var1 = x_sq_sum - 2 * u_1 * x_sum + u_1 * u_1 * win_size
    var2 = y_sq_sum - 2 * u_2 * y_sum + u_2 * u_2 * win_size
    cc = cross * cross / (var1 * var2 + np.finfo(float).eps)
    return torch.mean(cc)


def cross_correlation_loss(x, y, n, use_gpu=False):
    """ Loss function based on cross-correlation."""
    return 1. - cross_correlation(x, y, n, use_gpu)


def ncc(x, y):
    """Normalized cross-correlation metric computation."""
    mean_x = torch.mean(x, [1, 2, 3], keepdim=True)
    mean_y = torch.mean(y, [1, 2, 3], keepdim=True)
    mean_x2 = torch.mean(torch.pow(x, 2), [1, 2, 3], keepdim=True)
    mean_y2 = torch.mean(torch.pow(y, 2), [1, 2, 3], keepdim=True)
    stddev_x = torch.sum(torch.sqrt(
        mean_x2 - torch.pow(mean_x, 2)), [1, 2, 3], keepdim=True)
    stddev_y = torch.sum(torch.sqrt(
        mean_y2 - torch.pow(mean_y, 2)), [1, 2, 3], keepdim=True)
    val = torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))
    return val


def ncc_loss(x, y):
    """ Loss function based on normalized cross-correlation."""
    return 1. - ncc(x, y)


def smooothing_loss(input_value):
    """Smooothing loss based on derivative computation"""
    dy = torch.abs(input_value[:, :, 1:, :] - input_value[:, :, :-1, :])
    dx = torch.abs(input_value[:, :, :, 1:] - input_value[:, :, :, :-1])
    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def gradient(x):
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def deformation_smoothness_loss(flow):
    """
    Computes a deformation smoothness based loss as described here:
    https://link.springer.com/content/pdf/10.1007%2F978-3-642-33418-4_16.pdf
    """

    dx, dy = gradient(flow)

    dx2, dxy = gradient(dx)
    dyx, dy2 = gradient(dy)

    integral = torch.mul(dx2, dx2) + torch.mul(dy2, dy2) + torch.mul(dxy, dxy) + torch.mul(dyx, dyx)
    loss = torch.sum(integral, [1, 2, 3]).mean()
    return loss


def dice_score(x, y):
    """
    Dice metric computation.
    """
    top = 2 * torch.sum(x * y, [1, 2, 3])
    union = torch.sum(x + y, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    return dice


def dice_loss(x, y):
    """ Loss function based on dice score."""
    return 1. - dice_score(x, y)


def ssim_loss(x, y):
    """ Loss function based on SSIM metric."""
    ssim_measure = pytorch_ssim.SSIM()
    ssim_out = 1 - ssim_measure(x, y)
    return ssim_out


def L2Def(pred, target):
    """L2 norm computation. """
    loss = ((pred - target) ** 2).sum(dim=3) ** 0.5
    return loss.mean()


def mse(x, y):
    """Mean-squared error."""
    return torch.sum((x - y) ** 2, [1, 2, 3]).mean()


def construct_loss(loss_names, weights=None, n=9, sm_lambda=0.01, use_gpu=False, def_lambda=10, use_masks=True):
    """Construct loss from loss names."""
    if weights is None:
        weights = [1.] * len(loss_names)

    loss = []

    loss.append(lambda x, y, dx, dy: def_lambda * L2Def(dx, dy))
    for loss_item, w in zip(loss_names, weights):
        if loss_item == 'cross-corr':
            if use_masks:
                loss.append(lambda x, y, dx, dy: w * cross_correlation_loss(x[:, :1], y[:, :1], n, use_gpu))
            else:
                loss.append(lambda x, y, dx, dy: w * cross_correlation_loss(x, y, n, use_gpu))
        elif loss_item == 'ncc':
            loss.append(lambda x, y, dx, dy: w * ncc_loss(x, y))
        elif loss_item == 'dice':
            loss.append(lambda x, y, dx, dy: w * dice_loss(x, y))
        elif loss_item == 'ssim':
            loss.append(lambda x, y, dx, dy: w * ssim_loss(x, y))
        elif loss_item == 'mse':
            loss.append(lambda x, y, dx, dy: w * mse(x, y))
        elif loss_item == 'smooth':
            loss.append(lambda x, y, dx, dy: sm_lambda * smooothing_loss(x))
        else:
            raise NameError(f"No loss function named {loss_item}.")
    return lambda x, y, dx, dy: [sum(lo(x, y, dx, dy) for lo in loss), sum(lo(x, y, dx, dy) for lo in loss[1:]),
                                 loss[0](x, y, dx, dy)]


def custom_total_loss(reg, fixed, deform, diff):
    l_crosscorr = ncc_loss(reg, fixed)
    print('Normalized correlation loss', l_crosscorr)

    l2dif = (torch.sum(diff**2, [1,2,3])**0.5).mean()
    print('L2 diff affine loss', l2dif)

    l2 = (torch.sum((reg-fixed) ** 2, [1, 2, 3])).mean()
    print('L2 diff loss', l2)

    im_dice = dice_loss(reg, fixed)
    print('Im dice loss', im_dice)

    l_smooth = deformation_smoothness_loss(deform)
    print('Smooth loss', l_smooth)
    print()
    return l_crosscorr + im_dice + 0.0005*l_smooth + 0.03*l2dif + 0.007*l2, l_crosscorr