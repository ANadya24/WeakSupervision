from skimage.color import gray2rgb
from skimage import io
from skimage import exposure as exp
import numpy as np
import torch
import math
import pickle
from torch import nn

from segmentation.data_manipulation.utils import normalize
from segmentation.draw_and_show.show_utils import plot_triplet
from segmentation.prediction.crop_utils import imToCrop, cropToIm


def predict_item(image, model, divide_by=32, normalize_func=normalize, device="cpu"):
    """
    Предсказание одной картинки с предобработкой.
    """

    tensor_image = normalize_func(image)
    if len(image.shape) != 3:
        tensor_image = gray2rgb(tensor_image)

    h, w = tensor_image.shape[:2]
    pad_x = 0
    pad_y = 0

    if math.modf(h / divide_by)[0] != 0:
        new_h = (math.modf(h / divide_by)[1] + 1) * divide_by
        pad_y = int((new_h - h)//2)

    if math.modf(w / divide_by)[0] != 0:
        new_w = (math.modf(w / divide_by)[1] + 1) * divide_by
        pad_x = int((new_w - w)//2)

    tensor_image = np.pad(tensor_image, np.array([pad_y, pad_y, pad_x, pad_x, 0, 0]).reshape(3, 2))
    tensor_image = torch.Tensor(tensor_image).permute(2, 0, 1)[None].to(device)
    res_mask = model(tensor_image)
    res_mask = res_mask.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy()

    if pad_x or pad_x:
        res_mask = res_mask[pad_y: pad_y + h, pad_x: pad_x + w]

    return res_mask


def predict_items(images, model, divide_by=32, normalize_func=normalize, device="cpu"):
    """
    Предсказание одной картинки с предобработкой.
    """
    if isinstance(images, list):
        image_batch = np.stack(images)
    else:
        image_batch = images
    tensor_image_batch = normalize_func(image_batch)

    h, w = tensor_image_batch.shape[1:3]
    pad_x = 0
    pad_y = 0

    if math.modf(h / divide_by)[0] != 0:
        new_h = (math.modf(h / divide_by)[1] + 1) * divide_by
        pad_y = (new_h - h) // 2

    if math.modf(w / divide_by)[0] != 0:
        new_w = (math.modf(w / divide_by)[1] + 1) * divide_by
        pad_x = (new_w - w) // 2

    tensor_image_batch = np.pad(tensor_image_batch, np.array([0, 0, pad_y, pad_y, pad_x, pad_x, 0, 0]).reshape(3, 2))
    tensor_image_batch = torch.Tensor(tensor_image_batch).permute(0, 3, 1, 2).to(device)
    res_mask = model(tensor_image_batch)
    res_mask = res_mask.permute(0, 2, 3, 1).cpu().detach().numpy()

    if pad_x or pad_x:
        res_mask = res_mask[:, pad_y: pad_y + h, pad_x: pad_x + w]

    return res_mask


def predict_and_plot(img_label_pairs_fluo, model, num_frames=-1, device="cpu"):
    if num_frames == -1:
        num_frames = len(img_label_pairs_fluo)

    for img, mask in img_label_pairs_fluo[:num_frames]:
        res_mask = predict_item(img, model, device=device)
        markers = res_mask[:,:,1]
        res_mask = res_mask[:,:,0]
        plot_triplet(res_mask, mask>0, markers>0, 15)


def predict_and_save(img_label_pairs_fluo, model, out_dir, num_frames=-1, device="cpu",
                     adjust_gamma=False, gamma=0.7):

    res_masks = []
    if num_frames == -1:
        num_frames = len(img_label_pairs_fluo)

    for i, (img, mask) in enumerate(img_label_pairs_fluo[:num_frames]):
        if adjust_gamma:
            img = exp.adjust_gamma(img, gamma)
        res_mask = predict_item(img, model, device=device)
        io.imsave(out_dir.replace('.pkl', f'_{i}.png'), np.uint8(res_mask[:, :, 0]*255))
        res_masks.append(res_mask)
    with open(out_dir, 'wb') as file:
        pickle.dump(res_masks, file)
    return res_masks


def predict_crop_and_save(img_label_pairs_fluo, model, out_dir, num_frames=-1, device="cpu"):

    res_masks = []
    if num_frames == -1:
        num_frames = len(img_label_pairs_fluo)

    for i, (img, mask) in enumerate(img_label_pairs_fluo[:num_frames]):

        normalized_image = gray2rgb(normalize(img))
        crops = imToCrop(normalized_image, (480, 480, 3))
        res_mask_crops = predict_items(crops, normalize_func=lambda x: x, device=device)
        res_mask = cropToIm(res_mask_crops, img.shape)
        io.imsave(out_dir.replace('.pkl', f'_{i}.png'), np.uint8(res_mask[:, :, 0]*255))
        res_masks.append(res_mask)
    with open(out_dir, 'wb') as file:
        pickle.dump(res_masks, file)
    return res_masks


def load_checkpoint_w_model(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.module.to('cpu')
    model.eval()
    return model


def load_checkpoint(model, filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.module.to('cpu')
    model.eval()
    return model

