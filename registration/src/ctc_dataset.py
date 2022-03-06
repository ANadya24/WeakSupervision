import os
import random
import numpy as np
import cv2
import albumentations as A
import torch
from torch.utils import data


def pad(img, padwidth):
    h0, h1, w0, w1 = padwidth[:4]
    if len(padwidth) < 5:
        img = np.pad(img, ((h0, h1), (w0, w1)))
    else:
        img = np.pad(img, ((h0, h1), (w0, w1), (0, 0)))

    return img


def normalize(im):
    im = im - im.min()
    im = im / (im.max() + 1e-10)
    return im


class Dataset(data.Dataset):
    def __init__(self, path, im_size=(1, 100, 100), smooth=False, train=True, shuffle=True, use_crop=False):
        """Initialization"""
        self.path = path
        self.folders = os.listdir(path)

        self.length = sum([len(os.listdir(path + f)) for f in self.folders])
        print('Dataset length is ', self.length)
        self.seq = []

        for f in self.folders:
            if len(glob(path + f + '/*.tif')) > 1:
                for i, im in enumerate(glob(path + f + '/*.tif')):
                    self.seq.append((im, i))

        self.use_crop = use_crop
        self.im_size = im_size[1:]
        self.smooth = smooth
        self.train = train
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.seq)

        if self.train:
            self.aug_pipe = A.Compose([A.HorizontalFlip(p=0.3), A.VerticalFlip(p=0.3),
                                       A.ShiftScaleRotate(shift_limit=0.0225,
                                                          scale_limit=0.1, rotate_limit=15, p=0.2)],
                                      additional_targets={'image2': 'image'})

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        im_path, it = self.seq[index]
        number = int(im_path.split('crop')[-1].split('.')[0])

        if self.train:
            number2 = np.random.randint(number - 3, number + 3)
            if not os.path.exists(im_path.split('crop')[0] + f'crop{number2}.tif'):
                if os.path.exists(im_path.split('crop')[0] + f'crop{number + 1}.tif'):
                    number2 = number + 1
                elif os.path.exists(im_path.split('crop')[0] + f'crop{number - 1}.tif'):
                    number2 = number - 1
                else:
                    # number2 = number
                    cur_numbers = [int(item.split('crop')[-1].split('.')[0]) for \
                                   item in glob(im_path.split('crop')[0] + '*.tif')]
                    diffs = [abs(num - number) for num in cur_numbers]
                    number2 = cur_numbers[np.argmin(diffs)]

        else:
            if os.path.exists(im_path.split('crop')[0] + f'crop{number + 1}.tif'):
                number2 = number + 1
            elif os.path.exists(im_path.split('crop')[0] + f'crop{number - 1}.tif'):
                number2 = number - 1
            else:
                cur_numbers = [int(item.split('crop')[-1].split('.')[0]) for \
                               item in glob(im_path.split('crop')[0] + '*.tif')]
                diffs = [abs(num - number) for num in cur_numbers]
                number2 = cur_numbers[np.argmin(diffs)]
        if random.random() < 0.1:
            number2 = number

        fixed_image = cv2.imread(im_path, -1)
        moving_image = cv2.imread(im_path.split('crop')[0] + f'crop{number2}.tif', -1)

        fixed_image = normalize(fixed_image)
        moving_image = normalize(moving_image)
        h, w = fixed_image.shape[:2]
        if h != w:
            if h < w:
                fixed_image = pad(fixed_image, ((w - h) // 2, w - (w - h) // 2, 0, 0))
            else:
                fixed_image = pad(fixed_image, (0, 0, (h - w) // 2, h - (h - w) // 2))
        h, w = moving_image.shape[:2]
        if h != w:
            if h < w:
                moving_image = pad(moving_image, ((w - h) // 2, w - (w - h) // 2, 0, 0))
            else:
                moving_image = pad(moving_image, (0, 0, (h - w) // 2, h - (h - w) // 2))

        if self.use_crop:
            x0 = np.random.randint(0, w - self.im_size[1])
            y0 = np.random.randint(0, h - self.im_size[0])
            fixed_image = fixed_image[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            moving_image = moving_image[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
        else:
            c = min(h, w)
            fixed_image = fixed_image[:c, :c]
            moving_image = moving_image[:c, :c]
            fixed_image = cv2.resize(fixed_image, tuple(self.im_size))
            moving_image = cv2.resize(moving_image, tuple(self.im_size))

        if self.train:
            arr = self.aug_pipe(image=fixed_image, image2=moving_image)
            fixed_image = arr['image']
            moving_image = arr['image2']
        fixed_image = torch.Tensor(fixed_image[None]).float()
        moving_image = torch.Tensor(moving_image[None]).float()
        return fixed_image, moving_image


if __name__ == '__main__':
    from glob import glob
    from matplotlib import pyplot as plt

    path = '../data/DIC-C2DH-HeLa/v2/train/'
    dataset = Dataset(path, (1, 256, 256),
                      smooth=True, train=True, shuffle=True)
    fixed, moving = dataset[0]
    print(fixed.shape, fixed.max())
    fixed = np.uint8(fixed.numpy().transpose((1, 2, 0)) * 255)
    moving = np.uint8(moving.numpy().transpose((1, 2, 0)) * 255)
    tmp = np.concatenate([fixed, moving], axis=1)
    print(tmp.shape)
    cv2.imwrite('test1.jpg', tmp)
