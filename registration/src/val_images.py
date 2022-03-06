import os
import numpy as np
from skimage import io


def validate_images(images1, images2, images3, val_dir='val_images/', epoch=0, train=False):
    """Saves the results of registration to monitor the net training."""
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    images1 = images1.cpu().detach().numpy()
    images2 = images2.cpu().detach().numpy()
    images3 = images3.cpu().detach().numpy()
    for i, (im1, im2, im3) in enumerate(zip(images1, images2, images3)):
        im1 *= 255.
        im2 *= 255.
        im3 *= 255.
        im1 = im1[0].astype('uint8')
        im2 = im2[0].astype('uint8')
        im3 = im3[0].astype('uint8')
        if train:
            io.imsave(val_dir + f'train_epoch{epoch}_{i}_1.jpg', im1)
            io.imsave(val_dir + f'train_epoch{epoch}_{i}_2.jpg', im2)
            io.imsave(val_dir + f'train_epoch{epoch}_{i}_3.jpg', im3)
        else:
            io.imsave(val_dir + f'val_epoch{epoch}_{i}_1.jpg', im1)
            io.imsave(val_dir + f'val_epoch{epoch}_{i}_2.jpg', im2)
            io.imsave(val_dir + f'val_epoch{epoch}_{i}_3.jpg', im3)

        i += 1


def validate_deformations(def1, def2, val_dir='val_images/', epoch=0, train=False):
    """Saves the deformation fields during training to monitor the net training."""
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    def1 = def1.cpu().detach().numpy()
    def2 = def2.cpu().detach().numpy()

    for i, (d1, d2) in enumerate(zip(def1, def2)):
        minim = min(np.min(d1), np.min(d2))
        maxim = max(np.max(d1), np.max(d2))
        d1 = (d1 - minim) / (maxim - minim)
        d2 = (d2 - minim) / (maxim - minim)
        d1 *= 255.
        d2 *= 255.
#        im3 *= 255.
        d1 = d1.astype('uint8')
        d2 = d2.astype('uint8')
        im1 = np.concatenate([d1[0], d2[0]], axis=1)
        im2 = np.concatenate([d1[1], d2[1]], axis=1)
        im = np.concatenate([im1, im2], axis=0)
        if train:
            io.imsave(val_dir + f'train_epoch{epoch}_{i}_def.jpg', im)
        else:
            io.imsave(val_dir + f'val_epoch{epoch}_{i}_def.jpg', im)
        
        i += 1
