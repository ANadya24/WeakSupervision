import os
import argparse
import json
import shutil

import torch
from torch.utils import data
from torch import optim, nn

from ctc_dataset import Dataset
from model import RegNet
from train import train
from losses import custom_total_loss

use_gpu = torch.cuda.is_available()


def copytree(src, dst, symlinks=False, ignore=None):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(e)
                os.unlink(d)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    data_path = config["data_path"]
    data_path_val = config["data_path_val"]
    batch_size = config["batch_size"]
    max_epochs = config["num_epochs"]
    load_epoch = config["load_epoch"]
    model_path = config["model_path"]
    lr = config["learning_rate"]
    sm_lambda = config["smooth"]
    data_shape = config["data_shape"]
    num_workers = config["num_workers"]
    save_dir = config["save_dir"]
    logdir = config["log_dir"]
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    shutil.copy(args.config_file, save_dir + args.config_file.split('/')[-1])
    copytree('./', save_dir + '/src/')
    save_step = config["save_step"]
    image_dir = config["image_dir"]
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
    model_name = config["model_name"]
    use_tensorboard = config["tensorboard"]
    use_gpu = config["use_gpu"]

    device = torch.device("cuda" if use_gpu else "cpu")
    model = RegNet(2, image_size=data_shape[1], device=device)

    checkpoint = None
    if load_epoch:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfuly loaded state_dict from {model_path}")

    if use_gpu:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to('cuda')

    training_set = Dataset(data_path, data_shape,
                           smooth=False, train=True, shuffle=True)
    print("Length of train set", len(training_set))

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'pin_memory': True}
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': num_workers,
              'pin_memory': True}
    validation_set = Dataset(data_path_val, data_shape,
                             smooth=False, train=False, shuffle=False)
    validation_generator = data.DataLoader(validation_set, **params)
    print("Length of validation set", len(validation_set))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if load_epoch:
        assert checkpoint is not None
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss = custom_total_loss

    #    loss = construct_loss(["cross-corr"], weights=[1.], sm_lambda=sm_lambda,
    #                                use_gpu=use_gpu, n=9, def_lambda=def_lambda)

    if load_epoch:
        loss = checkpoint['loss']

    train(load_epoch, max_epochs, training_generator, validation_generator, model, optimizer,
          device, loss, save_dir, model_name, image_dir, save_step, use_gpu, use_tensorboard=use_tensorboard,
          logdir=logdir)
