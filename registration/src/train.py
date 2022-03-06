import os
import signal
import torch
from torch.utils.tensorboard import SummaryWriter
from val_images import validate_images



def train_step(model, optimizer, device, loss_func, save_step, image_dir,
               batch_moving, batch_fixed, epoch=0):
    model.train()
    optimizer.zero_grad()

    batch_fixed, batch_moving = batch_fixed.to(device), batch_moving.to(device)
   
    registered_image, deform, _, diff = model(batch_moving, batch_fixed)

    train_loss, corr_loss = loss_func(registered_image.to('cpu'),
                                      batch_fixed.to('cpu'),
                                      deform.to('cpu'), diff.to('cpu'))

    train_loss.backward()

    optimizer.step()

    if (epoch + 1) % save_step == 0:
        validate_images(batch_fixed[1:3], batch_moving[1:3], registered_image[1:3],
                        image_dir, epoch=epoch+1, train=True)

    return train_loss, corr_loss


def test_step(model, device, loss_func, save_step, image_dir, batch_moving, batch_fixed, epoch=0):
    model.eval()

    with torch.no_grad():

        batch_fixed, batch_moving = batch_fixed.to(
            device), batch_moving.to(device)
            
        registered_image, deform, _, diff = model(batch_moving, batch_fixed)
        val_loss, corr_loss = loss_func(registered_image.to('cpu'), batch_fixed.to('cpu'),
                                        deform.to('cpu'), diff.to('cpu'))

        if (epoch + 1) % save_step == 0:
            validate_images(batch_fixed[1:3], batch_moving[1:3],
                            registered_image[1:3], image_dir, epoch=epoch + 1, train=False)

        return val_loss, corr_loss


def train(load_epoch, max_epochs, train_loader, val_loader, model, optimizer,
          device, loss_func, save_dir, model_name, image_dir, save_step, use_gpu,
          use_tensorboard=False, logdir="./logs/"):

    def save_model(dev_count, name=model_name + f'_stop'):
        if dev_count > 1:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func},
                save_dir + name + f'_{epoch + 1}')

        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func},
                save_dir + name + f'{epoch + 1}')
        print(f"Successfuly saved state_dict in {save_dir + model_name + f'_stop_{epoch + 1}'}")

    def sig_handler(signum, frame):
        print('Saved intermediate result!')
        torch.cuda.synchronize()
        save_model(torch.cuda.device_count())

    signal.signal(signal.SIGINT, sig_handler)
    # Loop over epochs
    best_loss = 1000
    if use_tensorboard:
        os.makedirs(logdir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=logdir)
        global_i = 0
        global_j = 0
    for epoch in range(load_epoch, max_epochs):
        train_loss = 0
        total = 0
        val_loss = 0

        for batch_fixed, batch_moving in train_loader:
            loss, corr_loss = train_step(model, optimizer, device, loss_func, save_step, image_dir, batch_moving, batch_fixed, epoch=epoch)
            train_loss += loss.item()
            total += 1
            if use_tensorboard:
                summary_writer.add_scalar('loss', loss.item(), global_i)
                summary_writer.add_scalar('corr_loss', corr_loss.item(), global_i)
                global_i += 1

        train_loss /= total

        # Testing time
        total = 0
        for batch_fixed, batch_moving in val_loader:
            loss, corr_loss = test_step(model, device, loss_func, save_step, image_dir,
                                        batch_moving, batch_fixed, epoch=epoch)
            val_loss += loss.item()
            total += 1
            if use_tensorboard:
                summary_writer.add_scalar('val_loss', loss.item(), global_j)
                summary_writer.add_scalar('val_corr_loss', corr_loss.item(), global_j)
                global_j += 1

        val_loss /= total
        print('Epoch', epoch + 1, 'train_loss/test_loss: ', train_loss, '/', val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(torch.cuda.device_count(), model_name + 'best')
            print('New best model from validation')

        if (epoch+1) % save_step == 0:
            save_model(torch.cuda.device_count(), model_name)
