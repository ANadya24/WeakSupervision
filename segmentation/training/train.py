import torch
import time
import os
import copy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from loss import iou


def train_model(model, criterion, optimizer, scheduler, dataloaders, save_dir,
                device, num_epochs=25, prefix='stage1.', l2_lambda=0.01):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + '/logs/', exist_ok=True)
    os.makedirs(save_dir + '/checkpoints/', exist_ok=True)

    writer = SummaryWriter(save_dir + './logs/')

    since = time.time()

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_iou = 0.0
    best_loss = 100000.0
    total_batch_iter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            iou_metric = 0.0
            total = 0
            for batch in tqdm(dataloaders[phase]):
                inputs, labels, gt_weights = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                gt_weights = gt_weights.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    if 'weights_loss' in criterion:
                        loss = criterion['dice_loss'](logits, labels.float()) + \
                               criterion['weights_loss'](logits, labels.float(), gt_weights.float())
                    else:
                        loss = criterion['dice_loss'](logits, labels.float()) + \
                               criterion['ce_loss'](logits, labels.float())

                    l2_reg = torch.tensor(0.).to(device)
                    for param in model.module.parameters():
                        l2_reg += torch.norm(param)
                    loss += l2_lambda * l2_reg
                    if torch.isnan(loss).sum() > 0:
                        model.module.load_state_dict(best_model_wts)
                        writer.close()
                        return model

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                iou_item = iou(logits, labels.float())

                writer.add_scalar(f"Loss/{phase}", loss.item(), total_batch_iter)
                writer.add_scalar(f"IOU/{phase}", iou_item, total_batch_iter)
                total_batch_iter += 1

                iou_metric += iou_item * inputs.size(0)

                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / total
            epoch_iou = iou_metric / total

            print('{} Loss: {:.4f} IOU: {:.4f}'.format(
                phase, epoch_loss, epoch_iou))
            if phase == 'val' and epoch_iou > best_iou:
                if os.path.exists(save_dir + f'/checkpoints/{prefix}best.pth'):
                    os.rename(save_dir + f'/checkpoints/{prefix}best.pth', save_dir +
                              f'/checkpoints/{prefix}prev_best.pth')
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.module.state_dict())
                torch.save(best_model_wts, save_dir + f'/checkpoints/{prefix}best.pth')
                checkpoint = {'model': model, 'state_dict': model.module.state_dict()}
                torch.save(checkpoint, save_dir + f'/checkpoints/{prefix}best_full.pth')

            if phase == 'val' and epoch_loss < best_loss:
                if os.path.exists(save_dir + f'/checkpoints/{prefix}l.best.pth'):
                    os.rename(save_dir + f'/checkpoints/{prefix}l.best.pth',
                              save_dir + f'./checkpoints/{prefix}l.prev_best.pth')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.module.state_dict())
                torch.save(best_model_wts, save_dir + f'/checkpoints/{prefix}l.best.pth')
                checkpoint = {'model': model, 'state_dict': model.module.state_dict()}
                torch.save(checkpoint, save_dir + f'/checkpoints/{prefix}l.best_full.pth')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_iou))

    # load best model weights
    model.module.load_state_dict(best_model_wts)
    writer.close()
    return model