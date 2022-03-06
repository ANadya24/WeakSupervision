from torch import nn
import torch


def iou(pred, target):
    """ IOU metric calculation"""
    top = torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target - pred * target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    return dice


class FocalLoss(nn.Module):
    """Focal loss calculation"""
    def __init__(self, alpha=1, gamma=2, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        target = target.long()
        p_t = torch.where(target == 1, x, 1 - x)
        fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        fl = torch.where(target == 1, fl * self.alpha, fl)
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x


class BCEWeightsDigitLoss(nn.Module):
    """BCE with input weight presented by digit multiplication."""
    def __init__(self):
        super(BCEWeightsDigitLoss, self).__init__()
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, logits, targets, weights):
        loss = self.loss(logits, targets)
        loss = torch.mean(torch.mean(loss, dim=[1, 2, 3]) * weights)
        return loss


class BCEWeightsMapLoss(nn.Module):
    """BCE with input weight map multiplication."""
    def __init__(self):
        super(BCEWeightsMapLoss, self).__init__()
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, logits, targets, weights):
        loss = self.loss(logits, targets)
        loss = loss * weights
        loss = torch.mean(torch.mean(loss, dim=[1, 2, 3]))
        return loss

