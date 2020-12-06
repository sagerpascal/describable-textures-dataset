import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml


conf = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
num_classes = conf['num_classes']

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    __name__ = 'dice-loss'

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class WeightedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    '''
    Weighted BCEWithLogitsLoss (weighted according mask distribution)
    '''
    __name__ = 'weighted-bce-with-logits-loss'

    def __init__(self, mask_size, **kwargs):
        pos_weight = np.ones((num_classes, mask_size[0], mask_size[1]))
        weights = [0.956569204, 0.977921196, 1.408709632, 1.084427464, 1.315641729, 0.946239083, 0.952410208,
                   0.905183255, 1.038172265, 0.82662018, 0.803869166, 0.995701581, 1.018857432, 1.114780396,
                   0.905183255, 1.000248164, 0.944199775, 1.043115942, 0.954485176, 1.092540388, 0.946239083,
                   0.938134252, 1.076434142, 0.825063457, 0.914631932, 0.842516722, 1.076434142, 1.040638232,
                   1.014140499, 1.055683604, 0.973574879, 1.103548352, 1.269880277, 1.120482598, 0.954485176,
                   0.876217391, 1.087118351, 0.847405601, 0.960764683, 1.505528164, 1.929994254, 0.857355569,
                   0.895927803, 0.82662018, 0.988958681, 1.055683604]
        for i, w in enumerate(weights):
            pos_weight[i, :, :] = pos_weight[i, :, :] * w
        pos_weight = torch.FloatTensor(pos_weight)
        super().__init__(pos_weight=pos_weight, **kwargs)

    def forward(self, y_pr, y_gt):
        return super(WeightedBCEWithLogitsLoss, self).forward(y_pr, y_gt)