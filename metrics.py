
import torch
import numpy as np
import yaml


conf = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
loss = conf['loss']



def _preprocess(pr):
    if loss == 'cross-entropy':
        return torch.argmax(pr, dim=1)
    else:
        return pr

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _preprocess(pr)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


# def accuracyT5(pr, gt, ignore_channels=None):
#     pr_ = torch.argsort(pr, axis=1, descending=True)
#     gt_ = torch.argmax(gt, axis=1)
#     pr_, gt_ = _take_channels(pr_, gt_, ignore_channels=ignore_channels)
#
#     torch.logical_or()#TODO
#
#     tp = torch.sum(gt_ == pr_, dtype=pr_.dtype)
#     score = tp / gt_.view(-1).shape[0]
#     return score

def accuracyT1(pr, gt, ignore_channels=None):
    pr_ = torch.argmax(pr, dim=1)
    gt_ = gt if loss == 'cross-entropy' else torch.argmax(gt, dim=1)
    pr_, gt_ = _take_channels(pr_, gt_, ignore_channels=ignore_channels)

    tp = torch.sum(gt_ == pr_, dtype=pr_.dtype)
    score = tp / gt_.view(-1).shape[0]
    return score

def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    pr = _preprocess(pr)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _preprocess(pr)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _preprocess(pr)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


class Fscore(torch.nn.Module):

    __name__ = 'F-Score'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        return f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

class AccuracyT1(torch.nn.Module):
    __name__ = 'Top-1-Accuracy'

    def __init__(self, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        return accuracyT1(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )

class Accuracy(torch.nn.Module):
    __name__ = 'Accuracy'

    def __init__(self, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        return accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(torch.nn.Module):
    __name__ = 'Recall'

    def __init__(self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        return recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(torch.nn.Module):
    __name__ = 'Precision'

    def __init__(self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        return precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
