import numpy as np
import torch


def dice_score(pr, gt):
    pr = np.asarray(pr).astype(np.bool)
    gt = np.asarray(gt).astype(np.bool)

    intersection = np.logical_and(pr, gt)

    score = 2. * intersection.sum() / (pr.sum() + gt.sum())

    return score


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):

    activation_fn = torch.nn.Sigmoid()

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
        / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

