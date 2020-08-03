import numpy as np
import torch


def to_numpy(x):
    if not x.device == 'cpu':
        return x.cpu().detach().numpy()
    else:
        return x.detach().numpy()


def count_nonzero(x):
    """
    x: np.array of shape (1, ) or (2, ?)
    """
    return len(np.transpose(np.nonzero(x)))


def mae(pred, gt):
    """
    pred: np.array (B, 1, H, W)
    gt: np.array(B, 1, H, W)
    """
    pred[pred<=0.5] = 0.0
    pred[pred>0.5] = 1.0
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    mae = np.mean(np.abs(pred-gt))
    return float(mae)


def precision(pred, gt):
    """
    Defined as portion of correctly classified fg pixels.
    tp / tp + fp
    pred: np.array (B, 1, H, W)
    gt: np.array(B, 1, H, W)
    """
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    pred[pred<=0.5] = 0.0
    pred[pred>0.5] = 1.0
    diff = pred * gt
    prec = 0.0
    for im in range(pred.shape[0]):
        tp = count_nonzero(gt[im])
        prec += count_nonzero(diff[im]) / (count_nonzero(pred[im]) + 1e-6)

    return prec / pred.shape[0]


def recall(pred, gt):
    """
    Defined as the portion of
    tp / tp + fn
    pred: np.array size (B, 1, H, W)
    gt: np.array size (B, 1, H, W)
    """
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    pred[pred<=0.5] = 0.0
    pred[pred>0.5] = 1.0
    diff = pred * gt
    recall = 0.0
    for im in range(pred.shape[0]):
        recall += count_nonzero(diff[im]) / (count_nonzero(gt[im]) + 1e-6)

    return recall / pred.shape[0]


def f1(precision, recall, beta2):
    """
    precision: float
    recall: float
    beta: float
    """
    try:
        f1_score = (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall)
    except:
        f1_score = 0.0
        with open('error.txt', 'a+') as f:
            print(f'e: {precision}, {recall}')
            f.write(f'e: {precision}, {recall}\n')
    return f1_score
