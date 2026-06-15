# Adapted from https://github.com/AmazingDD/daisyRec/blob/dev/daisy/utils/metrics.py#L38

import numpy as np 

        
def Precision(pred, gt):
    return np.in1d(pred, list(gt)).sum() / len(pred)

def Recall(pred, gt):
    return np.in1d(pred, list(gt)).sum() / len(gt)

def MRR(pred, gt):
    mrr = 0.
    for index, item in enumerate(pred):
        if item in gt:
            mrr = 1 / (index + 1)
            break
    return mrr

def MAP(pred, gt):
    r = np.in1d(pred, list(gt))
    out = [r[:k+1].sum() / (k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0
    return np.mean(out)

def NDCG(pred, gt):
    def DCG(r):
        r = np.asfarray(r) != 0
        if r.size:
            dcg = np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
            return dcg
        return 0.

    r = np.in1d(pred, list(gt))
    idcg = DCG(sorted(r, reverse=True))
    if not idcg:
        ndcg = 0.
    else:
        ndcg = DCG(r) / idcg

    return ndcg


def HR(pred, gt):
    
    r = np.in1d(pred, list(gt))
    if r.sum():
        return 1
    return 0

def AUC(pred, gt):
    r = np.in1d(pred, list(gt))
    pos_num = r.sum()
    neg_num = len(pred) - pos_num

    pos_rank_num = 0
    for j in range(len(r) - 1):
        if r[j]:
            pos_rank_num += np.sum(~r[j + 1:])

    if pos_num * neg_num == 0:
        return 0
    auc = pos_rank_num / (pos_num * neg_num)
    return auc


def F1(pred, gt):
    r = np.in1d(pred, list(gt))
    pre = r.sum() / len(pred)
    rec = r.sum() / len(gt)
    if pre + rec == 0:
        return 0
    f1 = 2 * pre * rec / (pre + rec)
    return f1
