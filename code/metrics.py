
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import sparse
import bottleneck as bn
import random
import torch


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    hr = np.sum(right_pred)
    return {'recall': recall, 'precision': precis,'hr': hr}

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def test_one_batch(X, topks):
    sorted_items = X[0].cpu().numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    hr, ndcg = [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        hr.append(ret['hr'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'hr':np.array(hr), 
            'ndcg':np.array(ndcg)}
       

def outputPrediction(topks,pred_batch,predictUser): 
    _, rating_K = torch.topk(pred_batch, k=100)
    
    pred_batch = pred_batch.cpu().numpy()
    del pred_batch # 删掉cpu内存？

    groundTrueUser = [[u] for u in predictUser]
    pre_results = test_one_batch([rating_K, groundTrueUser],topks)
    pre_results['hr'] /= float(len(predictUser))
    pre_results['ndcg'] /= float(len(predictUser))
    HitRatio,HitRatio20,HitRatio50=pre_results['hr']
    NDCG,NDCG20,NDCG50=pre_results['ndcg']
    return HitRatio,HitRatio20,HitRatio50,NDCG,NDCG20,NDCG50



################### utils
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def set_seed(SEED):
    np.random.seed(SEED)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic=True
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

### old
def HR_at_k_batch(X_pred, heldout_batch, k=100):
    # print(heldout_batch)
    test_hits = 0
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1) # 从小到大，返回top的的index

    for i in range(batch_users):
        target_idx = heldout_batch[i]
        test_hits += target_idx in idx[i, :k]
    return test_hits/batch_users

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1) # 从小到大，返回top的的index

    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]# 前十个最大值的index

    tp = 1. / np.log2(np.arange(2, k + 2))


    test =np.arange(batch_users)[:, np.newaxis]
    pred = np.zeros(idx_topk.shape)
    for i in range(batch_users):
        # print(idx_topk[i])
        # print(heldout_batch[i])
        if heldout_batch[i] in idx_topk[i]:
            idx = list(idx_topk[i]).index(heldout_batch[i])
            pred[i][idx] = 1.
        
    DCG = (pred[np.arange(batch_users)[:, np.newaxis]] * tp).sum(axis=2).reshape(-1) # 1, 10 -> 1

    IDCG = np.array([(tp[:min(1, k)]).sum()
                     for n in range(batch_users)])
    # DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
    #                      idx_topk].toarray() * tp).sum(axis=1)
    # IDCG = np.array([(tp[:min(n, k)]).sum() +1.0
    #                  for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def evaluate(y_pred, y):
    # Array, csr_matrix
    # y = sparse.csr_matrix(y)
    hr10_list = []
    hr20_list = []
    hr50_list = []
    n10_list = []
    n20_list = []
    n50_list = []

    hr10 = HR_at_k_batch(y_pred, y, 10)
    hr20 = HR_at_k_batch(y_pred, y, 20)
    hr50 = HR_at_k_batch(y_pred, y, 50)
    n10 = NDCG_binary_at_k_batch(y_pred, y, 10)
    n20 = NDCG_binary_at_k_batch(y_pred, y, 20)
    n50 = NDCG_binary_at_k_batch(y_pred, y, 50)

    hr10_list.append([hr10])
    hr20_list.append([hr20])
    hr50_list.append([hr50])
    n10_list.append(n10)
    n20_list.append(n20)
    n50_list.append(n50)
 
    return np.mean(hr10_list), np.mean(hr20_list), np.mean(hr50_list), np.mean(n10_list), np.mean(n20_list), np.mean(n50_list)
