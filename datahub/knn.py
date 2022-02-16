import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error

def get_weight(D):
    w = 1/(D+1)
    w = w/np.sum(w, axis=1, keepdims=True)
    return w

def get_k_ind(fp, td, k=3, metric='euclidean', ap_mask=None, is_weighted=False, is_return_detail=False):
    k = int(min(k, len(fp.rssis)))
    fp_rssis = np.array(fp.rssis)
    td_rssis = np.array(td.rssis)
    if ap_mask is not None:
        _fp_rssis = fp_rssis[:,ap_mask]
        _td_rssis = td_rssis[:,ap_mask]
    else:
        _fp_rssis = fp_rssis
        _td_rssis = td_rssis
    if type(metric) == str:
        D = cdist(_td_rssis, _fp_rssis, metric=metric)
    else:
        D = metric(_td_rssis, _fp_rssis)
    ind = np.argsort(D)
    return (ind[:, :k], get_weight(D[:, :k]) if is_weighted else None, D[:, :k] if is_return_detail else None)

def knn(fp, td, k=3, metric='euclidean', ap_mask=None, is_weighted=False, is_return_detail=False):
    (k_ind,w,k_D) = get_k_ind(fp, td, k=k, metric=metric, ap_mask=ap_mask, is_weighted=is_weighted, is_return_detail=is_return_detail)
    k_cdns = np.array(fp.cdns).astype(np.float)[k_ind,:]
    if is_weighted and k>1:
        w = np.expand_dims(w, axis=2)
        td_cdns = np.sum(k_cdns*w, axis=1)
    else:
        td_cdns = np.mean(k_cdns, axis=1)
    if is_return_detail:
        return (td_cdns, k_ind, k_cdns, k_D)
    else:
        return td_cdns

def evaluate_knn(fp, td, k=3, metric='euclidean', ap_mask=None, is_weighted=False, is_return_ind=False):
    (td_cdns, k_ind) = knn(fp, td, k, metric=metric, ap_mask=ap_mask, is_weighted=is_weighted, is_return_ind=True)
    test_errs = error(td_cdns, td.cdns)
    return (test_errs, k_ind) if is_return_ind else test_errs

def _knn(fp, rssis, k=3, ap_mask=None):
    fp_rssis = fp.rssis
    k = int(min(k, fp.rssis.shape[0]))
    if ap_mask is not None:
        fp_rssis = fp_rssis[:,ap_mask]
        rssis = rssis[ap_mask]
    D = np.sqrt(np.sum((fp_rssis-rssis)**2, axis=1))
    ind = np.argsort(D)
    return ind[:k]

def _knn_cdn(fp, rssis, k=3):
    k_ind=_knn(fp, rssis, k=k)
    k_cdns = fp.cdns[k_ind,:]
    return np.mean(k_cdns, axis=0)

def _evaluate_knn(fp, td, k=3):
    td_cdns = []
    for rssis in td.rssis:
        td_cdns.append(_knn_cdn(fp, rssis))
    return error(td_cdns, td.cdns)

def single_cknn(fp, rssis, k1=10, k2=3, T=15, max_iterations=5):
    ap_mask = None
    for i in range(max_iterations):
        k_ind=_knn(fp, rssis, k=k1, ap_mask=ap_mask)
        rssis_mean = np.mean(np.vstack((fp.rssis[k_ind, :],  np.expand_dims(rssis, axis=0))), axis=0) - np.min(fp.rssis)
        rssis_mean[rssis_mean==0] = 1
        rssis_diff = np.mean(np.abs((fp.rssis[k_ind, :] - np.expand_dims(rssis, axis=0))), axis=0)
        _ap_mask = rssis_diff<T
        if (ap_mask is not None) and (np.sum(ap_mask)==np.sum(_ap_mask)):
            break
        else:
            ap_mask = _ap_mask
    k_cdns = fp.cdns[k_ind[:k2],:]
    return np.mean(k_cdns, axis=0), ap_mask

def cknn(fp, td_rssis, k1=10, k2=3, T=15):
    td_cdns = []
    ap_masks = []
    for rssis in td_rssis:
        td_cdn, ap_mask = single_cknn(fp, rssis, k1=k1, k2=k2, T=T, max_iterations=5)
        td_cdns.append(td_cdn)
        ap_masks.append(ap_mask)
    return td_cdns, ap_masks
    
def evaluate_cknn(fp, td, k1=10, k2=3, T=15):
    td_cdns, ap_masks = cknn(fp, td.rssis, k1=k1, k2=k2, T=T)
    return error(td_cdns, td.cdns), td_cdns, ap_masks

def error(a, b):
    return np.linalg.norm(a-b, axis=-1)

# backup
def single_knn_non(fp, rssis, min_rssi, k=3):
    k = int(min(k, len(fp.rssis)))
    fp_rssis = np.array(fp.rssis)
    td_rssis = np.array(rssis)
    td_nonzero_mask = td_rssis==min_rssi
    fp_rssis = fp_rssis[:,td_nonzero_mask]
    td_rssis = td_rssis[td_nonzero_mask]
    D = np.sqrt(np.sum((fp_rssis-td_rssis)**2, axis=1))
    ind = np.argsort(D)
    return ind[:k]

def evaluate_knn_non(fp, td, k=3):
    td_cdns = []
    min_rssi = np.min(fp.rssis)
    for rssis in td.rssis:
        k_ind=single_knn_non(fp, rssis, k=k, min_rssi=min_rssi)
        k_cdns = np.array(fp.cdns).astype(np.float)[k_ind,:]
        td_cdns.append(np.mean(k_cdns, axis=0))
    td_cdns = np.array(td_cdns)
    return np.linalg.norm(td_cdns-np.array(td.cdns).astype(np.float),axis=1)

def get_ap_mask_reduce(ap_num, aps):
    ap_mask = np.ones(ap_num, dtype=bool)
    ap_mask[aps] = False
    return ap_mask