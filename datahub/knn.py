import os
import numpy as np
from scipy.spatial.distance import cdist

def get_k_ind(fp, td, k=3, metric='euclidean', ap_mask=None):
    k = int(min(k, len(fp.rssis)))
    fp_rssis = np.array(fp.rssis)
    td_rssis = np.array(td.rssis)
    if ap_mask is not None:
        fp_rssis = fp_rssis[:,ap_mask]
        td_rssis = td_rssis[:,ap_mask]
    if type(metric) == str:
        D = cdist(td_rssis, fp_rssis, metric=metric)
    else:
        D = metric(td_rssis, fp_rssis)
    ind = np.argsort(D)
    return ind[:, :k]

def knn(fp, td, k=3, metric='euclidean', ap_mask=None):
    k_ind = get_k_ind(fp, td, k=k, metric=metric, ap_mask=ap_mask)
    k_cdns = np.array(fp.cdns).astype(np.float)[k_ind,:]
    return np.mean(k_cdns, axis=1)

def evaluate_knn(fp, td, k=3, metric='euclidean', ap_mask=None):
    td_cdns = knn(fp, td, k, metric=metric, ap_mask=ap_mask)
    return np.linalg.norm(td_cdns-np.array(td.cdns).astype(np.float),axis=1)

def single_knn(fp, rssis, min_rssi, k=3):
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
        k_ind=single_knn(fp, rssis, k=k, min_rssi=min_rssi)
        k_cdns = np.array(fp.cdns).astype(np.float)[k_ind,:]
        td_cdns.append(np.mean(k_cdns, axis=0))
    td_cdns = np.array(td_cdns)
    return np.linalg.norm(td_cdns-np.array(td.cdns).astype(np.float),axis=1)

def get_ap_mask_reduce(ap_num, aps):
    ap_mask = np.ones(ap_num, dtype=bool)
    ap_mask[aps] = False
    return ap_mask