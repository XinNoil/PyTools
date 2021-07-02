import os
import numpy as np
from scipy.spatial.distance import cdist

def get_k_ind(fp, td, k=3, metric='euclidean'):
    k = int(min(k, len(fp.rssis)))
    if type(metric) == str:
        D = cdist(np.array(td.rssis), np.array(fp.rssis), metric=metric)
    else:
        D = metric(np.array(td.rssis), np.array(fp.rssis))
    ind = np.argsort(D)
    return ind[:, :k]

def knn(fp, td, k=3, metric='euclidean'):
    k_ind = get_k_ind(fp, td, k=k, metric=metric)
    k_cdns = np.array(fp.cdns).astype(np.float)[k_ind,:]
    return np.mean(k_cdns, axis=1)

def evaluate_knn(fp, td, k=3, metric='euclidean'):
    td_cdns = knn(fp, td, k, metric=metric)
    test_errs = np.linalg.norm(td_cdns-np.array(td.cdns).astype(np.float),axis=1)
    return test_errs

def single_knn(fp, rssis, k=3, metric='euclidean'):
    k = int(min(k, len(fp.rssis)))
    import pdb;pdb.set_trace()
    if type(metric) == str:
        D = cdist(np.array(rssis), np.array(fp.rssis), metric=metric)
    else:
        D = metric(np.array(rssis), np.array(fp.rssis))
    import pdb;pdb.set_trace()

def evaluate_knn_non(fp, td, k=3, metric='euclidean'):
    for rssis in td.rssis:
        single_knn(fp, rssis, k=k, metric=metric)
    # test_errs = np.linalg.norm(td_cdns-np.array(td.cdns).astype(np.float),axis=1)
    # return test_errs