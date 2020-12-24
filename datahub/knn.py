import os
import numpy as np
from scipy.spatial.distance import cdist

def knn(fp, td, k=3):
    k = int(min(k, len(fp.rssis)))
    D = cdist(np.array(td.rssis), np.array(fp.rssis))
    ind = np.argsort(D)
    k_ind = ind[:, :k]
    k_cdns = np.array(fp.cdns).astype(np.float)[k_ind,:]
    return np.mean(k_cdns, axis=1)

def evaluate_knn(fp, td, k=3):
    td_cdns = knn(fp, td, k)
    test_errs = np.linalg.norm(td_cdns-np.array(td.cdns).astype(np.float),axis=1)
    return test_errs
