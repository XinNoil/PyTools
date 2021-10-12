import os
import numpy as np
from mtools import np_avg,np_intersect
from .h5db import DB
from scipy.spatial.distance import cdist

def get_filenames(folderlist, filenumlist, prefix):
    filenames = []
    for di,num in zip(folderlist, filenumlist):
        for fi in range(num):
            filenames.append(os.path.join('%d'%di, '%s%d.txt'% (prefix,fi)))
    return filenames

def reduce_dbs(db1, db2):
    if not db1.is_cdns_equal(db2.cdns):
        ia, ib = np_intersect(np.round(db1.cdns, 3), np.round(db2.cdns, 3))
        print('reduce_dbs(%s_%s, %s_%s): reduce %d,%d to %d,%d' %(db1.data_name, db2.dbtype, db2.data_name, db2.dbtype, len(db1), len(db2), np.sum(ia), np.sum(ib)))
        db1.set_mask(ia)
        db2.set_mask(ib)
    else:
        print('reduce_dbs(%s_%s, %s_%s): equal' %(db1.data_name, db2.dbtype, db2.data_name, db2.dbtype))

def set_value(rssi, m, v):
    rssi[m] = v
    return rssi

def get_filtered_gen(fp, gen, rss_range=[0,1], k=3):
    D = cdist(np.array(gen.rssis), np.array(fp.rssis))
    inds = np.argsort(D)
    k_inds = inds[:, :k]
    masks= [np.all(fp.rssis[k_ind] == rss_range[0], 0) for k_ind in k_inds]
    gen.rssis = np.vstack([set_value(rssi, mask, rss_range[0]) for mask, rssi in zip(masks, gen.rssis)])
    gen.rssis = cut_rssis(gen.rssis)
    return gen

def cut_rssis(rssis, rss_range=[0,1]):
    rssis[rssis<rss_range[0]] = rss_range[0]
    rssis[rssis>rss_range[1]] = rss_range[1]
    return rssis

def cut_record_db(db, cut_num):
    ind = np.zeros(db.rssis.shape[0], dtype=bool)
    start_ind = np.cumsum(db.RecordsNums) - db.RecordsNums
    end_ind = np.cumsum(db.RecordsNums)-1
    for i in range(cut_num):
        tmp_ind = np.min(np.stack((start_ind+i, end_ind)), axis=0)
        ind[tmp_ind] = True
    db.RecordsNums = np.min(np.stack((db.RecordsNums, np.zeros_like(db.RecordsNums)+cut_num)), axis=0)
    db.cdns = db.cdns[ind]
    db.rssis = db.rssis[ind]
    db.rp_no = db.rp_no[ind]
    db.rssis_avg = np_avg(db.rssis, db.RecordsNums)

def aug_db_i(db, aug_num, i, is_new=False, newdb=None):
    rssis = db.rssis[db.start_ind[i]:db.end_ind[i]]
    cdn = db.cdns[db.start_ind[i]]
    rp_no = db.rp_no[db.start_ind[i]]
    RecordsNum = db.RecordsNums[i]
    
    aug_ind_0 = np.random.randint(0, RecordsNum, size=(aug_num, rssis.shape[1]))
    aug_ind_1 = np.repeat(np.expand_dims(np.arange(rssis.shape[1]), axis=0),aug_num,axis=0)
    aug_rssis = rssis[aug_ind_0, aug_ind_1]
    aug_cdns = np.repeat(np.expand_dims(cdn,axis=0), aug_num, axis=0)
    aug_rp_no = np.repeat(rp_no, aug_num)
    if is_new:
        if len(newdb):
            newdb.rssis = np.vstack((newdb.rssis, aug_rssis))
            newdb.cdns = np.vstack((newdb.cdns, aug_cdns))
            newdb.rp_no = np.hstack((newdb.rp_no, aug_rp_no))
        else:
            newdb.rssis = aug_rssis
            newdb.cdns = aug_cdns
            newdb.rp_no = aug_rp_no
    else:
        db.rssis = np.insert(db.rssis, db.end_ind[i], aug_rssis, axis=0)
        db.cdns = np.insert(db.cdns, db.end_ind[i], aug_cdns, axis=0)
        db.rp_no = np.insert(db.rp_no, db.end_ind[i], aug_rp_no)
        db.RecordsNums[i] = db.RecordsNums[i]+aug_num

def aug_db(db, aug_num, is_new=False):
    set_ind(db)
    if is_new:
        newdb = DB(is_empty=True)
        for i in range(db.RecordsNums.shape[0]):
            aug_db_i(db, aug_num, i, is_new=True, newdb=newdb)            
        newdb.RecordsNums = np.zeros_like(db.RecordsNums)+aug_num
        return newdb
    else:
        for i in reversed(range(db.RecordsNums.shape[0])):
            aug_db_i(db, aug_num, i)
        set_ind(db)
    
def set_ind(db):
    db.end_ind = np.cumsum(db.RecordsNums)
    db.start_ind = db.end_ind - db.RecordsNums