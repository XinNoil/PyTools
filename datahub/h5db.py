import os,copy
import numpy as np
from datahub.wifi import get_bssids, process_rssis, WiFiData, set_bssids, get_ssids, normalize_rssis, unnormalize_rssis
from mtools import list_mask,list_con,load_h5,save_h5,load_json,save_json,csvread,np_avg,np_avg_std,np_intersect,np_repeat,np_mean_nonzero

class DB(object):
    def __init__(self, data_path='', data_name='', dbtype='db', filename='', \
                cdns=[], rssis=[], mags=[], bssids=[], RecordsNums=[], rp_no=[], db_no=[],\
                is_empty=False, is_print=True):
        if len(rssis) or is_empty:
            self.bssids = bssids
            self.cdns = cdns
            self.rssis = rssis
            self.mags = mags
            if len(RecordsNums):
                self.RecordsNums = RecordsNums
            else:
                self.RecordsNums = np.ones(len(self))
            self.rp_no = rp_no
            self.db_no = db_no
        else:
            self.data_path = data_path
            self.data_name = data_name
            self.dbtype = dbtype
            print(data_name)
            if os.path.exists(filename):
                print('load h5 file: %s'%filename)
                self.__dict__ = load_h5(filename)
                self.set_bssids(bssids)
            elif os.path.exists(self.save_name()):
                print('load h5 file: %s'%self.save_name())
                self.__dict__ = load_h5(self.save_name())
                self.set_bssids(bssids)
            else:
                raise Exception('DB is not existed: %s.'%(self.save_name() if filename=='' else filename))
        if is_print:
            self.print()
    
    def __len__(self):
        if type(self.rssis) == list:
            return len(self.rssis)
        else: # numpy array
            return self.rssis.shape[0] 
            
    def filename(self, postfix=None, ext=None):
        filename = self.data_name
        if postfix:
            filename = '%s_%s'%(filename, postfix)
        if ext:
            filename = '%s.%s'%(filename, ext)
        return os.path.join(self.data_path, filename)

    def save_name(self):
        return self.filename(postfix=self.dbtype, ext='h5')
    
    def save_h5(self, filename, is_print=True):
        save_h5(filename, self)
        if is_print:
            print('save to %s' % filename)
    
    def get_feature(self, feature_mode='R'):
        if feature_mode == 'R':
            return normalize_rssis(self.rssis) if np.max(self.rssis)<0 else self.rssis
        elif feature_mode == 'MM':
            return self.mags
        elif feature_mode == 'RMM':
            return np.hstack((self.mags, normalize_rssis(self.rssis) if np.max(self.rssis)<0 else self.rssis))
    
    def get_avg_feature(self, feature_mode='R'):
        if feature_mode == 'R':
            return normalize_rssis(self.rssis_avg) if np.max(self.rssis_avg)<0 else self.rssis_avg

    def get_label(self, label_mode=None):
        if label_mode and hasattr(self, label_mode):
            labels = getattr(self, label_mode)
            if type(labels) == list:
                labels = np.array(labels)
            if labels.ndim ==1:
                labels = np.expand_dims(labels, axis=1)
            return labels
        else:
            return self.cdns
    
    def get_dev(self, dev_dict, embedding=False):
        if not embedding:
            one_hot = np.zeros((1, len(dev_dict)))
            one_hot[0, dev_dict[self.dev]] = 1.0
            return np.repeat(one_hot, len(self), axis=0)
        else:
            return np.zeros(len(self), dtype=np.int)+dev_dict[self.dev]
    
    def avg_rssis(self, is_new=False):
        if is_new:
            return DB(rssis=np_avg(self.rssis, self.RecordsNums), cdns = np_avg(self.cdns, self.RecordsNums), bssids=self.bssids)
        else:
            self.rssis = np_avg(self.rssis, self.RecordsNums)
            self.cdns  = np_avg(self.cdns, self.RecordsNums)
    
    def repeat_avg(self):
        if not hasattr(self, 'rssis_avg'):
            self.rssis_avg = np_avg(self.rssis, self.RecordsNums)
        self.rssis_avg = np_repeat(self.rssis_avg, self.RecordsNums)

    def set_mask(self, mask):
        self_len = len(self)
        for k,v in zip(self.__dict__.keys(), self.__dict__.values()):
            if hasattr(v,'__len__'):
                if type(mask[0])==bool or (mask.dtype==np.bool_ if hasattr(mask, 'dtype') else False):
                    if len(v)==len(mask):
                        if type(v)==list and type(v[0])==str:
                            self.__dict__[k]=list_mask(v, mask)
                            continue
                        self.__dict__[k]=v[mask]
                elif len(v)==self_len:
                    self.__dict__[k]=v[mask]
    
    def set_bssids(self, bssids):
        if len(bssids):
            self.rssis = set_bssids(self.rssis, self.bssids, bssids)
            self.bssids = bssids
    
    def is_cdns_equal(self, cdns):
        if self.cdns.shape[0]!=cdns.shape[0]:
            return False
        return not np.any(np.abs(self.cdns-cdns)>0.01)
    
    def devs(self):
        return [self.dev]

    def shuffle(self, seed=None):
        if seed:
            np.random.seed(seed)
        p = np.random.permutation(len(self))
        for k,v in zip(self.__dict__.keys(), self.__dict__.values()):
            if hasattr(v,'__len__'):
                if len(v)==len(self):
                    if (type(v)==list)&(type(v[0])==str):
                        self.__dict__[k]=list_mask(v, p)
                    else:
                        self.__dict__[k]=v[p]
        return p

    def normalize_rssis(self):
        if np.mean(self.rssis)<0:
            self.rssis = normalize_rssis(self.rssis)

    def unnormalize_rssis(self):
        self.rssis = unnormalize_rssis(self.rssis)
    
    def print(self):
        if hasattr(self, 'rssis'):
            print('len: %d, min_rssis: %d'%(len(self),np.min(self.rssis)))
        print([(k, len(v) if type(v)==list else (v.shape if type(v)==np.ndarray else v)) for k,v in zip(self.__dict__.keys(), self.__dict__.values())])
    
    def set_db_no(self, i=0):
        if hasattr(self, 'db'):
            self.db.set_db_no(i)
        else:
            self.db_no = np.zeros((len(self),1))+i if i else np.zeros((len(self),1))
    
    def new(self):
        return DB(cdns=self.cdns, rssis=self.rssis, bssids=self.bssids, RecordsNums=self.RecordsNums, 
            rp_no=self.rp_no if hasattr(self, 'rp_no') else [], 
            db_no=self.db_no if hasattr(self, 'db_no') else [])

# WiFi database setting
class Setting(object):
    def __init__(self, cdns, wfiles): 
        self.cdns = cdns
        self.wfiles = wfiles
        self.print()
    
    def print(self):
        print(self.cdns.shape)
        print(len(self.wfiles))
    
    def set_mask(self, mask):
        self.cdns = self.cdns[mask]
        self.wfiles = self.wfiles[mask]
    
class SubDB(DB):
    def __init__(self, db, mask):
        self.db = db
        self.mask = mask
    
    @property
    def dev(self):
        return self.db.dev
    
    @property
    def rssis(self):
        return self.db.rssis[self.mask]
    
    @property
    def rssis_avg(self):
        return self.db.rssis_avg[self.mask]

    @property
    def cdns(self):
        return self.db.cdns[self.mask]
    
    @property
    def bssids(self):
        return self.db.bssids

    @property
    def RecordsNums(self):
        start_ind = np.cumsum(self.db.RecordsNums) - self.db.RecordsNums
        return self.db.RecordsNums[self.mask[start_ind]]

    @property
    def rp_no(self):
        return self.db.rp_no[self.mask]
    
    @property
    def db_no(self):
        return self.db.db_no[self.mask]
    
    def set_bssids(self, bssids):
        self.db.set_bssids(bssids)
    
    def shuffle(self):
        p = self.db.shuffle()
        self.mask = self.mask[p] # only support logical mask
    
    def devs(self):
        return self.db.devs()
    
    def __len__(self):
        return np.sum(self.mask) if len(self.db)==len(self.mask) and max(self.mask)<=1.0 else len(self.mask)
    
class DBs(DB):
    def __init__(self, dbs, set_bssids=False, bssids=None):
        self.dbs = dbs
        if set_bssids:
            if bssids is not None:
                self.bssids = bssids
            else:
                self.bssids = list(set(list_con([db.bssids for db in self.dbs])))
            for db in self.dbs:
                 db.set_bssids(self.bssids)
        else:
            bssids = self.dbs[0].bssids
        for db,i in zip(self.dbs, range(len(self.dbs))):
            db.set_db_no(i)
    
    @property
    def rssis(self):
        return np.vstack(tuple([db.rssis for db in self.dbs]))
    
    @property
    def cdns(self):
        return np.vstack(tuple([db.cdns for db in self.dbs]))
    
    @property
    def rp_no(self):
        base = np.insert(np.cumsum([np.max(db.rp_no)+1 for db in self.dbs])[:-1],0,0)
        return np.hstack(tuple([db.rp_no+_t for db,_t in zip(self.dbs, base)]))
    
    @property
    def db_no(self):
        return np.vstack(tuple([db.db_no for db in self.dbs]))
    
    @property
    def RecordsNums(self):
        return np.hstack(tuple([db.RecordsNums for db in self.dbs]))
    
    def devs(self):
        return [db.dev for db in self.dbs]

    def __len__(self):
        return sum([len(db) for db in self.dbs])
    
    def shuffle(self, seed=None):
        for db in self.dbs:
            db.shuffle(seed)
    
    def normalize_rssis(self):
        for db in self.dbs:
            db.normalize_rssis()
