import os,copy
import numpy as np
from datahub.wifi import get_bssids, process_rssis, WiFiData, set_bssids, get_ssids, normalize_rssis, unnormalize_rssis
from mtools import list_mask,list_con,load_h5,save_h5,load_json,save_json,csvread,np_avg,np_avg_std,np_intersect,np_repeat,np_mean_nonzero

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

class DB(object):
    # bssids, cdns, rssis, mags, RecordsNums
    def __init__(self, data_path='', data_name='', dbtype='db', cdns=[], wfiles=[], avg=False, bssids=[], save_h5_file=False, event=False, is_load_h5=True, is_save_h5=True, start_time=0, rssis=[], mags=[], RecordsNums=[], filename='', dev=0, merge_method='all'):
        if len(rssis):
            self.bssids = bssids
            self.cdns = cdns
            self.rssis = rssis
            self.mags = mags
            if len(RecordsNums):
                self.RecordsNums = RecordsNums
            else:
                self.RecordsNums = np.ones(len(self))
        else:
            self.data_path = data_path
            self.data_name = data_name
            self.dbtype = dbtype
            self.start_time = start_time
            print(data_name)
            if os.path.exists(filename) and is_load_h5:
                print('load h5 file: %s'%filename)
                self.__dict__ = load_h5(filename)
                self.set_bssids(bssids)
            elif os.path.exists(self.save_name(avg)) and is_load_h5:
                print('load h5 file: %s'%self.save_name(avg))
                self.__dict__ = load_h5(self.save_name(avg))
                self.set_bssids(bssids)
            elif os.path.exists(self.csv_name()):
                print('load csv file: %s'%self.csv_name())
                self.load_csv(avg)
                self.set_bssids(bssids)
            elif os.path.exists(self.zip_name()):
                print('load zip file: %s'%self.zip_name())
                self.cdns = np.array(cdns)
                self.wfiles = wfiles
                self.process_data(bssids, avg, save_h5_file, event, is_save_h5, merge_method)
            elif os.path.exists(os.path.join(self.data_path, self.data_name)):
                print('load txt file: %s'%os.path.join(self.data_path, self.data_name))
                self.cdns = np.array(cdns)
                self.wfiles = wfiles
                self.process_data(bssids, avg, save_h5_file, event, is_save_h5, merge_method)
        self.dev = dev

    def __len__(self):
        return self.rssis.shape[0]  

    def load_csv(self, avg):
        desc = load_json(os.path.join(self.data_path, '%s_data_description.json'%self.data_name))
        self.bssids = load_json(os.path.join(self.data_path, '%s_bssids.json'%self.data_name))
        if 'rssi_mask' in desc:
            self.bssids = list_mask(self.bssids, desc['rssi_mask'])
        data = csvread(self.csv_name())
        self.cdns  = np.round(data[:, 0: desc['cdn_dim']], 4)
        self.cdns_min = np.array(desc['cdn_min'])
        self.mags  = data[:, desc['cdn_dim']:desc['cdn_dim']+desc['mag_dim']]
        self.rssis = data[:, desc['cdn_dim']+desc['mag_dim']:desc['cdn_dim']+desc['mag_dim']+desc['rssi_dim']]
        if avg:
            self.get_RecordsNums(desc)
            self.avg()
        else:
            if 'training' in self.dbtype:
                self.get_RecordsNums(desc)
            else:
                self.RecordsNums = np.ones(self.rssis.shape[0])
        save_h5(self.save_name(avg), self)
    
    def get_RecordsNums(self, desc):
        self.RecordsNums = np.array(desc['RecordsNums'])
        if 'intv' in self.csv_name():
            mask = csvread(str.replace(str.replace(self.csv_name(), 'intv', 'mask_intv'), '.csv', '_fp.csv')).astype(bool)
            self.RecordsNums = self.RecordsNums[mask]
    
    def avg(self):
        if (not hasattr(self, 'RecordsNums')) & os.path.exists(os.path.join(self.data_path, '%s_data_description.json'%self.data_name)):
            desc = load_json(os.path.join(self.data_path, '%s_data_description.json'%self.data_name))
            self.RecordsNums = np.array(desc['RecordsNums'])
        if np.sum(self.RecordsNums) != self.cdns.shape[0]:
            raise Exception('Sum of RecordsNums %d is not equal with data length %d' % (np.sum(self.RecordsNums), self.cdns.shape[0]))
        self.RecordsNums = np.array(self.RecordsNums, dtype='i')
        self.cdns  = np.round(np_avg(self.cdns, self.RecordsNums), 4)
        self.mags  = np_avg(self.mags, self.RecordsNums)
        self.rssis,self.rssis_std = np_avg_std(self.rssis, self.RecordsNums)
        self.RecordsNums = np.ones(len(self))
    
    def avg_rssis(self):
        self.rssis = np_avg(self.rssis, self.RecordsNums)
        self.cdns  = np_avg(self.cdns, self.RecordsNums)

    def filename(self, postfix=None, ext=None):
        filename = self.data_name
        if postfix:
            filename = '%s_%s'%(filename, postfix)
        if ext:
            filename = '%s.%s'%(filename, ext)
        return os.path.join(self.data_path, filename)
    
    def prefilename(self, filename):
        return os.path.join(self.pre_path(), filename.replace(os.path.sep,'-').replace('txt','h5'))

    def zip_name(self):
        return self.filename(ext='zip')
    
    def save_name(self, avg):
        return self.filename(postfix=self.dbtype+'_avg', ext='h5') if avg else self.filename(postfix=self.dbtype, ext='h5')
    
    def csv_name(self):
        return self.filename(postfix=self.dbtype, ext='csv')
    
    def pre_path(self):
        return self.filename(postfix='pre')
    
    def bssids_list_name(self):
        return os.path.join(self.pre_path(), 'bssids_list.json')
    
    def max_rssis_list_name(self):
        return os.path.join(self.pre_path(), 'max_rssis_list.json')
    
    def bssids_name(self):
        return os.path.join(self.pre_path(), 'bssids.json')
        
    def process_data(self, bssids, avg, save_h5_file, event, is_save_h5, merge_method):
        if not os.path.exists(self.pre_path()):
            os.mkdir(self.pre_path())
        self.process_wifi(avg, save_h5_file, event, merge_method)
        if (not avg) and len(self.cdns):
            self.cdns = np_repeat(self.cdns, self.RecordsNums)
        if is_save_h5:
            save_h5(self.save_name(avg), self)
        if bssids:
            self.set_bssids(bssids)
    
    def process_bssids_max_rssis(self):
        if os.path.exists(self.zip_name()):
            bssids_results = [get_bssids(filename, self.zip_name(), [], []) for filename in self.wfiles]
        else:
            bssids_results = [get_bssids(os.path.join(self.data_path, self.data_name, filename)) for filename in self.wfiles]
        bssids_list = [bssids for bssids, max_rssis in bssids_results]
        max_rssis_list = [max_rssis for bssids, max_rssis in bssids_results]
        save_json(self.bssids_list_name(), bssids_list)
        save_json(self.max_rssis_list_name(), max_rssis_list)
        return bssids_list, max_rssis_list
    
    def process_wifi(self, avg, save_h5_file, event, merge_method):
        if os.path.exists(self.bssids_list_name()) & os.path.exists(self.max_rssis_list_name()):
            bssids_list = load_json(self.bssids_list_name())
            max_rssis_list = load_json(self.max_rssis_list_name())
        else:
            bssids_list, max_rssis_list = self.process_bssids_max_rssis()
        if os.path.exists(self.bssids_name()):
            self.bssids = load_json(self.bssids_name())
        else:
            bssids = [bssid for bssids in bssids_list for bssid in bssids]
            max_rssis = [max_rssi for max_rssis in max_rssis_list for max_rssi in max_rssis]
            bssids = list(set(list_mask(bssids, [max_rssi>=-80 for max_rssi in max_rssis])))
            bssids.sort()
            print('bssids size: %d'%len(bssids))
            self.bssids = bssids
            save_json(self.bssids_name(), self.bssids)
        self.process_rssis_all(avg, bssids_list, save_h5_file, event, merge_method)
    
    def scan_ssids(self):
        ssids_results = [get_ssids(filename, self.zip_name(), [], []) for filename in self.wfiles]
        bssids_list = [bssids for bssids, ssids in ssids_results]
        ssids_list  = [ssids  for bssids, ssids in ssids_results]
        bssids = list_con(bssids_list)
        ssids  = list_con(ssids_list)
        bssids_u = list(set(bssids))
        ssids_u  = [ssids[bssids.index(bssid)] for bssid in bssids_u]
        return bssids_u, ssids_u

    def process_rssis_all(self, avg, bssids_list, save_h5_file, event, merge_method):
        self.rssis = []
        if not avg:
            self.RecordsNums = []
        for filename,bssids in zip(self.wfiles, bssids_list):
            if os.path.exists(self.prefilename(filename)):
                wiFiData = self.load_prefile(filename)
            elif os.path.exists(self.zip_name()):
                wiFiData = process_rssis(filename, bssids, self.zip_name(), event)
            else:
                wiFiData = process_rssis(os.path.join(self.data_path, self.data_name, filename), bssids, None, event)
            if (not os.path.exists(self.prefilename(filename))) & save_h5_file:
                save_h5(self.prefilename(filename), wiFiData)
            rssis = set_bssids(wiFiData.rssis, bssids, self.bssids)
            if not event:
                if merge_method=='nonzero':
                    rssis = np.array([np_mean_nonzero(x) for x in np.array_split(rssis, np.floor(len(rssis)/5.0), axis=0)])
                elif merge_method=='all':
                    rssis = np.array([np.mean(x, 0) for x in np.array_split(rssis, np.floor(len(rssis)/5.0), axis=0)])
                else:
                    rssis = np.array(rssis, dtype='i')
                rssis = rssis[self.start_time:]
            if avg:
                if merge_method=='nonzero':
                    self.rssis.append(np_mean_nonzero(rssis))
                elif merge_method=='all':
                    self.rssis.append(np.mean(rssis, axis=0))
                else:
                    rssis = np.array(rssis, dtype='i')
            else:
                self.rssis.append(rssis)
                self.RecordsNums.append(len(rssis))
        self.rssis = np.vstack(self.rssis)
        if avg:
            self.RecordsNums = np.ones(len(self))
        else:
            self.RecordsNums = np.array(self.RecordsNums)
    
    def load_prefile(self, filename):
        wiFiData = WiFiData()
        wiFiData.__dict__ = load_h5(self.prefilename(filename))
        return wiFiData
    
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
        if bssids:
            self.rssis = set_bssids(self.rssis, self.bssids, bssids)
            self.bssids = bssids
    
    def is_cdns_equal(self, cdns):
        if self.cdns.shape[0]!=cdns.shape[0]:
            return False
        return not np.any(np.abs(self.cdns-cdns)>0.01)
    
    def get_feature(self, feature_mode='R'):
        if feature_mode == 'R':
            return normalize_rssis(self.rssis) if np.max(self.rssis)<0 else self.rssis
        elif feature_mode == 'MM':
            return self.mags
        elif feature_mode == 'RMM':
            return np.hstack((self.mags, normalize_rssis(self.rssis) if np.max(self.rssis)<0 else self.rssis))
    
    def get_label(self, label_mode=None):
        if label_mode and hasattr(self, label_mode):
            return self.__dict__[label_mode]
        else:
            return self.cdns
    
    def devs(self):
        return [self.dev]
    
    def get_dev(self, dev_dict, embedding=False):
        if not embedding:
            one_hot = np.zeros((1, len(dev_dict)))
            one_hot[0, dev_dict[self.dev]] = 1.0
            return np.repeat(one_hot, len(self), axis=0)
        else:
            return np.zeros(len(self), dtype=np.int)+dev_dict[self.dev]

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
    
    def normalize_rssis(self):
        self.rssis = normalize_rssis(self.rssis)

    def unnormalize_rssis(self):
        self.rssis = unnormalize_rssis(self.rssis)
    
    def print(self):
        if hasattr(self, 'rssis'):
            print('len: %d'%len(self))
        print([(k, len(v) if type(v)==list else (v.shape if type(v)==np.ndarray else v)) for k,v in zip(self.__dict__.keys(), self.__dict__.values())])

class SubDB(object):
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
    def cdns(self):
        return self.db.cdns[self.mask]
    
    @property
    def rp_no(self):
        return self.db.rp_no[self.mask]
    
    def devs(self):
        return self.db.devs()
    
    def __len__(self):
        return np.sum(self.mask) if len(self.db)==len(self.mask) and max(self.mask)<=1.0 else len(self.mask)
    
    def get_feature(self, feature_mode='R'):
        return self.db.get_feature(feature_mode)[self.mask]
    
    def get_label(self, label_mode=None):
        return self.db.get_label(label_mode)[self.mask]
    
    def get_dev(self, dev_dict, embedding=False):
        if not embedding:
            one_hot = np.zeros((1, len(dev_dict)))
            one_hot[0, dev_dict[self.db.dev]] = 1.0
            return np.repeat(one_hot, len(self), axis=0)
        else:
            return np.zeros(len(self), dtype=np.int)+dev_dict[self.db.dev]

class DBs(object):
    def __init__(self, dbs):
        self.dbs = dbs
    
    @property
    def rssis(self):
        return np.vstack(tuple([db.rssis for db in self.dbs]))
    
    @property
    def cdns(self):
        return np.vstack(tuple([db.cdns for db in self.dbs]))
    
    @property
    def rp_no(self):
        return np.vstack(tuple([db.rp_no for db in self.dbs]))
    
    def devs(self):
        return [db.dev for db in self.dbs]

    def __len__(self):
        return sum([len(db) for db in self.dbs])
    
    def shuffle(self, seed=None):
        for db in self.dbs:
            db.shuffle(seed)
    
    def get_feature(self, feature_mode='R'):
        return np.vstack(tuple([db.get_feature(feature_mode) for db in self.dbs]))
    
    def get_label(self, label_mode=None):
        return np.vstack(tuple([db.get_label(label_mode) for db in self.dbs]))
    
    def get_dev(self, dev_dict=None, embedding=False):
        return np.hstack(tuple([db.get_dev(dev_dict, embedding) for db in self.dbs])) if embedding else np.vstack(tuple([db.get_dev(dev_dict, embedding) for db in self.dbs]))
    
    def normalize_rssis(self):
        for db in self.dbs:
            db.normalize_rssis()

# class avgDB(object):
#     def __init__(self, dbs):
#         self.cdns = dbs[0].cdns
#         self.rssis = 

#     # def __len__(self):
#     #     return self.rssis.shape[0]

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

from scipy.spatial.distance import cdist

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