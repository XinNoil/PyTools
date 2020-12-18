import os,copy
import numpy as np
from datahub.wifi import get_bssids, process_rssis, WiFiData, set_bssids, get_ssids, normalize_rssis
from mtools import list_mask,list_con,load_h5,save_h5,load_json,save_json,csvread,np_avg,np_intersect,np_repeat

class DB(object):
    def __init__(self, data_path, data_name, dbtype='db', cdns=[], wfiles=[], avg=False, bssids=[], save_h5_file=False, event=False, is_load_h5=True, start_time=-4):
        self.data_path = data_path
        self.data_name = data_name
        self.dbtype = dbtype
        self.start_time = start_time
        print(data_name)
        if os.path.exists(self.save_name(avg)) and is_load_h5:
            source = 'h5'
            print('%s is using %s file'%(dbtype, source))
            self.__dict__ = load_h5(self.save_name(avg))
            self.data_path = data_path
            if bssids:
                self.set_bssids(bssids)
        elif os.path.exists(self.csv_name()):
            source = 'csv'
            print('%s is using %s file'%(dbtype, source))
            self.load_csv(avg)
        elif os.path.exists(self.zip_name()):
            source = 'zip'
            print('%s is using %s file'%(dbtype, source))
            self.cdns = np.array(cdns)
            self.wfiles = wfiles
            self.process_data(bssids, avg, save_h5_file, event)
        if hasattr(self, 'rssis'):
            print('len: %d'%len(self))

    def __len__(self):
        return self.rssis.shape[0]

    def load_csv(self, avg):
        desc = load_json(os.path.join(self.data_path, '%s_data_description.json'%self.data_name))
        self.bssids = load_json(os.path.join(self.data_path, '%s_bssids.json'%self.data_name))
        data = csvread(self.csv_name())
        self.cdns  = np.round(data[:, 0: desc['cdn_dim']]+np.array(desc['cdn_min']), 4)
        self.mags  = data[:, desc['cdn_dim']:desc['cdn_dim']+desc['mag_dim']]
        self.rssis = data[:, desc['cdn_dim']+desc['mag_dim']:desc['cdn_dim']+desc['mag_dim']+desc['rssi_dim']]
        if avg:
            self.RecordsNums = np.array(desc['RecordsNums'])
            self.avg()
        else:
            self.RecordsNums = np.ones(self.rssis.shape[0])
        save_h5(self.save_name(avg), self)
    
    def avg(self):
        if np.sum(self.RecordsNums) != self.cdns.shape[0]:
            raise Exception('Sum of RecordsNums %d is not equal with data length %d' % (np.sum(self.RecordsNums), self.cdns.shape[0]))
        self.cdns  = np.round(np_avg(self.cdns, self.RecordsNums), 4)
        self.mags  = np_avg(self.mags, self.RecordsNums)
        self.rssis = np_avg(self.rssis, self.RecordsNums)
        self.RecordsNums = np.ones(len(self))

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
        return os.path.join(self.pre_path(), '%s_%s_bssids_list.json'%(self.data_name, self.dbtype))
    
    def max_rssis_list_name(self):
        return os.path.join(self.pre_path(), '%s_%s_max_rssis_list.json'%(self.data_name, self.dbtype))
    
    def bssids_name(self):
        return os.path.join(self.pre_path(), '%s_%s_bssids.json'%(self.data_name, self.dbtype))
        
    def process_data(self, bssids, avg, save_h5_file, event):
        if not os.path.exists(self.pre_path()):
            os.mkdir(self.pre_path())
        self.process_wifi(bssids, avg, save_h5_file, event)
        if (not avg) and len(self.cdns):
            self.cdns = np_repeat(self.cdns, self.RecordsNums)
        save_h5(self.save_name(avg), self)
    
    def process_bssids_max_rssis(self):
        bssids_results = [get_bssids(filename, self.zip_name(), [], []) for filename in self.wfiles]
        bssids_list = [bssids for bssids, max_rssis in bssids_results]
        max_rssis_list = [max_rssis for bssids, max_rssis in bssids_results]
        save_json(self.bssids_list_name(), bssids_list)
        save_json(self.max_rssis_list_name(), max_rssis_list)
        return bssids_list, max_rssis_list
    
    def process_wifi(self, bssids, avg, save_h5_file, event):
        if os.path.exists(self.bssids_list_name()) & os.path.exists(self.max_rssis_list_name()):
            bssids_list = load_json(self.bssids_list_name())
            max_rssis_list = load_json(self.max_rssis_list_name())
        else:
            bssids_list, max_rssis_list = self.process_bssids_max_rssis()
        if bssids:
            self.bssids = bssids
        elif os.path.exists(self.bssids_name()):
            self.bssids = load_json(self.bssids_name())
        else:
            bssids = [bssid for bssids in bssids_list for bssid in bssids]
            max_rssis = [max_rssi for max_rssis in max_rssis_list for max_rssi in max_rssis]
            bssids = list(set(list_mask(bssids, [max_rssi>=-80 for max_rssi in max_rssis])))
            bssids.sort()
            print('bssids size: %d'%len(bssids))
            self.bssids = bssids
            save_json(self.bssids_name(), self.bssids)
        self.process_rssis_all(avg, bssids_list, save_h5_file, event)
    
    def scan_ssids(self):
        ssids_results = [get_ssids(filename, self.zip_name(), [], []) for filename in self.wfiles]
        bssids_list = [bssids for bssids, ssids in ssids_results]
        ssids_list  = [ssids  for bssids, ssids in ssids_results]
        bssids = list_con(bssids_list)
        ssids  = list_con(ssids_list)
        bssids_u = list(set(bssids))
        ssids_u  = [ssids[bssids.index(bssid)] for bssid in bssids_u]
        return bssids_u, ssids_u

    def process_rssis_all(self, avg, bssids_list, save_h5_file, event):
        self.rssis = []
        if not avg:
            self.RecordsNums = []
        for filename,bssids in zip(self.wfiles, bssids_list):
            if os.path.exists(self.prefilename(filename)):
                wiFiData = self.load_prefile(filename)
            else:
                wiFiData = process_rssis(filename, bssids, self.zip_name(), event)
                if save_h5_file:
                    save_h5(self.prefilename(filename), wiFiData)
            rssis = set_bssids(wiFiData.rssis, bssids, self.bssids)
            if not event:
                rssis = np.array([np.mean(x, 0) for x in np.array_split(rssis, np.floor(len(rssis)/5.0), axis=0)])
                rssis = rssis[self.start_time:]
            if avg:
                self.rssis.append(np.mean(rssis, axis=0))
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
        for k,v in zip(self.__dict__.keys(), self.__dict__.values()):
            if hasattr(v,'__len__'):
                if (type(mask[0])==bool or type(mask[0])==np.bool_) & (len(v)==len(self)):
                    if (type(v)==list)&(type(v[0])==str):
                        self.__dict__[k]=list_mask(v, mask)
                    elif type(v)==np.ndarray:
                        self.__dict__[k]=v[mask]
                elif len(v)==len(self):
                    self.__dict__[k]=v[mask]
    
    def set_bssids(self, bssids):
        self.rssis = set_bssids(self.rssis, self.bssids, bssids)
        self.bssids = bssids
    
    def is_cdns_equal(self, cdns):
        if self.cdns.shape[0]!=cdns.shape[0]:
            return False
        return not np.any(np.abs(self.cdns-cdns)>0.01)
    
    def get_feature(self, feature_mode='R'):
        if feature_mode == 'R':
            return normalize_rssis(self.rssis) if np.max(self.rssis)<0 else self.rssis
        elif feature_mode is 'MM':
            return self.mags
        elif feature_mode is 'RMM':
            return np.hstack((self.mags, normalize_rssis(self.rssis) if np.max(self.rssis)<0 else self.rssis))
    
    def get_label(self, label_mode):
        return self.cdns

    def shuffle(self):
        p = np.random.permutation(len(self))
        for k,v in zip(self.__dict__.keys(), self.__dict__.values()):
            if hasattr(v,'__len__'):
                if len(v)==len(self):
                    if (type(v)==list)&(type(v[0])==str):
                        self.__dict__[k]=list_mask(v, p)
                    else:
                        self.__dict__[k]=v[p]

class SubDB(object):
    def __init__(self, db, mask):
        self.db = db
        self.mask = mask
    
    def __len__(self):
        return np.sum(self.mask) if len(self.db)==len(self.mask) & max(self.mask)>1 else len(self.mask)
    
    def get_feature(self, feature_mode='R'):
        return self.db.get_feature(feature_mode)[self.mask]
    
    def get_label(self, label_mode):
        return self.db.get_label(label_mode)[self.mask]

class DBs(object):
    def __init__(self, dbs):
        self.dbs = dbs
    
    def __len__(self):
        return sum([len(db) for db in self.dbs])
    
    def shuffle(self):
        for db in self.dbs:
            db.shuffle()
    
    def get_feature(self, feature_mode='R'):
        return np.vstack(tuple([db.get_feature(feature_mode) for db in self.dbs]))
    
    def get_label(self, label_mode):
        return np.vstack(tuple([db.get_label(label_mode) for db in self.dbs]))

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

class Setting(object):
    def __init__(self, cdns, wfiles): 
        self.cdns = cdns
        self.wfiles = wfiles
