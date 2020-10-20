import os,copy
import numpy as np
from itertools import compress
from datahub.wifi import get_bssids, process_rssis, WiFiData, set_bssids
from mtools import load_h5,save_h5,load_json,save_json,csvread,np_avg,np_intersect

class DB(object):
    def __init__(self, data_path, data_name, dbtype='db', cdns=[], wfiles=[], avg=False):
        self.data_path = data_path
        self.data_name = data_name
        self.dbtype = dbtype
        print(data_name)
        if os.path.exists(self.save_name()):
            source = 'h5'
            print('%s is using %s file'%(dbtype, source))
            self.__dict__ = load_h5(self.save_name())
        elif os.path.exists(self.csv_name()):
            source = 'csv'
            print('%s is using %s file'%(dbtype, source))
            self.load_csv(avg)
        elif os.path.exists(self.zip_name()):
            source = 'zip'
            print('%s is using %s file'%(dbtype, source))
            self.cdns = cdns
            self.wfiles = wfiles
            self.process_data()        
    
    def load_csv(self, avg):
        desc = load_json(os.path.join(self.data_path, '%s_data_description.json'%self.data_name))
        r = np.array(desc['RecordsNums'])
        self.bssids = load_json(os.path.join(self.data_path, '%s_bssids.json'%self.data_name))
        data = csvread(self.csv_name())
        if avg:
            if np.sum(r) != data.shape[0]:
                raise Exception('Sum of RecordsNums %d is not equal with data length %d' % (np.sum(r), data.shape[0]))
            self.cdns  = np.round(np_avg(data[:, 0: desc['cdn_dim']]+np.array(desc['cdn_min']), r), 3)
            self.mags  = np_avg(data[:, desc['cdn_dim']:desc['cdn_dim']+desc['mag_dim']], r)
            self.rssis = np_avg(data[:, desc['cdn_dim']+desc['mag_dim']:desc['cdn_dim']+desc['mag_dim']+desc['rssi_dim']], r)
        else:
            self.cdns  = np.round(data[:, 0: desc['cdn_dim']]+np.array(desc['cdn_min']), 3)
            self.mags  = data[:, desc['cdn_dim']:desc['cdn_dim']+desc['mag_dim']]
            self.rssis = data[:, desc['cdn_dim']+desc['mag_dim']:desc['cdn_dim']+desc['mag_dim']+desc['rssi_dim']]
        save_h5(self.save_name(), self)
    
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
    
    def save_name(self):
        return self.filename(postfix=self.dbtype, ext='h5')
    
    def csv_name(self):
        return self.filename(postfix=self.dbtype, ext='csv')
    
    def pre_path(self):
        return self.filename(postfix='pre')
    
    def bssids_list_name(self):
        return self.filename(postfix='bssids_list', ext='json')
    
    def bssids_name(self):
        return self.filename(postfix='bssids', ext='json')
        
    def process_data(self, db=None):
        if not os.path.exists(self.pre_path()):
            os.mkdir(self.pre_path())
        self.process_wifi(db)
        save_h5(self.save_name(), self)
    
    def process_wifi(self, db=None):
        if os.path.exists(self.bssids_list_name()):
            bssids_list = load_json(self.bssids_list_name())
        else:
            bssids_list = [get_bssids(filename, self.zip_name()) for filename in self.wfiles]
            save_json(self.bssids_list_name(), bssids_list)
        if db:
            self.bssids = db.bssids
        elif os.path.exists(self.bssids_name()):
            self.bssids = load_json(self.bssids_name())
        else:
            bssids = [bssid for bssids in bssids_list for bssid in bssids]
            bssids.sort()
            self.bssids = bssids
            save_json(self.bssids_name(), self.bssids)
        self.process_rssis_all(bssids_list)

    def process_rssis_all(self, bssids_list):
        self.rssis = []
        for filename,bssids in zip(self.wfiles, bssids_list):
            if os.path.exists(self.prefilename(filename)):
                wiFiData = self.load_prefile(filename)
            else:
                wiFiData = process_rssis(filename, bssids, self.zip_name())
                save_h5(self.prefilename(filename), wiFiData)
            self.rssis.append(np.mean(set_bssids(wiFiData.rssis, bssids, self.bssids), axis=0))
        self.rssis = np.array(self.rssis)
    
    def load_prefile(self, filename):
        wiFiData = WiFiData()
        wiFiData.__dict__ = load_h5(self.prefilename(filename))
        return wiFiData
    
    def set_mask(self, mask):
        for k,v in zip(self.__dict__.keys(), self.__dict__.values()):
            if len(v)==len(mask):
                if (type(v)==list)&(type(v[0])==str):
                    # print('set mask to %s' % k)
                    self.__dict__[k]=list(compress(v, mask))
                elif type(v)==np.ndarray:
                    # print('set mask to %s' % k)
                    self.__dict__[k]=v[mask, :]
    
    def set_bssids(self, bssids):
        inter_bssids = set(self.bssids).intersection(set(bssids))
        inter_index_self = [self.bssids.index(bssid) for bssid in inter_bssids]
        inter_index_new  = [bssids.index(bssid)     for bssid in inter_bssids]
        rssis = np.zeros((self.rssis.shape[0], len(bssids)))
        rssis[:, inter_index_new] = self.rssis[:, inter_index_self]
        self.rssis = rssis
        self.bssids = bssids
    
    def is_cdns_equal(self, cdns):
        if self.cdns.shape[0]!=cdns.shape[0]:
            return False
        return not np.any(self.cdns-cdns)

def get_filenames(folderlist, filenumlist, prefix):
    filenames = []
    for di,num in zip(folderlist, filenumlist):
        for fi in range(num):
            filenames.append(os.path.join('%d'%di, '%s%d.txt'% (prefix,fi)))
    return filenames

def reduce_dbs(db1, db2):
    if not db1.is_cdns_equal(db2.cdns):
        print('reduce %s_%s, %s_%s' %(db1.data_name, db2.dbtype, db2.data_name, db2.dbtype))
        ia, ib = np_intersect(db1.cdns, db2.cdns)
        db1.set_mask(ia)
        db2.set_mask(ib)