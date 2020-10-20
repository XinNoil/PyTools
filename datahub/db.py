import os,copy
import numpy as np
from datahub.wifi import get_bssids, process_rssis, WiFiData, set_bssids
from mtools import list_mask,load_h5,save_h5,load_json,save_json,csvread,np_avg,np_intersect

class DB(object):
    def __init__(self, data_path, data_name, dbtype='db', cdns=[], wfiles=[], avg=False, bssids=[]):
        self.data_path = data_path
        self.data_name = data_name
        self.dbtype = dbtype
        print(data_name)
        if os.path.exists(self.save_name(avg)):
            source = 'h5'
            print('%s is using %s file'%(dbtype, source))
            self.__dict__ = load_h5(self.save_name(avg))
        elif os.path.exists(self.csv_name()):
            source = 'csv'
            print('%s is using %s file'%(dbtype, source))
            self.load_csv(avg)
        elif os.path.exists(self.zip_name()):
            source = 'zip'
            print('%s is using %s file'%(dbtype, source))
            self.cdns = np.array(cdns)
            self.wfiles = wfiles
            self.process_data(bssids, avg)
    
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
        self.cdns  = np_avg(self.cdns, self.RecordsNums)
        self.mags  = np_avg(self.mags, self.RecordsNums)
        self.rssis = np_avg(self.rssis, self.RecordsNums)

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
    
    def bssids_name(self):
        return os.path.join(self.pre_path(), '%s_bssids.json'%(self.data_name))
        
    def process_data(self, bssids, avg):
        if not os.path.exists(self.pre_path()):
            os.mkdir(self.pre_path())
        self.process_wifi(bssids, avg)
        save_h5(self.save_name(avg), self)
    
    def process_wifi(self, bssids, avg):
        if os.path.exists(self.bssids_list_name()):
            bssids_list = load_json(self.bssids_list_name())
        else:
            bssids_results = [get_bssids(filename, self.zip_name(), [], []) for filename in self.wfiles]
            # bssids_results = [get_bssids(filename, self.zip_name()) for filename in self.wfiles]
            bssids_list = [bssids for bssids, max_rssis in bssids_results]
            max_rssis_list = [max_rssis for bssids, max_rssis in bssids_results]
            save_json(self.bssids_list_name(), bssids_list)
        if bssids:
            self.bssids = bssids
        elif os.path.exists(self.bssids_name()):
            self.bssids = load_json(self.bssids_name())
        else:
            bssids = [bssid for bssids in bssids_list for bssid in bssids]
            max_rssis = [max_rssi for max_rssis in max_rssis_list for max_rssi in max_rssis]
            bssids = list(set(list_mask(bssids, [max_rssi>=-80 for max_rssi in max_rssis])))
            bssids.sort()
            print('bssids size: '%len(bssids))
            self.bssids = bssids
            save_json(self.bssids_name(), self.bssids)
        self.process_rssis_all(avg, bssids_list)

    def process_rssis_all(self, avg, bssids_list):
        self.rssis = []
        if not avg:
            self.RecordsNums = []
        for filename,bssids in zip(self.wfiles, bssids_list):
            if os.path.exists(self.prefilename(filename)):
                wiFiData = self.load_prefile(filename)
            else:
                wiFiData = process_rssis(filename, bssids, self.zip_name())
                # save_h5(self.prefilename(filename), wiFiData)
            rssis = set_bssids(wiFiData.rssis, bssids, self.bssids)
            rssis = np.array([np.mean(x, 0) for x in np.array_split(rssis, np.floor(len(rssis)/5.0), axis=0)])
            if avg:
                self.rssis.append(np.mean(rssis, axis=0))
            else:
                self.rssis.append(rssis)
                self.RecordsNums.append(len(rssis))
        self.rssis = np.vstack(self.rssis)
        if avg:
            self.RecordsNums = np.ones(self.rssis.shape[0])
        else:
            self.RecordsNums = np.array(self.RecordsNums)
    
    def load_prefile(self, filename):
        wiFiData = WiFiData()
        wiFiData.__dict__ = load_h5(self.prefilename(filename))
        return wiFiData
    
    def set_mask(self, mask):
        for k,v in zip(self.__dict__.keys(), self.__dict__.values()):
            if len(v)==len(mask):
                if (type(v)==list)&(type(v[0])==str)&(len(v)==np.sum(self.RecordsNums)):
                    self.__dict__[k]=list_mask(v, mask)
                elif type(v)==np.ndarray:
                    self.__dict__[k]=v[mask]
    
    def set_bssids(self, bssids):
        self.rssis = set_bssids(self.rssis, self.bssids, bssids)
        self.bssids = bssids
    
    def is_cdns_equal(self, cdns):
        if self.cdns.shape[0]!=cdns.shape[0]:
            return False
        return not np.any(np.abs(self.cdns-cdns)>0.01)

def get_filenames(folderlist, filenumlist, prefix):
    filenames = []
    for di,num in zip(folderlist, filenumlist):
        for fi in range(num):
            filenames.append(os.path.join('%d'%di, '%s%d.txt'% (prefix,fi)))
    return filenames

def reduce_dbs(db1, db2):
    if not db1.is_cdns_equal(db2.cdns):
        print('reduce_dbs(%s_%s, %s_%s): reduce' %(db1.data_name, db2.dbtype, db2.data_name, db2.dbtype))
        ia, ib = np_intersect(db1.cdns, db2.cdns)
        db1.set_mask(ia)
        db2.set_mask(ib)
    else:
        print('reduce_dbs(%s_%s, %s_%s): equal' %(db1.data_name, db2.dbtype, db2.data_name, db2.dbtype))