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

def _print(s, is_print):
    if is_print:
        print(s)

class DB(object):
    # bssids, cdns, rssis, mags, RecordsNums
    def __init__(self, data_path='', data_name='', dbtype='db', cdns=[], wfiles=[], avg=False, \
                bssids=[], save_h5_file=False, event=False, is_load_h5=True, is_save_h5=True, \
                start_time=0, rssis=[], mags=[], RecordsNums=[], filename='', dev=0, is_empty=False, \
                merge_method='all', is_print=True, rp_no=[]):
        _print('', is_print)
        if len(rssis) or is_empty:
            self.bssids = bssids
            self.cdns = cdns
            self.rssis = rssis
            self.mags = mags
            self.rp_no = rp_no
            if len(RecordsNums):
                self.RecordsNums = RecordsNums
            else:
                self.RecordsNums = np.ones(len(self))
        else:
            self.data_path = data_path
            self.data_name = data_name
            self.dbtype = dbtype
            self.start_time = start_time
            print('data name: %s'%data_name)
            if os.path.exists(filename) and is_load_h5:
                _print('load h5 file: %s'%filename, is_print)
                self.__dict__ = load_h5(filename)
                self.set_bssids(bssids)
            elif os.path.exists(self.save_name(avg)) and is_load_h5:
                _print('load h5 file: %s'%self.save_name(avg), is_print)
                self.__dict__ = load_h5(self.save_name(avg))
                self.set_bssids(bssids)
            elif os.path.exists(self.csv_name()):
                _print('load csv file: %s'%self.csv_name(), is_print)
                self.load_csv(avg)
                self.set_bssids(bssids)
            elif os.path.exists(self.zip_name()):
                _print('load zip file: %s'%self.zip_name(), is_print)
                self.cdns = np.array(cdns)
                self.wfiles = wfiles
                self.process_data(bssids, avg, save_h5_file, event, is_save_h5, merge_method)
            elif os.path.exists(os.path.join(self.data_path, self.data_name)):
                _print('load txt file: %s'%os.path.join(self.data_path, self.data_name), is_print)
                self.cdns = np.array(cdns)
                self.wfiles = wfiles
                self.process_data(bssids, avg, save_h5_file, event, is_save_h5, merge_method)
        self.dev = dev
        self.print(is_print)

    def save_h5(self, filename=None, avg=False):
        if filename is None:
            save_h5(self.save_name(avg), self)
            print('save to %s' % self.save_name(avg))
        else:
            save_h5(filename, self)
            print('save to %s' % filename)

    def __len__(self):
        if type(self.rssis) == list:
            return len(self.rssis)
        else: # numpy array
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
    
    def save_name(self, avg=False):
        return self.filename(postfix=self.dbtype+'_avg', ext='h5') if avg else self.filename(postfix=self.dbtype, ext='h5')
    
    def csv_name(self):
        return self.filename(postfix=self.dbtype, ext='csv')
    
    def pre_path(self):
        return self.filename(postfix='pre')
    
    def bssids_list_name(self):
        return os.path.join(self.pre_path(), '%s_%s_bssids_list.json'%(self.data_name, self.dbtype))
    
    def max_rssis_list_name(self):
        return os.path.join(self.pre_path(), '%s_%s_max_rssis_list.json'%(self.data_name, self.dbtype))
    
    def bssids_name(self, postfix=''):
        return os.path.join(self.pre_path(), '%s_%s_bssids%s.json'%(self.data_name, self.dbtype, postfix))
        
    def process_data(self, bssids, avg, save_h5_file, event, is_save_h5, merge_method):
        if not os.path.exists(self.pre_path()):
            os.mkdir(self.pre_path())
        self.process_wifi(bssids, avg, save_h5_file, event, merge_method)
        if (not avg) and len(self.cdns):
            self.cdns = np_repeat(self.cdns, self.RecordsNums)
        if is_save_h5:
            save_h5(self.save_name(avg), self)
    
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
    
    def process_wifi(self, _bssids, avg, save_h5_file, event, merge_method):
        if os.path.exists(self.bssids_list_name()) & os.path.exists(self.max_rssis_list_name()):
            bssids_list = load_json(self.bssids_list_name())
            max_rssis_list = load_json(self.max_rssis_list_name())
        else:
            bssids_list, max_rssis_list = self.process_bssids_max_rssis()
        if os.path.exists(self.bssids_name('unfilterd')):
            self.bssids = load_json(self.bssids_name('unfilterd'))
            self.process_rssis_all(avg, bssids_list, save_h5_file, event, merge_method)
            bssids_filterd = load_json(self.bssids_name())
        else:
            bssids = [bssid for bssids in bssids_list for bssid in bssids]
            self.bssids = bssids
            save_json(self.bssids_name('unfilterd'), self.bssids)
            self.process_rssis_all(avg, bssids_list, save_h5_file, event, merge_method)

            max_rssis = [max_rssi for max_rssis in max_rssis_list for max_rssi in max_rssis]
            bssids_filterd = list(set(list_mask(bssids, [max_rssi>=-80 for max_rssi in max_rssis])))
            bssids_filterd.sort()
            save_json(self.bssids_name(), self.bssids_filterd)
        if _bssids:
            self.set_bssids(_bssids)
        else:
            self.set_bssids(bssids_filterd)
    
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
                    rssis = np.array(rssis)
                rssis = rssis[self.start_time:]
            if avg:
                if merge_method=='nonzero':
                    self.rssis.append(np_mean_nonzero(rssis))
                elif merge_method=='all':
                    self.rssis.append(np.mean(rssis, axis=0))
                else:
                    rssis = np.array(rssis)
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
        if len(bssids):
            self.rssis = set_bssids(self.rssis, self.bssids, bssids)
            self.bssids = bssids
    
    def is_cdns_equal(self, cdns):
        if self.cdns.shape[0]!=cdns.shape[0]:
            return False
        return not np.any(np.abs(self.cdns-cdns)>0.01)
    
    def devs(self):
        return [self.dev]
    
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
            return self.__dict__[label_mode]
        else:
            return self.cdns
    
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
        return p

    def normalize_rssis(self):
        self.rssis = normalize_rssis(self.rssis)

    def unnormalize_rssis(self):
        self.rssis = unnormalize_rssis(self.rssis)
    
    def print(self, is_print=True):
        if is_print:
            if hasattr(self, 'rssis'):
                print('len: %d'%len(self))
            print([(k, len(v) if type(v)==list else (v.shape if type(v)==np.ndarray else v)) for k,v in zip(self.__dict__.keys(), self.__dict__.values())])
            print('')

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
    def bssids(self):
        return self.db.bssids

    @property
    def RecordsNums(self):
        start_ind = np.cumsum(self.db.RecordsNums) - self.db.RecordsNums
        return self.db.RecordsNums[self.mask[start_ind]]

    @property
    def rp_no(self):
        return self.db.rp_no[self.mask]
    
    def set_bssids(self, bssids):
        self.db.set_bssids(bssids)
    
    def new(self):
        return DB(cdns=self.cdns, rssis=self.rssis, bssids=self.bssids, RecordsNums=self.RecordsNums, rp_no=self.rp_no)
    
    def shuffle(self):
        p = self.db.shuffle()
        self.mask = self.mask[p] # only support logical mask
    
    def devs(self):
        return self.db.devs()
    
    def __len__(self):
        return np.sum(self.mask) if len(self.db)==len(self.mask) and max(self.mask)<=1.0 else len(self.mask)
    
    def get_feature(self, feature_mode='R'):
        return self.db.get_feature(feature_mode)[self.mask]
    
    def get_avg_feature(self, feature_mode='R'):
        return self.db.get_avg_feature(feature_mode)[self.mask]
    
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
    def RecordsNums(self):
        return np.hstack(tuple([db.RecordsNums for db in self.dbs]))
    
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
    
    def new(self):
        return DB(cdns=self.cdns, rssis=self.rssis, bssids=self.bssids, RecordsNums=self.RecordsNums, rp_no=self.rp_no)

# class avgDB(object):
#     def __init__(self, dbs):
#         self.cdns = dbs[0].cdns
#         self.rssis = 

#     # def __len__(self):
#     #     return self.rssis.shape[0]

def get_save_name(data_path, data_name, dbtype, avg=False):
    return get_filename(data_path, data_name, postfix=dbtype+'_avg', ext='h5') if avg else get_filename(data_path, data_name, postfix=dbtype, ext='h5')

def get_filename(data_path, data_name, postfix=None, ext=None):
    filename = data_name
    if postfix:
        filename = '%s_%s'%(filename, postfix)
    if ext:
        filename = '%s.%s'%(filename, ext)
    return os.path.join(data_path, filename)

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