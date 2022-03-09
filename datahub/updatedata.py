import os
import numpy as np
from datahub.db import DB
from mtools import list_mask

def get_months_dates(data_path):
    months = os.listdir(data_path)
    months = list_mask(months, [month.startswith('2') for month in months])
    months.sort()
    filenames = [os.listdir(os.path.join(data_path, month)) for month in months]
    data_names = [[filename[:6] for filename in filenames_] for filenames_ in filenames]
    dates = [list(set(data_names_)) for data_names_ in data_names]
    dates = [list_mask(dates_, [date.startswith('2') for date in dates_]) for dates_ in dates]
    for dates_ in dates:
        dates_.sort()
    return months, dates

def gpath(data_path, date):
    return os.path.join(data_path, date[:4])

data_path = os.environ['LONG_UPDATE'] if 'LONG_UPDATE' in os.environ else None
data_path_h5 = os.path.join(os.environ['DEEPPRINT_DATA_PATH'], 'h5')

months, dates = get_months_dates(data_path)

def get_DB(data_ver, date, dbtypes, is_print=False, device_i=1):
    if data_ver == 'origin':
        return DB(gpath(data_path, date), '%s-D%d'%(date, device_i), dbtypes[0], avg=True, is_print=is_print)
    elif data_ver == 'h5':
        return DB(data_path_h5, '%s-D%d'%(date, device_i), dbtypes[1], is_print=is_print)
    else:
        raise Exception('Unexpected data_ver')

def get_h5_DB(dataname, dbtype, is_print=False):
    db = DB(data_path_h5, dataname, dbtype, is_print=is_print)
    db.normalize_rssis()
    return db

def update_buffer(new_bssids, buffer, n=2):
    new_valid_bssids = []
    disappear_bssids = list(set(buffer.keys()).difference(new_bssids))
    for bssid in disappear_bssids:
        buffer.pop(bssid)
    for bssid in new_bssids:
        if bssid in buffer:
            buffer[bssid]+=1
            if buffer[bssid]>=n:
                buffer.pop(bssid)
                new_valid_bssids.append(bssid)
        else:
            buffer[bssid] = 1
    return new_valid_bssids

def get_max_rssis(_max_rssis, bssids, all_bssids):
    bssid_index = np.array([all_bssids.index(bssid) for bssid in bssids])
    return _max_rssis[bssid_index]

def max_rssi_filter(bssids, _max_rssis, threshold=-80):
    return list_mask(bssids, [max_rssi>threshold for max_rssi in _max_rssis])