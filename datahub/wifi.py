import csv,zipfile
import numpy as np
import itertools
from more_itertools import chunked
from io import StringIO
from mtools import csvwrite,load_h5

def normalize_rssis(rssis):
    rssis = (rssis+100)/80
    rssis[rssis>1] = 1
    rssis[rssis<0] = 0
    return rssis

def get_text_stream(filename, zip_archive):
    filename = filename.replace('\\','/')
    if filename not in zip_archive.namelist():
        filename = '/%s'%filename
    csvbytes = zip_archive.read(filename).splitlines()
    csvstr = [str.split(line.decode('utf-8'),',') for line in csvbytes]
    return csvstr

def update_reader_bssids(reader, bssids, max_rssis):
    for row in reader:
        if (row[1] not in bssids):
            bssids.append(row[1])
            max_rssis.append(int(row[2]))
        elif row[1] in bssids:
            if int(row[2])>max_rssis[bssids.index(row[1])]:
                max_rssis[bssids.index(row[1])]=int(row[2])
    return bssids, max_rssis

def get_bssids(filename, zipfilename=None, bssids=[], max_rssis=[]):
    if zipfilename:
        with zipfile.ZipFile(zipfilename, 'r') as zip_archive:
            bssids, max_rssis=update_reader_bssids(get_text_stream(filename, zip_archive), bssids, max_rssis)
    else:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            bssids, max_rssis=update_reader_bssids(csv.reader(f), bssids, max_rssis)
    return bssids, max_rssis

def update_reader_ssids(reader, bssids, ssids):
    for row in reader:
        if (row[1] not in bssids):
            bssids.append(row[1])
            if len(row)>=9:
                ssids.append(row[8])
            else:
                ssids.append('')
                import pdb; pdb.set_trace()
    return bssids, ssids

def get_ssids(filename, zipfilename=None, bssids=[], ssids=[]):
    if zipfilename:
        with zipfile.ZipFile(zipfilename, 'r') as zip_archive:
            bssids, ssids=update_reader_ssids(get_text_stream(filename, zip_archive), bssids, ssids)
    else:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            bssids, ssids=update_reader_ssids(csv.reader(f), bssids, ssids)
    return bssids, ssids

def loadWiFiData(reader, bssids, event):
    recordno = 0
    rssi = [-100]*len(bssids)
    rssis = []
    for row in reader:
        if int(row[0])>recordno:
            rssis.append(rssi)
            recordno=int(row[0])
            rssi=[-100]*len(bssids)
        if row[1] in bssids:
            rssi[bssids.index(row[1])]=int(row[2])
    rssis.append(rssi)
    return WiFiData(bssids, np.array(rssis).astype(float))

class WiFiData(object):
    def __init__(self, bssids=None, rssis=None, filename=None):
        if filename:
            self.__dict__ = load_h5(filename)
        else:
            self.bssids = bssids
            self.rssis  = rssis

def process_rssis(filename, bssids, zip_name=[], event=False):
    if zip_name:
        with zipfile.ZipFile(zip_name, 'r') as zip_archive:
            return loadWiFiData(get_text_stream(filename, zip_archive), bssids, event)
    else:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            reader=csv.reader(f)
            return loadWiFiData(reader, bssids, event)

def set_bssids(rssis, bssids, bssids_new):
    inter_bssids = set(bssids).intersection(set(bssids_new))
    inter_index_self = [bssids.index(bssid)     for bssid in inter_bssids]
    inter_index_new  = [bssids_new.index(bssid) for bssid in inter_bssids]
    rssis_new = np.zeros((rssis.shape[0], len(bssids_new)))
    if np.max(rssis)<0:
        rssis_new = rssis_new-100
    rssis_new[:, inter_index_new] = rssis[:, inter_index_self]
    return rssis_new