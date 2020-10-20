import csv,zipfile
import numpy as np
import itertools
from more_itertools import chunked
from io import StringIO
from mtools import csvwrite

def get_text_stream(filename, zip_archive):
    filename = filename.replace('\\','/')
    if filename not in zip_archive.namelist():
        filename = '/%s'%filename
    csvbytes = zip_archive.read(filename).splitlines()
    csvstr = [str(line)[2:-1] for line in csvbytes]
    csvstr = '\n'.join(csvstr)
    text_stream = StringIO(csvstr)
    return text_stream

def update_reader_bssids(reader, threshold):
    bssids=[]
    max_rssis=[]
    for row in reader:
        if (row[1] not in bssids) & (int(row[2])>=threshold):
            bssids.append(row[1])
            max_rssis.append(int(row[2]))
        elif row[1] in bssids:
            if int(row[2])>max_rssis[bssids.index(row[1])]:
                max_rssis[bssids.index(row[1])]=int(row[2])
    bssids.sort()
    return bssids

def get_bssids(filename, zip_name=None, threshold=-80):
    if zip_name:
        with zipfile.ZipFile(zip_name, 'r') as zip_archive:
            reader = csv.reader(get_text_stream(filename, zip_archive))
            bssids = update_reader_bssids(reader, threshold)
    else:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            bssids = update_reader_bssids(reader, threshold)
    return bssids

def loadWiFiData(reader, bssids):
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
    def __init__(self, bssids=None, rssis=None):
        self.bssids = bssids
        self.rssis  = rssis

def process_rssis(filename, bssids, zip_name=[]):
    if zip_name:
        with zipfile.ZipFile(zip_name, 'r') as zip_archive:
            reader=csv.reader(get_text_stream(filename, zip_archive))
            return loadWiFiData(reader, bssids)
    else:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            reader=csv.reader(f)
            return loadWiFiData(reader, bssids)

def set_bssids(rssis, bssids, bssids_new):
    inter_bssids = set(bssids).intersection(set(bssids_new))
    inter_index_self = [bssids.index(bssid)     for bssid in inter_bssids]
    inter_index_new  = [bssids_new.index(bssid) for bssid in inter_bssids]
    rssis_new = np.zeros((rssis.shape[0], len(bssids_new)))
    rssis_new[:, inter_index_new] = rssis[:, inter_index_self]
    return rssis_new