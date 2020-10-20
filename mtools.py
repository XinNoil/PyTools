import csv,json,h5py,os,time
import scipy.io
import numpy as np

# lists
def list_find(l):
    return [i for i, x in enumerate(l) if x]

def list_mask(l, m):
    return [i for i,r in zip(l,m) if r]

# numpys
def str2np(s):
    return np.array(s, dtype='S')

def np2str(n):
    return [bytes.decode(s) for s in n] if n.size>1 else n.astype(str)

def np_count(arr):
    key = np.unique(arr)
    count = {}
    for k in key:
        count[k] = arr[arr == k].size
    return count

def np_intersect(arr1, arr2):
    return np.in1d(arr1.view([('',arr1.dtype)]*arr1.shape[1]), arr2.view([('',arr2.dtype)]*arr2.shape[1])), np.in1d(arr2.view([('',arr2.dtype)]*arr2.shape[1]), arr1.view([('',arr1.dtype)]*arr1.shape[1]))

def np_avg(arr, r):
    arrs = np.split(arr, r.cumsum()[:-1])
    return np.array([np.mean(x, 0) for x in arrs])

# IO: json, h5, csv, mat
def tojson(o):
    return json.dumps(o, default=lambda obj: obj.__dict__, sort_keys=True)

def toobj(strjson):
    json.loads(strjson)

def load_json(filename):
    json_file=open(filename, 'r')
    json_string=json_file.readline()
    json_file.close()
    return json.loads(json_string)

def save_json(filename, obj):
    str_json=tojson(obj)
    with open(filename,'w') as f:
        f.write(str_json)

def save_h5(filename, obj):
    f=h5py.File(filename,'w')
    for v,k in zip(obj.__dict__.values(), obj.__dict__.keys()):
        if len(v)>0:
            if (type(v[0])==str):
                v = str2np(v)
        f[k]=v
    f.close()

def load_h5(filename):
    f = h5py.File(filename,'r')
    __dict__=dict((k,v[()]) for k,v in zip(f.keys(), f.values()))
    __dict__=dict((k,np2str(v)) if v.dtype.char=='S' else (k,v) for k,v in zip(__dict__.keys(), __dict__.values()))
    return __dict__

def csvread(filename):
    return np.loadtxt(filename,delimiter=',')

def csvwrite(filename, data, delimiter=',', fmt='%.4f'):
    np.savetxt(filename ,data, delimiter=delimiter, fmt=fmt)

def loadcsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

def save_mat(filename, **kwargs):
    for key in kwargs:
        if type(kwargs[key]) is not np.ndarray:
            kwargs[key] = np.array(kwargs[key])
    scipy.io.savemat(filename, kwargs)
