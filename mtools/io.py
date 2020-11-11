import os, csv, json, h5py
import scipy.io
import numpy as np
from .np import str2np, np2str

# IO: json, h5, csv, mat
def tojson(o, ensure_ascii=True):
    return json.dumps(o, default=lambda obj: obj.__dict__, sort_keys=True,ensure_ascii=ensure_ascii)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def toobj(strjson):
    json.loads(strjson)

def load_json(filename):
    json_file=open(filename, 'r')
    json_string=json_file.readline()
    json_file.close()
    return json.loads(json_string)

def save_json(filename, obj, ensure_ascii=True):
    str_json=tojson(obj, ensure_ascii)
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
