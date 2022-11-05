import os, csv, json, h5py, zipfile
import scipy.io
import numpy as np
from .np import str2np, np2str

# IO: json, h5, csv, mat
def tojson(o, ensure_ascii=True):
    return json.dumps(o, default=lambda obj: obj.__dict__, sort_keys=True,ensure_ascii=ensure_ascii)

def check_dir(path, is_file=False):
    if is_file:
        sub_paths = path.split(os.path.sep)
        path = os.path.sep.join(sub_paths[:-1])
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('mkdir fail: %s'%path)
    return path

def file_dir(file):
    '''
    get path of *.py : file_dir(__file__)
    '''
    return os.path.split(os.path.abspath(file))[0]

def toobj(strjson):
    json.loads(strjson)

def load_json(filename, encoding=None):
    json_file=open(filename, 'r', encoding=encoding)
    json_strings=json_file.readlines()
    json_string=''.join(json_strings)
    json_file.close()
    return json.loads(json_string)

def save_json(filename, obj, ensure_ascii=True, encoding=None):
    str_json=tojson(obj, ensure_ascii)
    with open(filename, 'w', encoding=encoding) as f:
        f.write(str_json)
        f.close()

def save_h5(filename, obj, utf=False):
    f=h5py.File(filename,'w')
    if hasattr(obj, '__dict__'):
        __dict__ = obj.__dict__
    else:
        __dict__ = obj
    for v,k in zip(__dict__.values(), __dict__.keys()):
        if hasattr(v,'__len__'):
            if len(v)>0:
                if (type(v[0])==str):
                    if not utf:
                        v = str2np(v)
                    else:
                        v = np.array(v, h5py.string_dtype(encoding='utf-8'))
        f[k]=v
    f.close()

def load_h5(filename):
    f = h5py.File(filename,'r')
    __dict__=dict((k,v[()]) for k,v in zip(f.keys(), f.values()))
    for k,v in zip(__dict__.keys(), __dict__.values()):
        if type(v) == np.bytes_:
            __dict__[k] = str(np.char.decode(v))
        elif type(v) == bytes:
            __dict__[k] = v.decode()
        elif type(v)==np.ndarray and v.dtype.char=='S':
            __dict__[k] = np2str(v)
    return __dict__

def csvread(filename,delimiter=','):
    return np.loadtxt(filename,delimiter=delimiter)

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

def load_mat(filename):
    return scipy.io.loadmat(filename)

def get_zip_filenames(zip_src):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        return fz.namelist()
    else:
        return False

def write_file(file_name, str_list, encoding=None):
    file_=open(file_name, 'w', encoding=encoding)
    file_.writelines(['%s\n'%s for s in str_list])
    file_.close()

def read_file(file_name, encoding=None):
    file_=open(file_name, 'r', encoding=encoding)
    str_list = file_.read().splitlines()
    file_.close()
    return str_list

# print
def print_mat_eq(name,val):
    print('%s=%s;'%(name, val))