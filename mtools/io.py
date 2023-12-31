import os, csv, json, h5py, zipfile
import scipy.io
import numpy as np
from .np import str2np, np2str
import os.path as osp
import re

def glob(walk_path, pattern, _type='file'):
    paths = os.walk(walk_path)
    file_paths = []
    for path, dir_list, file_list in paths:
        if _type == 'file':
            for file_name in file_list:
                if re.match(pattern, file_name):
                    file_paths.append(osp.join(path, file_name))
        elif _type == 'dir':
            for file_name in dir_list:
                if re.match(pattern, file_name):
                    file_paths.append(osp.join(path, file_name))
    return file_paths

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# IO: json, h5, csv, mat
def tojson(o, ensure_ascii=True):
    return json.dumps(o, default=lambda obj: obj.__dict__, sort_keys=True,ensure_ascii=ensure_ascii, cls=NumpyEncoder)

def _print(*args, is_print=True):
    if is_print:
        print(*args)

def check_dir(path, is_file=False, is_print=True):
    if is_file:
        sub_paths = path.split(os.path.sep)
        path = os.path.sep.join(sub_paths[:-1])
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            _print('mkdir: %s'%path, is_print=is_print)
        except:
            _print('mkdir fail: %s'%path, is_print=is_print)
    else:
        _print('mkdir exist: %s'%path, is_print=is_print)
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

def save_h5(filename, obj, utf=False, mode='w'):
    f=h5py.File(filename,mode)
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
    f.close()
    return __dict__

def csvread(filename,delimiter=',',**kwargs):
    return np.loadtxt(filename,delimiter=delimiter, **kwargs)

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

def print_each(str_list):
    for _ in str_list:
        print(_)

def print_cfg_params(obj, cfg, log):
    for key in cfg.keys():
        if key in obj.__dict__:
            log.info(f"self.{key}={obj.__dict__[key]}")