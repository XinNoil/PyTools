import os,sys,argparse
import numpy as np
from itertools import compress, chain
from .io import load_json
from collections import ChainMap

# str
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def join_path(a, *p):
    return os.path.join(a, *p)

def fixed_len_str(var, fixed_len=18):
    string = str(var)
    if len(string)<fixed_len:
        return string
    else:
        return '%s..'%string[:fixed_len]

fls = fixed_len_str

def str_insert(a, b, c):
    str_list = list(a)    # 字符串转list
    str_list.insert(b, c)  # 在指定位置插入字符串
    str_out = ''.join(str_list)    # 空字符连接
    return  str_out

# lists
def list_find(l, a=None):
    if a is None:
        return [i for i, x in enumerate(l) if x]
    else:
        return [i for i, x in enumerate(l) if x==a]

def list_mask(l, m):
    return list(compress(l, m))

def list_con(l):
    return list(chain(*l))

def list_ind(l, ind):
    return [l[i] for i in ind]

def list_avg(l):
    return np.mean(np.array(l),0).tolist()

def list_remove(l, v):
    return list(filter(lambda a: a != v, l))

def unique(l):
    return list(set(l))

def intersection(a, b):
    c = list(set(a).intersection(b))
    i_a = [a.index(x) for x in c]
    i_b = [b.index(x) for x in c]
    return c, i_a, i_b

def for_print(l):
    for x in l:
        print(x)

def list_not(l):
    return list(map(lambda x:not x, l))

def list_and(l1,l2):
    return list(map(lambda x,y: x and y, l1,l2))

def lb2li(l): # bool list to int list
    return list(map(lambda x:int(x), l))

def union(l, sort=True):
    nl = list(set(list_con(l)))
    if sort:
        nl.sort()
    return nl

# dict
def merge_dict(d1, d2):
    d3 = {}
    d3.update(d1)
    d3.update(d2)
    return d3

def dict_con(a):
    return dict(ChainMap(*a))

# tuple
def tuple_ind(l, ind):
    return tuple(l[i] for i in ind)

def not_none(a, b):
    return b if (a is None) else a

# class
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    def __str__(self) -> str:
        return str(self.__dict__)

get_Non = not_none