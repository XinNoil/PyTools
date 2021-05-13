import argparse
import numpy as np
from itertools import compress, chain

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

# lists
def list_find(l):
    return [i for i, x in enumerate(l) if x]

def list_mask(l, m):
    return list(compress(l, m))

def list_con(l):
    return list(chain(*l))

def list_ind(l, ind):
    return [l[i] for i in ind]

def list_avg(l):
    return np.mean(np.array(l),0).tolist()

def intersection(a, b):
    c = list(set(a).intersection(b))
    i_a = [a.index(x) for x in c]
    i_b = [b.index(x) for x in c]
    return c, i_a, i_b

def for_print(l):
    for x in l:
        print(x)

# dict
def merge_dict(d1, d2):
    d3 = {}
    d3.update(d1)
    d3.update(d2)
    return d3