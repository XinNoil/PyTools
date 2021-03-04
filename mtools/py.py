from itertools import compress, chain

# lists
def list_find(l):
    return [i for i, x in enumerate(l) if x]

def list_mask(l, m):
    return list(compress(l, m))

def list_con(l):
    return list(chain(*l))

def list_ind(l, ind):
    return [l[i] for i in ind]

def intersection(a, b):
    c = list(set(a).intersection(b))
    i_a = [a.index(x) for x in c]
    i_b = [b.index(x) for x in c]
    return c, i_a, i_b

def for_print(l):
    for x in l:
        print(x)

# str
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

