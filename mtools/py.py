from itertools import compress, chain

# lists
def list_find(l):
    return [i for i, x in enumerate(l) if x]

def list_mask(l, m):
    return list(compress(l, m))

def list_con(l):
    return list(chain(*l))