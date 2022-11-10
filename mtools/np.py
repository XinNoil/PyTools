# numpy functions
import numpy as np

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

def np_avg_std(arr, r):
    arrs = np.split(arr, r.cumsum()[:-1])
    return np.array([np.mean(x, 0) for x in arrs]), np.array([np.std(x, 0) for x in arrs])

def np_mean_nonzero(data, zero_value=-100, axis=0):
    data_sum = np.sum(data, axis=axis)
    data_zero_num = np.sum(data==zero_value, axis=axis)
    data_nonzero_num = data.shape[axis] - data_zero_num
    data_sum -= zero_value*data_zero_num
    data_sum[data_nonzero_num==0] = zero_value
    data_nonzero_num[data_nonzero_num==0] = 1
    return data_sum/data_nonzero_num

def np_union_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def np_repeat(data, nums):
    return np.vstack(tuple([np.array(np.tile(row, (s, 1)), dtype=data.dtype) for row, s in zip(data, nums)]))

def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2, axis=-1))

def np_normalize(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))