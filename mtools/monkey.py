# -*- coding: UTF-8 -*-

import os
import pdb

import pandas as pd
pd.set_option('display.float_format',lambda x : '%.4f' % x)

import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind':'{:f}'.format})


blackboard = {}
magicHolderInstance = None
to_writes = ""

def get_cfg(cfg, name, default=None):
    if name in cfg:
        return cfg[name]
    else:
        return default

def b_set(name, value):
    global blackboard
    blackboard[name] = value
    return

def b_get(name):
    global blackboard
    return blackboard.get(name, None)

class MagicHolder():
    def __init__(self):
        self.holder_dict = {}
    
    def reset(self, name):
        self.holder_dict[name] = None
        # print(f"Reset: self.holder_dict[{name}] = None") 

    def magic_append(self, target, name):
        list_size = len(target)

        if name not in self.holder_dict:
            # print(f"name {name} not in self.holder_dict") 
            self.holder_dict[name] = None

        if self.holder_dict[name] is None:
            # print(f"self.holder_dict[{name}] is None") 
            self.holder_dict[name] = [[] for _ in range(list_size)]

        for i in range(list_size):
            self.holder_dict[name][i].append(target[i])

    def magic_get(self, name, func=None, keep=False):
        if name not in self.holder_dict:
            print(f"{name} Not Registered, Returning None")
            return None
        
        if self.holder_dict[name] is None:
            print(f"Nothing Appended To {name}, Returning None")
            return None
        
        if func is not None:
            ret = list(map(func, self.holder_dict[name]))
        else:
            ret = self.holder_dict[name]
        
        if keep==False:
            self.reset(name)

        return ret


def magic_append(target, name="default"):
    '''
        magic_append 和 magic_get 能够简化频繁出现的以下操作
        ```
            # 不使用 magic_append
            A = []
            B = []
            C = []
            for:
                A.append(a)
                B.append(b)
                C.append(c)
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)
            
            # 使用 magic_append
            for:
                magic_append([a, b, c], "SomeName")
            A, B, C = magic_get("SomeName", np.array)
        ```
    '''
    global magicHolderInstance
    if magicHolderInstance is None:
        magicHolderInstance = MagicHolder()
    # print(f"[DEBUG] Magic Append name: {name}")
    magicHolderInstance.magic_append(target, name)

def magic_get(name="default", func=None, keep=False):
    global magicHolderInstance
    if magicHolderInstance is None:
        return None
    return magicHolderInstance.magic_get(name, func, keep)

# # 对一个list的所有成员施加同一个函数后返回list
# def list_apply(list_in, func):
#     return list(map(func, list_in))


def seed_everything(seed, strict_mode=False):
    import torch
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if strict_mode:
        torch.backends.cudnn.deterministic = True

def write(content, newline=True):
    global to_writes
    to_writes += content
    if newline:
        to_writes += "\n"

def save(file_path, append=False):
    global to_writes
    if append:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(to_writes)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(to_writes)
    to_writes = ""

def describe(data, format=None, columns=None, func=print):
    content = pd.DataFrame(data, columns=columns).describe(percentiles=[.25, .5, .75, .95, .99])
    if format is None:
        pass
    elif format == "csv":
        content = content.T.to_csv()
    elif format == "json":
        content = content.T.to_json()
    if func:
        func(content.T)
    return content

# def get_mean_std(data:np.array, axis=None):
#     if axis is not None:
#         return np.mean(data, axis), np.std(data, axis)
#     else:
#         return np.mean(data), np.std(data)
        
def sns_bar_label(g, ax_num='single', fmt="%.2f", fontsize=10, label_type='edge', color='black'):
    if ax_num == "single":
        for bars in g.ax.containers:
            g.ax.bar_label(bars, fmt=fmt, fontsize=fontsize, label_type=label_type, color=color)
    elif ax_num == "multi":
        for ax in g.axes.flat:
            for bars in ax.containers:
                ax.bar_label(bars, fmt=fmt, fontsize=fontsize, label_type=label_type, color=color)

def set_allowed_gpu(gpu_list):
    b_set('allowed_gpu', gpu_list)

def set_current_device(current_gpu):
    b_set('current_device', current_gpu)

def get_current_device():
    device = b_get('current_device')
    if device is None:
        gpu_id = get_free_gpu()
        device = f"cuda:{gpu_id}"
        set_current_device(device)
    return device

def get_free_gpu():
    allowed_gpu = b_get('allowed_gpu')

    import subprocess
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A5 GPU | grep Free", shell=True
    )
    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").strip().split("\n")

    memory_available = [int(x.split()[2]) for x in gpu_info]
    gpu_id_freemem_descending = np.argsort(memory_available)[::-1]
    if allowed_gpu and isinstance(allowed_gpu, list):
        gpu_id_freemem_descending = [_ for _ in gpu_id_freemem_descending if _ in allowed_gpu]
        
    if len(gpu_id_freemem_descending) > 0:
        return gpu_id_freemem_descending[0]
    else:
        raise RuntimeError(f"There is No Available GPU!")    


def get_class(module_path, class_name):
    import importlib
    desired_module = importlib.import_module(module_path)
    desired_class = getattr(desired_module, class_name)
    return desired_class

def eval_dict_values(dict_to_parse):
    params_dict = {}
    for key, value in dict_to_parse.items():
        try:
            params_dict[key] = eval(str(value))
        except:
            params_dict[key] = value
    return params_dict


import torch

def batch_to_device(batch_data, device=None):
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(get_current_device() if device is None else device)
    else:
        return tuple(batch_to_device(item) for item in batch_data)
    # return (lambda device=device, batch_data=batch_data: [k.to(device) for k in batch_data])()
    
# if __name__ == "__main__":
#     magic_append([0,2,4,5,6], "UnitTest")
#     magic_append([1,3,4,5,7], "UnitTest")
#     magic_append([2,8,6,4,1], "UnitTest")
#     magic_append([6,9,8,5,7], "UnitTest")

#     pdb.set_trace()
#     rst = magic_get("UnitTest", func=np.array, keep=True)