import os
import numpy as np
from .io import read_file, write_file
from .py import is_number
from itertools import groupby

def is_csv(name):
    return name.endswith('_evaluate.csv')

def get_i(row):
    i =  row[:row.index(',')]
    return int(i) if is_number(i) else -1

def get_i2(row):
    row = row[row.index(',')+1:]
    return get_i(row)

# avg table
def avg_list(str_list):
    group_list = [list(g) for k,g in groupby(str_list[1:],key=get_i)]
    rows = []
    for group in group_list:
        table = [row.split(',') for row in group]
        _row = table[0]
        for c in range(6,len(_row)):
            if is_number(_row[c]):
                items = [float(row[c]) for row in table]
                _row[c] = str(round(np.mean(np.array(items)), 3))
        rows.append(','.join(_row))
    rows.insert(0, str_list[0])
    return rows

def sort_file(path, csv_name, sort_key=get_i):
    filename = os.path.join(path, csv_name)
    str_list = read_file(filename)
    sort_str_list = str_list[1:]
    sort_str_list.sort(key=sort_key)
    sort_str_list.insert(0, str_list[0])
    write_file(filename.replace('_evaluate.csv','_evaluate_sorted.csv'), sort_str_list)
    return sort_str_list

def avg_file(path, csv_name, sort_str_list):
    filename = os.path.join(path, csv_name)
    avg_str_list = avg_list(sort_str_list)
    write_file(filename.replace('_evaluate.csv','_evaluate_avg_sorted.csv'), avg_str_list)
    return [','.join([path, row]) for row in avg_str_list]

def sort_dir(path):
    print(path)
    dir_list = os.listdir(path)
    csv_names = list(filter(is_csv, dir_list))
    if len(csv_names):
        results = {}
        for csv_name in csv_names:
            print(os.path.join(path, csv_name))
            sort_str_list = sort_file(path, csv_name)
            results[csv_name] = avg_file(path, csv_name, sort_str_list)
    else:
        results = {}
        for folder in dir_list:
            if os.path.isdir(os.path.join(path, folder)):
                _results = sort_dir(os.path.join(path, folder))
                for csv_name in _results:
                    if csv_name in results:
                        results[csv_name].extend(_results[csv_name])
                    else:
                        results[csv_name] = _results[csv_name]
    return results

def avg_dir(path, results):
    if len(list(filter(is_csv, os.listdir(path)))) == 0:
        for csv_name in results:
            avg_name = csv_name.replace('_evaluate.csv','_evaluate_avg.csv')
            write_file(os.path.join(path, avg_name), results[csv_name])
            sort_name = csv_name.replace('_evaluate.csv','_evaluate_avg_sorted.csv')
            write_file(os.path.join(path, sort_name), sort_file(path, avg_name, sort_key=get_i2))

def sort_eval(path):
    results = sort_dir(path)
    avg_dir(path, results)