# -*- coding: UTF-8 -*-

import os, argparse
import ipdb as pdb

import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.4f}'.format)

import mtools.monkey as mk
from mtools import str2bool, list_con
import os.path as osp

def get_sub_dataframe(exp_run, exp_dir, epoch_dir, prefix, val_loss, epoch):
    eval_file_path = f"{exp_run}/{exp_dir}/evaluation/{epoch_dir}/{prefix}_eval_{val_loss}_best_des.csv" if prefix is not None else f"{work_dir}/{exp_dir}/evaluation/{epoch_dir}/eval_{val_loss}_best_des.csv"
    if not osp.exists(eval_file_path):
        print(eval_file_path)
        return
    des_df = pd.read_csv(eval_file_path, index_col=0)
    des_df.reset_index(inplace=True)
    des_df.rename(columns={'index': "item"}, inplace=True)
    keys = ['exp_run', 'prefix', 'epoch', 'val_metric']
    vals = [exp_run, prefix, epoch, val_loss]
    for keyeqval in exp_dir.strip().split(","):
        key = keyeqval.split("=")[0].strip().split(".")[-1].strip()
        val = keyeqval.split("=")[1].strip()
        keys.append(key)
        vals.append(val)
    vals = [[val] for val in vals]
    par_df = pd.DataFrame(dict(zip(keys, vals)))
    return pd.merge(par_df, des_df, how='cross')

def get_exp_runs(work_dir, exp_runs=[]):
    if osp.exists(osp.join(work_dir, 'multirun.yaml')):
        exp_runs.append(work_dir)
    for path in os.listdir(work_dir):
        if not osp.isdir(osp.join(work_dir, path)):
            continue
        else:
            exp_runs = get_exp_runs(osp.join(work_dir, path), exp_runs)
    return exp_runs

def get_dataframe(work_dir, prefix, val_loss):
    # work_dir is Output/<Path.out_dir>
    assert osp.isdir(work_dir)
    exp_runs = get_exp_runs(work_dir, [])
    print(exp_runs)
    for exp_run in exp_runs:
        for exp_dir in sorted(os.listdir(exp_run)):
            if not osp.isdir(f"{exp_run}/{exp_dir}"):
                continue
            try:
                epoch_dirs = os.listdir(f"{exp_run}/{exp_dir}/evaluation")
            except:
                print(f"{exp_run}/{exp_dir}/evaluation no item")
                if not args.ignore:
                    pdb.set_trace()
                continue
            
            if len(epoch_dirs):
                try:
                    epochs = [int(epoch_dir.split('<')[1]) for epoch_dir in epoch_dirs]
                except:
                    epoch_dirs = ['']
                    epochs = ['default']
            else:
                epoch_dirs = ['']
                epochs = ['default']
            epochs, epoch_dirs = zip(*sorted(zip(epochs, epoch_dirs)))
            for epoch, epoch_dir in zip(epochs, epoch_dirs):
                df = get_sub_dataframe(exp_run, exp_dir, epoch_dir, prefix, val_loss, epoch)
                if df is not None:
                    mk.magic_append([df], 'sub_dfs')
    merge_df = mk.magic_get('sub_dfs', pd.concat)
    if merge_df is not None:
        return merge_df[0]
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all results of experiments.")
    parser.add_argument('workdir',              type=str, nargs='+', default=None, help='the list of output dirs, or the father dir of output dirs')
    parser.add_argument('-l','--workdir_level', type=int, default=0, help='set the level of workdir, 0: output dirs, 1: the father dir of output dirs')
    parser.add_argument('-p','--prefix',        type=str, default=None, help='the prefix of the output csv, such as the filename of your output csv is {prefix}_eval_{val_loss}_best_des.csv')
    parser.add_argument('-o','--outdir',        type=str, help='the output dir of the summary csv')
    parser.add_argument('-v','--val_loss',      type=str, nargs='+', default=['Valid_Loss'], help='the list of monitor loss, the length of the list should be the same as the length of workdir')
    parser.add_argument('-i','--ignore',        type=str2bool, default=False, help='ignore the error of no item in evaluation dir, please add this option if your program is not finished')
    args = parser.parse_args()
    print(args)
    if args.workdir_level:
        subworkdirs = list_con([[osp.join(_, __) for __ in os.listdir(_)] for _ in args.workdir])
        args.workdir = list(filter(osp.isdir, subworkdirs))
    if len(args.val_loss)==1:
        args.val_loss = args.val_loss*len(args.workdir)
    assert len(args.val_loss) == len(args.workdir)
    for work_dir, val_loss in zip(args.workdir, args.val_loss):
        df = get_dataframe(work_dir, args.prefix, val_loss)
        # df.to_csv(f"{out_dir}/Summary_{suffix}.csv", index=False, header=True)
        if df is not None:
            mk.magic_append([df], 'dfs')
    [df] = mk.magic_get('dfs', pd.concat)
    if df is not None:
        columns = df.columns.to_list()
        item_index = columns.index('item')
        columns = columns[:item_index] + columns[item_index+11:] + columns[item_index:item_index+11]
        df = df.reindex(columns=columns)
        os.makedirs(args.outdir, exist_ok=True)
        if args.prefix:
            df.to_csv(f"{args.outdir}/Summary_{args.prefix}.csv", index=False, header=True)
        else:
            df.to_csv(f"{args.outdir}/Summary.csv", index=False, header=True)
