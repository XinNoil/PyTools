import argparse
import os.path as osp
import pandas as pd
from mtools import str2bool
import ipdb as pdb

parser = argparse.ArgumentParser()
parser.add_argument('-p','--summary_path',  type=str)
parser.add_argument('-n','--name',          type=str, default=None)
parser.add_argument('-c','--columns',       type=str, nargs='+', default=None)
parser.add_argument('-nc','--new_columns',  type=str, nargs='+', default=None)
parser.add_argument('-d','--defaults',      type=str, nargs='+', default=None)
parser.add_argument('-r','--replaces',      type=str, nargs='+', default=None)
parser.add_argument('-t','--pivot_table',   type=str, nargs='+', default=None)
parser.add_argument('--order',              type=str, nargs='+', default=None)
args = parser.parse_args()

df = pd.read_csv(args.summary_path)

print(' '.join(df.columns.to_list()))
if args.order is not None:
    order_values = df[args.order[0]].unique()
    print(' '.join(order_values))
print('')

if args.columns is not None:
    print(args.columns)
    df = df[args.columns]
    if args.order is not None:
        column = args.order[0]
        items = args.order[1:]
        df.insert(0, 'tmp_order', len(items))
        for i,item in enumerate(items):
            df.loc[df[column]==item, 'tmp_order'] = i
        # for _ in order_values:
        #     if _ not in items:
        #         df.drop(df[df[column] == _].index, inplace=True)

    if args.new_columns is not None:
        df.rename(columns=dict(zip(args.columns, args.new_columns)), inplace=True)

columns = df.columns.to_list()
if args.defaults is not None:
    for column, default in zip(columns, args.defaults):
        df.loc[df[column]=='default', column] = default
if args.replaces is not None:
    for replace in args.replaces:
        column, old_str, new_str = replace.split(':')
        df[column] = df[column].str.replace(old_str, new_str)
if args.pivot_table is not None:
    columns.remove(args.pivot_table[1])
    columns.remove(args.pivot_table[0])
    df = pd.pivot_table(df, values=args.pivot_table[0], index=columns, columns=args.pivot_table[1])
df = df.sort_values(by=columns)
df.reset_index(inplace=True)
if args.order is not None:
    df.drop(columns='tmp_order', inplace=True)
    
print(df)
df.to_csv(args.summary_path.replace('.csv', f'_{args.name}.csv'), index=False)
