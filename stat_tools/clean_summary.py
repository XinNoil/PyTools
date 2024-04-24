import argparse
import os.path as osp
import pandas as pd
from mtools import str2bool
import ipdb as pdb

parser = argparse.ArgumentParser(description="Clean the summary of experiments.")
parser.add_argument('-p','--summary_path',  type=str, help='the outdir of Summary.py')
parser.add_argument('-n','--name',          type=str, default=None, help="the postfix name of the output file")
parser.add_argument('-c','--columns',       type=str, nargs='+', default=None, help="the columns to be output")
parser.add_argument('-nc','--new_columns',  type=str, nargs='+', default=None, help="the new columns name")
parser.add_argument('-d','--defaults',      type=str, nargs='+', default=None, help="the default values for the columns")
parser.add_argument('-r','--replaces',      type=str, nargs='+', default=None, help="the list of replaces for the columns, $new_column$:old_str:new_str, eg., -r method:tmp:, method:dnn:DNN")
parser.add_argument('-t','--pivot_table',   type=str, nargs='+', default=None, help="the parameters of pivot_table, $value_column$ $columns$, eg., -t Error max_err_limit")
parser.add_argument('--order',              type=str, nargs='+', default=None, help="sort rows by one column, eg., --order method, you can also specify the order, eg., order method DNN CNN LSTM")
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
if 'index' in df.columns.to_list():
    df.drop(columns='index', inplace=True)
if args.order is not None:
    if 'tmp_order' in df.columns:
        df.drop(columns='tmp_order', inplace=True)

print(df)
df.to_csv(args.summary_path.replace('.csv', f'_{args.name}.csv'), index=False)
