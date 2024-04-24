# -*- coding: UTF-8 -*-
"""
将一个目录里的所有实验结果整理并展示出来, 可以直接复制到Excel中
"""
import argparse
import pandas as pd
import ipdb as pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stat the usefull results of experiments.")
    parser.add_argument('summary_path',         type=str, default=None, help="the outdir of Summary.py")
    parser.add_argument('-m','--metric',        type=str, default="Test Error", help="the useful metric to save, use comma to split if you there are multiple metrics")
    parser.add_argument('-n','--name',          type=str, default=None, help="extra name to save")
    parser.add_argument('-e','--epoch',         type=str, default=None, help="specify the epoch to save, default is all epochs")
    parser.add_argument('-c','--column',        type=str, default=None, help="unused")
    args = parser.parse_args()
    summary_path = args.summary_path
    table_name = '_'+args.name if args.name is not None else ""

    df = pd.read_csv(f"{summary_path}/Summary.csv")
    df = df.fillna("default")

    columns = df.columns.to_list()
    columns = columns[:columns.index('item')+1]
    if 'seed' in columns:
      columns.remove('seed')
    
    df = df.groupby(columns)[["mean", 'std']].mean()

    df.to_csv(f"{summary_path}/Summary_grouped.csv")

    item_index = df.index.names.index('item')
    df = df.loc[(*[slice(None)]*item_index, args.metric.split(','))]
    # df = df.to_frame()
    df["mean"] *=100
    df["std"] *=100

    epoch = args.epoch
    if epoch is None:
      names = list(df.index.names[:item_index])
      names.remove('epoch')
      table = pd.pivot_table(df, values="mean", index=names, columns=["epoch"])
      print(table.to_string())
      table.to_csv(f"{summary_path}/Summary_table{table_name}.csv", float_format='{:.4f}'.format)

      table = pd.pivot_table(df, values="std", index=names, columns=["epoch"])
      print(table.to_string())
      table.to_csv(f"{summary_path}/Summary_table{table_name}_std.csv", float_format='{:.4f}'.format)
    else:
      column = args.column
      epoch_index = df.index.names.index('epoch')
      df = df.loc[(*[slice(None)]*epoch_index, epoch)]
      names = list(df.index.names)
      names.remove(column)
      table = pd.pivot_table(df, values="mean", index=names, columns=[column])
      print(table.to_string())
      table.to_csv(f"{summary_path}/Summary_table{table_name}.csv", float_format='{:.4f}'.format)
