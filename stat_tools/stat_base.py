# -*- coding: UTF-8 -*-
"""
将一个目录里的所有实验结果整理并展示出来, 可以直接复制到Excel中

Usage:
  stat_base.py <summary_path> [options]

Options:
  -m <metric>, --metric <metric>  指标
  -e <epoch>, --epoch <epoch>  周期
  -c <column>, --column <column>  列
  -n <name>, --name <name>  名称
"""
from docopt import docopt
import pandas as pd
import ipdb as pdb

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    summary_path = arguments.summary_path
    metric = arguments.metric if arguments.metric is not None else "Test Error"

    table_name = '_'+arguments.name if arguments.name is not None else ""

    df = pd.read_csv(f"{summary_path}/Summary.csv")
    df = df.fillna("default")

    columns = df.columns.to_list()
    columns = columns[:columns.index('item')+1]
    if 'seed' in columns:
      columns.remove('seed')
    
    df = df.groupby(columns)[["mean", 'std']].mean()

    df.to_csv(f"{summary_path}/Summary_grouped.csv")

    item_index = df.index.names.index('item')
    df = df.loc[(*[slice(None)]*item_index, metric.split(','))]
    # df = df.to_frame()
    df["mean"] *=100
    df["std"] *=100

    epoch = arguments.epoch
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
      epoch = int(epoch)
      column = arguments.column
      epoch_index = df.index.names.index('epoch')
      df = df.loc[(*[slice(None)]*epoch_index, epoch)]
      names = list(df.index.names)
      names.remove(column)
      table = pd.pivot_table(df, values="mean", index=names, columns=[column])
      print(table.to_string())
      table.to_csv(f"{summary_path}/Summary_table{table_name}.csv", float_format='{:.4f}'.format)
