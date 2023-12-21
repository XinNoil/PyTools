import pandas as pd

def sort_df(df, by, custom_dict):
    return df.sort_values(by=[by], key=lambda x: x.map(custom_dict))

def select_df(df, column, value):
    if isinstance(value, list):
        return df[df[column].isin(value)]
    else:
        return df[df[column]==value]