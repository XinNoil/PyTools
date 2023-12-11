def sort_df(df, by, custom_dict):
    return df.sort_values(by=[by], key=lambda x: x.map(custom_dict))