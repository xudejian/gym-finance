import os
import pandas as pd


def filter_and(base, f):
    if base is None:
        return f
    return base & f


def csv_loader(name, index_name, from_date=None, to_date=None):
    base_dir = '/content/drive/MyDrive/colab/finance'
    if not os.path.exists(base_dir):
        base_dir = os.path.expanduser("~/.cache/finance")
    path = os.path.join(base_dir, name + '.csv')
    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    filter_ = None
    if from_date:
        filter_ = filter_and(filter_, (df.index >= from_date))
    if to_date:
        filter_ = filter_and(filter_, (df.index < to_date))
    if filter_ is not None:
        return df.loc[filter_]
    return df
