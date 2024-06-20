import os
import numpy as np
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


def load_stock_data(targets=[], watches=[], from_date=None, to_date=None):
    datasets_targets = [csv_loader(i, "Date", from_date, to_date) for i in targets]
    datasets_watches = [csv_loader(i, "Date", from_date, to_date) for i in watches]

    df = datasets_targets[0]
    prices = df.loc[:, 'Open'].to_numpy()
    if len(datasets_targets) + len(datasets_watches) < 2:
        # signal_features = np.array([df.values.tolist()])
        # return prices.astype(np.float32), signal_features.astype(np.float32)
        pass

    all_dates = df.index
    for i in range(len(datasets_targets)):
        df1 = datasets_targets[i]
        df1 = df1[(df1.index >= df.index.min()) & (df1.index <= df.index.max())]
        all_dates = all_dates.union(df1.index)
        datasets_targets[i] = df1
    for i in range(len(datasets_watches)):
        df1 = datasets_watches[i]
        df1 = df1[(df1.index >= df.index.min()) & (df1.index <= df.index.max())]
        all_dates = all_dates.union(df1.index)
        datasets_watches[i] = df1

    dfs = []
    keys = []
    for i in range(len(datasets_targets)):
        df1 = datasets_targets[i]
        df1 = df1.reindex(all_dates)
        df1.fillna(0, inplace=True)
        dfs.append(df1)
        keys.append("t{}".format(i))
    for i in range(len(datasets_watches)):
        df1 = datasets_watches[i]
        df1 = df1.reindex(all_dates)
        df1.fillna(0, inplace=True)
        dfs.append(df1)
        keys.append("w{}".format(i))
    combined = pd.concat(dfs, axis=1, keys=keys)
    # Flatten the MultiIndex columns
    combined.columns = [f'{file}_{col}' for file, col in combined.columns]
    # Combine the columns into nested lists for each row
    combined = combined.apply(
            lambda row: [row[col:col+len(df.columns)].tolist() for col in range(0, len(row), len(df.columns))],
            axis=1)
    # Reset index to bring Date back as a column
    # combined = combined.reset_index()

    # Rename the columns
    # combined.columns = ['Date', 'Values']
    signal_features = np.array(combined.values.tolist())
    return prices.astype(np.float32), signal_features.astype(np.float32)
