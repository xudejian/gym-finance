import numpy as np
import pandas as pd

from .trading_env import TradingEnv


class StocksEnv(TradingEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self, datasets_targets, datasets_watches):
        df = datasets_targets[0]
        prices = df.loc[:, 'Open'].to_numpy()

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

    def _total_profit(self):
        current_price = self.prices[self._current_tick]
        return self._position[0] * current_price + self._position[2] - self._initial_balance

    def _calculate_reward(self, action):
        current_price = self.prices[self._current_tick]
        max_shares = self._position[0] + self._position[2] // current_price
        new_shares = max_shares * action // self.action_space.n
        if new_shares < self._position[0] and self._position[0]>0:
            return (self._position[0] - new_shares) * (current_price - self._position[1]/self._position[0])

        return 0

    def _update_position(self, action):
        current_price = self.prices[self._current_tick]
        shares = self._position[0]
        cost = self._position[1]
        balance = self._position[2]
        max_shares = shares + balance // current_price
        new_shares = max_shares * action // self.action_space.n
        if new_shares < shares:
            self._position = [
                    new_shares,
                    cost*(shares-new_shares)/shares,
                    balance + (shares-new_shares)*current_price]
        elif new_shares > shares:
            self._position = [
                    new_shares,
                    cost + (new_shares-shares)*current_price,
                    balance - (new_shares-shares)*current_price]
        return new_shares - shares
