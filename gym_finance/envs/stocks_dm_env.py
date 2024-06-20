from bsuite.environments import base
import dm_env
import numpy as np
from .data_loader import load_stock_data


class StocksDMEnv(base.Environment):
    def __init__(self, prices, signal_features, epoch_size=20):
        super().__init__()
        self._initial_balance = 100000
        self.prices = prices
        self.signal_features = signal_features

        # episode
        self._episode_length = epoch_size
        self._start_tick = None
        self._end_tick = None
        self._current_tick = self._start_tick
        self._position = [0, 0, self._initial_balance]
        self._position_history = None
        self._total_reward = None
        self.first_buy = None

    def _reset(self) -> dm_env.TimeStep:
        """Returns a `timestep` namedtuple as per the regular `reset()` method."""
        self._start_tick = np.random.randint(
                0, max(0, len(self.prices) - self._episode_length))
        self._end_tick = min(self._start_tick + self._episode_length, len(self.prices)-1)
        self._current_tick = self._start_tick
        self._position = [0, 0, self._initial_balance]
        self._position_history = [0] * len(self.prices)
        self._total_reward = 0.
        self.first_buy = None

        return dm_env.restart(self._get_observation())

    def _get_observation(self):
        return self.signal_features[self._current_tick]

    def bsuite_info(self) -> Dict[str, Any]:
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit(),
            position=self._position,
            first_buy=self.first_buy,
        )

    def _step(self, action: int) -> dm_env.TimeStep:
        """Returns a `timestep` namedtuple as per the regular `step()` method."""
        reward = self._calculate_reward(action)
        self._update_position(action)
        self._position_history[self._current_tick] = action

        self._current_tick += 1
        if self._current_tick >= self._end_tick:
            return dm_env.termination(reward=reward, observation=self._get_observation())

        return dm_env.transition(reward=reward, observation=self._get_observation())

    def _total_profit(self):
        current_price = self.prices[self._current_tick]
        return self._position[0] * current_price + self._position[2] - self._initial_balance

    def _calculate_reward(self, action):
        current_price = self.prices[self._current_tick]
        if current_price < 0.01:
            return 0
        max_shares = self._position[0] + self._position[2] // current_price
        new_shares = int(max_shares * action)
        if new_shares < self._position[0] and self._position[0]>0:
            return (self._position[0] - new_shares) * (current_price - self._position[1]/self._position[0])

        return 0

    def _update_position(self, action):
        current_price = self.prices[self._current_tick]
        if current_price < 0.01:
            return
        shares = self._position[0]
        cost = self._position[1]
        balance = self._position[2]
        max_shares = shares + balance // current_price
        new_shares = int(max_shares * action)
        if new_shares < shares:
            self._position = [
                    new_shares,
                    cost*new_shares/shares,
                    balance + (shares-new_shares)*current_price]
        elif new_shares > shares:
            if self.first_buy is None:
                self.first_buy = current_price
            self._position = [
                    new_shares,
                    cost + (new_shares-shares)*current_price,
                    balance - (new_shares-shares)*current_price]
        return new_shares - shares

    def observation_spec(self):
        return dm_env.specs.BoundedArray(
                name='observation', shape=(1,), dtype=np.float32,
                minimum=0.0, maximum=np.inf)

    def action_spec(self):
        return dm_env.specs.BoundedArray(
                name='action', shape=(), dtype=np.float32,
                minimum=0.0, maximum=1.0)


def load_my_stock_env(seed: int) -> base.Environment:
    prices, features = load_stock_data(
            targets=["GOOGL"],
            from_date="2001-01-01",
            to_date="2020-01-01",
            )
    return StocksDMEnv(prices=prices, signal_features=features)
