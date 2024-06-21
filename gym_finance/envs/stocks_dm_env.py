from typing import Any,Dict,Optional
from bsuite.environments import base
import dm_env
import numpy as np
from .data_loader import load_stock_data

_ACTIONS = (-100, -20, 0, 20, 100)
# sell_all, sell_20p, noop, buy_20p, buy_all


class StocksDMEnv(base.Environment):
    def __init__(self, prices, signal_features, epoch_size=60):
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

        self.bsuite_num_episodes = 10000

    def _reset(self) -> dm_env.TimeStep:
        """Returns a `timestep` namedtuple as per the regular `reset()` method."""
        self._start_tick = np.random.randint(0, len(self.prices))
        self._end_tick = min(self._start_tick + self._episode_length, len(self.prices)-1)
        print("start_tick", self._start_tick, "end_tick", self._end_tick)
        self._current_tick = self._start_tick
        self._position = [0, 0, self._initial_balance]
        self._position_history = [0] * len(self.prices)
        self._total_reward = 0.
        self.first_buy = None

        return dm_env.restart(self._observation())

    def _observation(self):
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
        v_action = 2 #noop
        if action >=0 and action < len(_ACTIONS):
            v_action = _ACTIONS[action]
        reward = np.float32(self._calculate_reward(v_action))
        self._total_reward += reward
        self._update_position(v_action)
        self._position_history[self._current_tick] = v_action

        self._current_tick += 1
        if self._current_tick >= self._end_tick:
            self._current_tick = self._end_tick
            return dm_env.termination(reward=reward, observation=self._observation())

        return dm_env.transition(reward=reward, observation=self._observation())

    def _total_profit(self):
        current_price = self.prices[self._current_tick]
        return self._position[0] * current_price + self._position[2] - self._initial_balance

    def _calculate_reward(self, v_action):
        current_price = self.prices[self._current_tick]
        if current_price < 0.01 or v_action >= 0:
            return 0
        # max_shares = self._position[0] + self._position[2] // current_price
        # new_shares = int(max_shares * v_action / 100)
        sell = min(int(abs(self._position[0] * v_action / 100)), self._position[0])
        if sell <= self._position[0] and self._position[0]>0:
            return sell * (current_price - self._position[1]/self._position[0])

        return 0

    def _update_position(self, v_action):
        current_price = self.prices[self._current_tick]
        if current_price < 0.01 or v_action == 0:
            return 0
        shares = self._position[0]
        cost = self._position[1]
        balance = self._position[2]
        if v_action > 0:
            buy_shares = balance * v_action / 100 // current_price
            buy_cost = buy_shares * current_price
            self._position = [
                    shares + buy_shares,
                    cost + buy_cost,
                    balance - buy_cost]
            if self.first_buy is None:
                self.first_buy = current_price
            return buy_shares
        if v_action < 0 and shares > 0:
            sell_shares = int(shares * v_action / 100)
            sell_cost = sell_shares * cost/shares
            self._position = [
                    shares + sell_shares,
                    cost + sell_cost,
                    balance - sell_shares * current_price]
            return sell_shares
        return 0

    def observation_spec(self):
        return dm_env.specs.BoundedArray(
                name='observation', shape=(6,), dtype=np.float32,
                minimum=0.0, maximum=np.inf)

    def action_spec(self):
        return dm_env.specs.DiscreteArray(num_values=len(_ACTIONS), name='action')

    def reward_spec(self):
        return dm_env.specs.Array(shape=(), dtype=np.float32, name='reward')


def load_stock_env(seed: int, **kwargs) -> base.Environment:
    prices, features = load_stock_data(**kwargs)
    return StocksDMEnv(prices=prices, signal_features=features)
