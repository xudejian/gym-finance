from absl.testing import absltest
from gym_finance.envs.stocks_dm_env import StocksDMEnv
from dm_env import test_utils

import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

    def make_object_under_test(self):

        # Date,Open,High,Low,Close,Adj Close,Volume
        one = np.array([10,20,9,15,15,1000], dtype=np.float32)
        data = np.array([one + i for i in range(100)], dtype=np.float32)
        prices = data[:, 0]
        return StocksDMEnv(prices=prices, signal_features=data)

if __name__ == '__main__':
    absltest.main()
