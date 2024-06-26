from .trading_env import TradingEnvGym


class StocksEnvGym(TradingEnvGym):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _total_profit(self):
        current_price = self.prices[self._current_tick]
        return self._position[0] * current_price + self._position[2] - self._initial_balance

    def _calculate_reward(self, action):
        # 0 sell 1 hold 2 buy
        if action != 0:
            return 0
        current_price = self.prices[self._current_tick]
        return self._position[0] * max(0,current_price) - self._position[1]

    def _update_position(self, action):
        # 0 sell 1 hold 2 buy
        if action == 1:
            return 0
        current_price = self.prices[self._current_tick]
        if current_price < 0.01:
            return
        shares = self._position[0]
        cost = self._position[1]
        balance = self._position[2]
        max_shares = shares + balance // current_price
        if action == 2:
            if self.first_buy is None:
                self.first_buy = current_price
            self._position = [
                    max_shares,
                    cost + (max_shares-shares)*current_price,
                    balance - (max_shares-shares)*current_price]
            return
        self._position = [
                0, 0, balance + shares*current_price]
