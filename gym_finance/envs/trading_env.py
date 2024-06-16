import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import numpy as np

from .data_loader import csv_loader


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # Up (close > open)
RED = (255, 0, 0)  # Down (close < open)

candle_width = 5  # Width of each candlestick


class TradingEnv(gym.Env):

    metadata = {
            'render_modes': ['human'],
            'render_fps': 60,
            }

    def __init__(self, balance=10000, window_size=5, dataset_loader=None, targets=[],
                 watches=[], from_date=None, to_date=None, epoch_size=20, render_mode=None):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        if dataset_loader is None:
            dataset_loader = csv_loader
        datasets_targets = [dataset_loader(i, "Date", from_date, to_date) for i in targets]
        datasets_watches = [dataset_loader(i, "Date", from_date, to_date) for i in watches]

        self.window_size = window_size
        self.prices, self.signal_features = self._process_data(datasets_targets, datasets_watches)

        # spaces
        self.action_space = gym.spaces.Discrete(100)
        INF = 1e10
        shape = (window_size,) + self.signal_features.shape[1:]
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=shape, dtype=np.float32,
        )

        # episode
        self._epoch_size = epoch_size
        self._start_tick = None
        self._end_tick = None
        self._terminated = None
        self._truncated = None
        self._current_tick = None
        self._initial_balance = balance
        self._position = [0,0,balance]
        self._position_history = None
        self._total_reward = None
        self.first_buy = None

        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._terminated = False
        self._truncated = False
        self._start_tick = self.np_random.integers(
                self.window_size, max(self.window_size, len(self.prices) - self._epoch_size))
        self._end_tick = min(self._start_tick + self._epoch_size, len(self.prices)-1)
        self._current_tick = self._start_tick
        self._position = [0, 0, self._initial_balance]
        self._position_history = [0] * len(self.prices)
        self._total_reward = 0.

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._terminated = False
        self._truncated = False
        self._current_tick += 1

        reward = self._calculate_reward(action)
        self._total_reward += reward

        if self._current_tick == self._end_tick:
            self._terminated = True
        if self._position[0] > 0 and action == 0:
            self._terminated = True

        self._update_position(action)
        self._position_history[self._current_tick] = action
        # print(self._current_tick, action)
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, self._terminated, self._truncated, info

    def _get_info(self):
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit(),
            position=self._position,
            first_buy=self.first_buy,
        )
        return info

    def _get_observation(self):
        if self._current_tick >= self.window_size:
            return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        else:
            output = self.signal_features[:self._current_tick+1]
            input_rows, input_cols = output.shape
            padding = np.zeros(
                    (self.window_size - input_rows, input_cols), dtype=np.float32)
            return np.vstack((padding, output))

    def render(self):
        if self.render_mode is None:
            return
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install pygame`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                        (self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.screen.fill(WHITE)
            pygame.display.set_caption('Candlestick Chart ')
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 16)

        def draw_candlestick(surface, x, open_price, high_price, low_price,
                             close_price, candle_width, chart_height, HL, LL):
            def _h(x):
                return int(chart_height * (1 - (x-LL) / HL))
            color = GREEN if close_price > open_price else RED
            pygame.draw.line(surface, BLACK, (x, _h(high_price)), (x, _h(low_price)), 1)
            rect = pygame.Rect(
                    x - candle_width // 2,
                    _h(max(open_price, close_price)),
                    candle_width,
                    int(abs(close_price - open_price)*chart_height/HL))
            pygame.draw.rect(surface, color, rect)

        end_index = int(self._current_tick)
        start_index = max(0, end_index-self.screen_width*9 //
                          (candle_width + 2)//10)
        HH, LL = 0, 1e10
        for i in range(max(0, end_index-self._epoch_size*3), end_index):
            _, H, L, _, _, _ = self.signal_features[i][0]
            HH = max(HH, H)
            LL = min(LL, L)
        surf_width = self.screen_width - 10
        surf_height = self.screen_height - 10
        surf = pygame.Surface((surf_width, surf_height))
        surf.fill(WHITE)
        for i in range(start_index, end_index):
            x = (i-start_index) * (candle_width + 2) + 5
            O, H, L, C, _, _ = self.signal_features[i][0]
            draw_candlestick(surf, x, O, H, L, C, candle_width, surf_height, HH-LL, LL)
        prev = 0
        pv = 0
        for i in range(start_index, end_index):
            p = self._position_history[i]
            x = (i-start_index) * (candle_width + 2) + 5
            v = surf_height * (1 - p / (self.action_space.n*10))
            if i > start_index:
                pygame.draw.line(surf, BLACK, (prev, pv), (x, v), 1)
            prev = x
            pv = v
        if self.first_buy:
            current_price = self.prices[self._current_tick]
            base_profit = int((current_price - self.first_buy) * 100 / self.first_buy)
            profit = int(self._total_profit() * 100 / self._initial_balance)
            msg = f'Profit: {profit}% / {base_profit}%'
            text = self.font.render(msg, True, BLACK)
            surf.blit(text, (0,0))
        text = self.font.render(f'Reward: {int(self._total_reward)}', True, BLACK)
        surf.blit(text, (0,20))
        self.screen.blit(surf, (5, 5))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    def save_rendering(self, filepath):
        if self.screen is not None:
            import pygame
            pygame.image.save(self.screen, filepath)

    def _process_data(self, datasets_targets, datasets_watches):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _total_profit(self):
        raise NotImplementedError

    def _update_position(self, action):
        raise NotImplementedError
