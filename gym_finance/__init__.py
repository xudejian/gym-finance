from gymnasium.envs.registration import register


register(
    id='stocks-v0',
    entry_point='gym_finance.envs:StocksEnvGym',
)
