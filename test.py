import gymnasium as gym
import gym_finance
import gym_finance.envs


prices, data = gym_finance.envs.load_stock_data(
        from_date="2001-01-01", to_date="2020-01-01",
        targets=["GOOGL"])
env = gym.make('stocks-v0',
               prices=prices,
               data=data,
               epoch_size=2000,
               render_mode='human')

print(env.action_space)
print(env.observation_space)
observation, info = env.reset()
while True:
    action = env.action_space.sample()
    print(action)
    observation, reward, done, _, info = env.step(action)

    env.render()
    if done:
        print("info:", info)
        break

input("Press Enter to continue...")
env.unwrapped.save_rendering("test.png")
env.close()
