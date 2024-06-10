import gymnasium as gym
import gym_finance
import time


env = gym.make('stocks-v0',
               balance=10000,
               targets=["GOOGL"], watches=['^IXIC'],
               render_mode='human')

print(env.action_space)
print(env.observation_space)
observation, info = env.reset(seed=2023)
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    env.render()
    if done:
        print("info:", info)
        break

env.unwrapped.save_rendering("test.png")
env.close()
