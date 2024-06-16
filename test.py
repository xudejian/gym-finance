import gymnasium as gym
import gym_finance
import time


env = gym.make('stocks-v0',
               balance=10000,
               from_date="2001-01-01",
               to_date="2020-01-01",
               targets=["GOOGL"], watches=['^IXIC'],
               render_mode='human')

print(env.action_space)
print(env.observation_space)
observation, info = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    env.render()
    if done:
        print("info:", info)
        break

input("Press Enter to continue...")
env.unwrapped.save_rendering("test.png")
env.close()
