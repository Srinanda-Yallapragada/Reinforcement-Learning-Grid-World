import gymnasium as gym
import numpy as np
import final_proj_in
from time import sleep

env = gym.make("GridWorld-v0", render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env=env, video_folder="./gifs", name_prefix="test-video",episode_trigger=lambda x: x % 2 == 0)

obs, _ = env.reset()
rewards_list = []
observations_list = []

total_reward = 0

# actions = [0, 0, 0, 0, 0, 0, 0, 0, 0,1,3]
# 0: np.array([1, 0]),  # right
# 1: np.array([0, 1]),  # down
# 2: np.array([-1, 0]),  # left
# 3: np.array([0, -1]),  # up
# actions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1]
actions = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]
frames = []

for step, action in enumerate(actions):

    new_obs, reward, done, truncated, info = env.step(action)
    # env.render()  # Render the environment step-by-step
    frames.append(env.render())

    total_reward += reward
    rewards_list.append(reward)
    observations_list.append(new_obs)

    if done:
        break


env.close()

print(f"Total Reward: {total_reward}")
print("Rewards Sequence:", rewards_list)
print("Observation Sequence:", observations_list)
print(frames)