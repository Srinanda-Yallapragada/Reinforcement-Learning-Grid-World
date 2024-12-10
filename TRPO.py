import gymnasium as gym

from stable_baselines3 import PPO, DQN, SAC
from sb3_contrib import TRPO

import numpy as np
# certain class order per episodes rather than just 1 class, n classrooms, d classes in a day d is your schedule.
# train only on some starting states, and see if it generalizes.

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import final_proj_in 

# Monitor evaluation environment
eval_env = Monitor(gym.make("GridWorld-v0"))


# For tensorboard visualization
tensorboard_log_dir = "./tensorboard_logs/"

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./best_logs/", log_path="./logs/", eval_freq=1000,deterministic=True, render=False)

# Initialize a model and define the environment
model = TRPO(
    "MultiInputPolicy",
    "GridWorld-v0",
    device="cpu",
    tensorboard_log=tensorboard_log_dir,
    learning_rate=0.02,
)

#Model learning 
model.learn(50000, callback=eval_callback)

# Load the best model from the runs
model=TRPO.load(path="./best_logs/best_model.zip")

# Evaluate the trained model
env = gym.make("GridWorld-v0", render_mode="rgb_array")

# Save a video of the visualization
env = gym.wrappers.RecordVideo(env=env, video_folder="./gifs", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)


print("\nRunning the Trained Agent\n")
# Done keeps track of whether the episode has terminated
done = False
# Count limits the number of steps per episode to 30, after 30 steps, the agent runs out of time 
count = 0

# Reset the environment
obs, info = env.reset()

while not done and count < 30:
    # Predict the action
    action, _ = model.predict(obs, deterministic=True)
    # Convert NumPy array to plain integer  
    action = int(action)  
    # Take a step according to the policy
    obs, reward, done, truncated, info = env.step(action) 
    # Update count
    count += 1

print("Visualization Saved")
env.close()