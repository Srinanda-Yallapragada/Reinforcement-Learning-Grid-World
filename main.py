import gymnasium as gym

from stable_baselines3 import PPO, DQN, SAC
# use DQN
# SAC
# PPO

# certian class order per episodes rather than just 1 class, n classrooms, d classes in a day d is your schedule.

# train only on some starting states, and see if it generalizes.

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import final_proj_in  # Import the package to ensure registration happens

# Separate evaluation env
# eval_env = Monitor(gym.make("GridWorld-v0", render_mode="human"))
eval_env = Monitor(gym.make("GridWorld-v0"))
#

# for tensorboard visualization
tensorboard_log_dir = "./tensorboard_logs/"

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=1000,
                             deterministic=True, render=False)

model = PPO(
    "MultiInputPolicy",
    "GridWorld-v0",
    device="cpu",
    tensorboard_log=tensorboard_log_dir,
    learning_rate=0.0001,
)
model.learn(400000, callback=eval_callback)

# Evaluate the trained model with human rendering
env = gym.make("GridWorld-v0", render_mode="human")  # Enable human rendering
obs, info = env.reset()
done = False

print("\n=== Visualizing the Trained Agent ===\n")
while not done:
    action, _ = model.predict(obs, deterministic=True)  # Predict the action
    action = int(action)  # Convert NumPy array to plain integer
    obs, reward, done, truncated, info = env.step(action)  # Perform the step
    env.render()  # Render the environment step-by-step
print("Visualization Complete!")
env.close()

# ####################################################
# import os
# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# from stable_baselines3 import SAC, PPO
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecNormalize
#
# env_id = "CartPole-v1"
# n_training_envs = 1
# n_eval_envs = 5
#
#
# # Create log dir where evaluation results will be saved
# eval_log_dir = "./eval_logs/"
# tensorboard_log_dir = "./tensorboard_logs/"
# os.makedirs(eval_log_dir, exist_ok=True)
# os.makedirs(tensorboard_log_dir, exist_ok=True)
#
# # Initialize a vectorized training environment with default parameters
# train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)
#
# # Separate evaluation env, with different parameters passed via env_kwargs
# # Eval environments can be vectorized to speed up evaluation.
# eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0)
#
# # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
# # When using multiple training environments, agent will be evaluated every
# # eval_freq calls to train_env.step(), thus it will be evaluated every
# # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
# eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
#                               log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
#                               n_eval_episodes=5, deterministic=True,
#                               render=False)
#
#
#
# # Set up and train the model
# model = PPO("MlpPolicy", train_env, tensorboard_log=tensorboard_log_dir, device='cpu')
# model.learn(100000, callback=[eval_callback])


####################################
#
# import gymnasium as gym
#
# from stable_baselines3 import DQN
# from stable_baselines3.common.evaluation import evaluate_policy
#
#
# # Create environment
# env = gym.make("LunarLander-v3", render_mode="rgb_array")
#
# # Instantiate the agent
# model = DQN("MlpPolicy", env, verbose=1)
# # Train the agent and display a progress bar
# model.learn(total_timesteps=int(50000), progress_bar=True)
# # Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading
#
# # Load the trained agent
# # NOTE: if you have loading issue, you can pass `print_system_info=True`
# # to compare the system on which the model was trained vs the current one
# # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
# model = DQN.load("dqn_lunar", env=env)
#
# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#
# # Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")
