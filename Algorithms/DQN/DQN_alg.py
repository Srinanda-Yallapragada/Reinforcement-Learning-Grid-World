import gymnasium as gym

from stable_baselines3 import PPO, DQN, SAC
import numpy as np
# certain class order per episodes rather than just 1 class, n classrooms, d classes in a day d is your schedule.
# train only on some starting states, and see if it generalizes.

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import final_proj_in 

def DQN_run(lr, df):
    # Monitor evaluation environment
    eval_env = Monitor(gym.make("GridWorld-v0"))

    #Set random seed
    np.random.seed(0)

    # For tensorboard visualization
    tensorboard_log_dir = "./DQN/results/tensorboard_logs/"

    tb_log_name="LR"+str(lr)+"DF"+str(df)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./DQN/results/logs/", log_path="./DQN/results/logs/", eval_freq=1000,deterministic=True, render=False)

    # Initialize a model and define the environment
    model = DQN(
        "MultiInputPolicy",
        env="GridWorld-v0",
        tensorboard_log=tensorboard_log_dir,
        learning_rate=lr,
        gamma=df,
        # exploration_fraction=0.4,
        # batch_size=64,
        # target_update_interval=1000,
        # verbose=1
        # exploration_final_eps=0.1,
    )

    #Model learning 
    model.learn(120000, callback=eval_callback,tb_log_name=tb_log_name)

    # # Load the best model from the runs
    model=DQN.load(path="./DQN/results/logs/best_model.zip")

    # Evaluate the trained model
    env = gym.make("GridWorld-v0", render_mode="rgb_array")

    # Save a video of the visualization
    env = gym.wrappers.RecordVideo(env=env, video_folder="./DQN/results/gifs", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)


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