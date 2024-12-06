from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="final_proj_in.envs:GridWorldEnv",
    max_episode_steps=300,
)
