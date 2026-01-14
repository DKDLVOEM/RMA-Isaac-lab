import gymnasium as gym
from .pendulum_env_cfg import PendulumEnvCfg

gym.register(
    id="Isaac-Pendulum-Tracking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PendulumEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.pendulum.agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
    },
)
