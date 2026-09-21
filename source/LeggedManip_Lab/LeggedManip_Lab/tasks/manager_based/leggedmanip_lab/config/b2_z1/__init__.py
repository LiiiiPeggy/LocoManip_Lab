import gymnasium as gym
from . import agents

# Flat
gym.register(
    id="B2-Z1-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:B2Z1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2Z1FlatPPORunnerCfg",
    },
)

gym.register(
    id="B2-Z1-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:B2Z1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2Z1FlatPPORunnerCfg",
    },
)

# WBC
gym.register(
    id="B2-Z1-WBC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:B2Z1WBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2Z1WBCPPORunnerCfg",
    },
)

gym.register(
    id="B2-Z1-WBC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:B2Z1WBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2Z1WBCPPORunnerCfg",
    },
)

