import gymnasium as gym
from . import agents

# Flat
gym.register(
    id="AGO-Z1-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AGOZ1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AGOZ1FlatPPORunnerCfg",
    },
)

gym.register(
    id="AGO-Z1-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AGOZ1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AGOZ1FlatPPORunnerCfg",
    },
)

# WBC
gym.register(
    id="AGO-Z1-WBC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:AGOZ1WBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AGOZ1WBCPPORunnerCfg",
    },
)

gym.register(
    id="AGO-Z1-WBC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:AGOZ1WBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AGOZ1WBCPPORunnerCfg",
    },
)

