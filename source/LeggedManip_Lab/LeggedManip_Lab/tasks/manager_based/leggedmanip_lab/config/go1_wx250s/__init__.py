import gymnasium as gym
from . import agents

# Flat
gym.register(
    id="GO1-WX250S-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go1WX250SFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go1WX250SFlatPPORunnerCfg",
    },
)

gym.register(
    id="GO1-WX250S-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go1WX250SFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go1WX250SFlatPPORunnerCfg",
    },
)

# WBC
gym.register(
    id="GO1-WX250S-WBC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:Go1WX250SWBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go1WX250SWBCPPORunnerCfg",
    },
)

gym.register(
    id="GO1-WX250S-WBC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:Go1WX250SWBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go1WX250SWBCPPORunnerCfg",
    },
)

