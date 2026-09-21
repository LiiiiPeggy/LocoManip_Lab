# Environment Details

- **Control frequency**: 50 Hz (simulation dt = 0.005s, decimation = 4)
- **Action space**: Joint position targets for all leg + arm joints
- **Observation space**: Proprioceptive (joint states, base velocity, projected gravity, height scans) with history length of 3
- **Commands**: End-effector pose tracking + base velocity commands

---

## Project Structure

```
LeggedManip_Lab/
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                     # Training entry point (RSL-RL PPO)
│   │   ├── play.py                      # Inference / policy evaluation
│   │   └── cli_args.py                  # CLI argument definitions
│   ├── list_envs.py                     # List all registered Gym environments
│   ├── zero_agent.py                    # Zero-action agent for debugging
│   ├── random_agent.py                  # Random-action agent for debugging
├── source/LeggedManip_Lab/
│   ├── config/extension.toml            # Isaac Sim extension metadata
│   ├── setup.py                         # Package install script
│   └── LeggedManip_Lab/
│       ├── assets/                      # Robot model files (USD + articulation configs)
│       │   ├── go2_arx5/                #   Unitree Go2 + ARX5 arm
│       │   ├── go2_piper/               #   Unitree Go2 + AgileX Piper arm
│       │   ├── go1_arx5/                #   Unitree Go1 + ARX5 arm
│       │   ├── go1_wx250s/              #   Unitree Go1 + WX250S arm
│       │   ├── b2_z1/                   #   Unitree B2 + Z1 arm
│       │   ├── b1_z1/                   #   Unitree B1 + Z1 arm
│       │   └── ago_z1/                  #   Aliengo + Z1 arm
│       └── tasks/manager_based/leggedmanip_lab/
│           ├── leggedmanip_lab_env_cfg.py   # Core env config (scene, MDP, curriculum)
│           ├── mdp/
│           │   ├── rewards.py               # ~25 reward function implementations
│           │   ├── observations.py          # Observation term implementations
│           │   ├── events.py                # Domain randomization events
│           │   ├── curriculums.py           # Curriculum learning (terrain, velocity, pose)
│           │   ├── pose_command_wbc.py      # WBC end-effector pose command generator
│           │   └── cfg/command_cfg.py       # Command configuration dataclasses
│           └── config/                      # Per-robot Gym registration + env variants
│               ├── go2_arx5/                #   flat_env_cfg / wbc_env_cfg
│               ├── go2_piper/
│               ├── go1_arx5/
│               ├── go1_wx250s/
│               ├── b2_z1/
│               ├── b1_z1/
│               └── ago_z1/
└── mujoco/deploy/deploy_mujoco/         # MuJoCo deployment scripts per platform
    ├── go2_arx5/
    ├── go2_piper/
    ├── go1_arx5/
    ├── go1_wx250s/
    ├── b2_z1/
    ├── ago_z1/
    └── keyboard_controller.py           # Keyboard teleoperation for MuJoCo
```

## Key Files

| File | Description |
|------|-------------|
| `scripts/rsl_rl/train.py` | Training entry point. Supports `--task`, `--num_envs`, `--headless`, `--max_iterations`, `--experiment_name`, `--run_name`, `--resume`, `--video`. |
| `scripts/rsl_rl/play.py` | Inference entry point. Loads a checkpoint and exports to JIT/ONNX. |
| `leggedmanip_lab_env_cfg.py` | Central environment configuration. Defines scene setup, observation space, action space, reward terms, termination conditions, domain randomization, and curriculum learning. All per-platform configs inherit from this. |
| `mdp/rewards.py` | All reward function implementations — end-effector tracking, velocity tracking, joint penalties, gait symmetry, foot clearance, etc. |
| `mdp/observations.py` | Observation terms including joint states, base velocity, projected gravity, height scans, and history stacking. |
| `mdp/curriculums.py` | Curriculum learning logic that progressively expands command ranges (velocity, pose) and terrain difficulty based on training performance. |
| `mdp/events.py` | Domain randomization — physics materials, mass, center of mass, actuator gains, inertia, and push events. |
| `config/{robot}/` | Each robot has two env configs (`flat_env_cfg.py`, `wbc_env_cfg.py`) and a PPO agent config (`agents/rsl_rl_ppo_cfg.py`). They inherit from the core `LeggedManipLabEnvCfg` and override robot-specific settings. |
| `assets/{robot}/` | USD articulation files and Python articulation configs defining joint limits, actuator parameters, and collision geometry for each robot. |
