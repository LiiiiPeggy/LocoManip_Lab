# 环境参数

| 参数 | 说明 |
|------|------|
| 控制频率 | 50 Hz（仿真步长 dt = 0.005s，decimation = 4）|
| 动作空间 | 腿部 + 机械臂所有关节的位置目标 |
| 观测空间 | 本体感知信息（关节状态、基座速度、重力投影、高度扫描），历史长度为 3 |
| 指令 | 末端执行器位姿跟踪 + 基座速度指令 |

---

## 项目结构

```
LeggedManip_Lab/
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                     # 训练入口（RSL-RL PPO）
│   │   ├── play.py                      # 推理 / 策略评估
│   │   └── cli_args.py                  # 命令行参数定义
│   ├── list_envs.py                     # 列出所有已注册的 Gym 环境
│   ├── zero_agent.py                    # 零动作调试智能体
│   ├── random_agent.py                  # 随机动作调试智能体
├── source/LeggedManip_Lab/
│   ├── config/extension.toml            # Isaac Sim 扩展元数据
│   ├── setup.py                         # 包安装脚本
│   └── LeggedManip_Lab/
│       ├── assets/                      # 机器人模型文件（USD + 关节配置）
│       │   ├── go2_arx5/
│       │   ├── go2_piper/
│       │   ├── go1_arx5/
│       │   ├── go1_wx250s/
│       │   ├── b2_z1/
│       │   ├── b1_z1/
│       │   └── ago_z1/
│       └── tasks/manager_based/leggedmanip_lab/
│           ├── leggedmanip_lab_env_cfg.py   # 核心环境配置（场景、MDP、课程学习）
│           ├── mdp/
│           │   ├── rewards.py               # 奖励函数实现
│           │   ├── observations.py          # 观测项实现
│           │   ├── events.py                # 域随机化事件
│           │   ├── curriculums.py           # 课程学习逻辑
│           │   ├── pose_command_wbc.py      # WBC 末端位姿指令生成器
│           │   └── cfg/command_cfg.py       # 指令配置数据类
│           └── config/                      # 各机器人 Gym 注册 + 环境变体
│               ├── go2_arx5/
│               ├── go2_piper/
│               └── ...
└── mujoco/deploy/deploy_mujoco/         # 各平台 MuJoCo 部署脚本
    ├── go2_arx5/
    ├── go2_piper/
    ├── go1_arx5/
    ├── go1_wx250s/
    ├── b2_z1/
    ├── ago_z1/
    └── keyboard_controller.py           # MuJoCo 键盘遥操作
```

## 关键文件说明

| 文件 | 说明 |
|------|------|
| `scripts/rsl_rl/train.py` | 训练入口，支持 `--task`、`--num_envs`、`--headless`、`--max_iterations`、`--resume` 等参数 |
| `scripts/rsl_rl/play.py` | 推理入口，加载 checkpoint 并导出为 JIT/ONNX 格式 |
| `leggedmanip_lab_env_cfg.py` | 核心环境配置，定义场景、观测空间、动作空间、奖励项、终止条件、域随机化和课程学习，各平台配置均继承自此文件 |
| `mdp/rewards.py` | 所有奖励函数实现，包括末端跟踪、速度跟踪、关节惩罚、步态对称性、足端离地高度等 |
| `mdp/observations.py` | 观测项实现，包括关节状态、基座速度、重力投影、高度扫描及历史堆叠 |
| `mdp/curriculums.py` | 课程学习逻辑，根据训练进度动态扩展速度、位姿指令范围和地形难度 |
| `mdp/events.py` | 域随机化，包括物理材质、质量、质心、执行器增益、惯量和外力扰动 |
| `config/{robot}/` | 每个机器人包含两个环境配置文件和一个 PPO 智能体配置文件，均继承自核心配置并覆盖机器人特定参数 |
| `assets/{robot}/` | USD 关节文件及 Python 关节配置，定义关节限位、执行器参数和碰撞几何 |
