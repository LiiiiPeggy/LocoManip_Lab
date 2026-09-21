# LeggedManip Lab 项目详细介绍

## 1. 项目概述

LeggedManip Lab 是一个基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)（即 Omniverse Isaac Sim 的强化学习扩展框架）构建的四足机器人全身操作（Locomotion-Manipulation）强化学习训练平台。项目使用 **RSL-RL** 库提供的近端策略优化（PPO）算法，在 GPU 并行仿真环境下同时训练数千个环境实例，实现高效的策略学习。

核心目标：**训练一个统一的神经网络策略，同时控制四足机器人的腿部运动（ locomotion ）和机械臂操作（ manipulation ），实现全身协调控制（ Whole-Body Control ）。**

---

## 2. 研究背景

传统的机器人控制通常将 locomotion（移动）和 manipulation（操作）解耦为两个独立模块——先走到目标位置，再停下来操作。这种方式在动态场景下效率低下，且无法利用身体重心调整来辅助机械臂够取更远的目标。全身控制（WBC）则要求腿和臂协同工作：例如机器人可以主动倾斜身体以扩大机械臂的工作空间，或在行走过程中同步调整末端执行器位姿。

LeggedManip Lab 将 7 种不同的四足+机械臂组合平台统一到同一个训练框架中，提供 Flat（平地）、WBC（全身控制）两种标准化训练模式，支持大规模并行训练与课程学习。

---

## 3. 支持的机器人平台

项目共支持 **7 种机器人平台**，涵盖从小型（Go1）到中型（Go2/B1/B2）四足机器人，搭载不同类型的机械臂：

| 编号 | 任务 ID 前缀 | 四足平台 | 机械臂 | 腿部刚度/阻尼 | 站立高度 |
|:---:|-------------|:-------:|:-----:|:----------:|:------:|
| 1 | GO2-ARX5 | Unitree Go2 | ARX5 (6-DOF) | 30.0 / 0.6 | 0.55 m |
| 2 | GO2-PIPER | Unitree Go2 | Piper (6-DOF) | 30.0 / 0.6 | 0.35 m |
| 3 | GO1-ARX5 | Unitree Go1 | ARX5 (6-DOF) | 30.0 / 0.6 | 0.55 m |
| 4 | GO1-WX250S | Unitree Go1 | WX250S (6-DOF) | 30.0 / 0.6 | 0.55 m |
| 5 | B2-Z1 | Unitree B2 | Z1 (6-DOF) | 150.0 / 5.0 | 0.55 m |
| 6 | B1-Z1 | Unitree B1 | Z1 (6-DOF) | 150.0 / 5.0 | 0.55 m |
| 7 | AGO-Z1 | AGO | Z1 (6-DOF) | 120.0 / 3.0 | 0.55 m |

其中：
- **Go2 系列**搭载 ARX5/Piper 机械臂，腿刚度适中，是项目的主要验证平台
- **B1/B2 系列**搭载 Z1 机械臂，腿刚度较高（150 Nm/rad），适合大负载场景
- **Go1 系列**搭载 ARX5/WX250S 机械臂，是较早期的平台
- **AGO-Z1** 为异构平台，扩展框架的通用性

所有机器人的机械臂和腿部关节均使用 `DelayedPDActuatorCfg`（延迟PD执行器），实现位置控制接口。

---

## 4. 训练模式

每个平台支持两种训练模式，总计 **14 个训练任务**（外加 14 个对应的 Play 评估任务，合计 28 个 Gym 环境）：

### 4.1 Flat（平地模式）

- **地形**：平面（`terrain_type = "plane"`），无高度扫描传感器
- **特点**：最简单的训练设置，消除地形变化带来的复杂度，专注于基础的运动-操作协调能力训练
- **适用场景**：室内平坦地面、工厂车间等结构化环境
- **关键差异**：禁用地形课程学习、移除推搡事件（push_robot）、禁用高度扫描观测

### 4.2 WBC（全身控制模式）

- **命令生成器**：使用 `UniformPoseWBCCommand` 替代 Flat 模式中的标准 `UniformPoseCommand`
- **核心区别**：末端执行器指令在 **身体基座坐标系（Body Frame）** 下计算，而非世界坐标系。指令指定末端执行器相对 link0（基座）的位置和朝向，鼓励策略学习以身体姿态调整来辅助机械臂的够取
- **特点**：策略需要学习同时调整腿部和身体姿态来扩大末端执行器的工作空间（例如主动倾斜身体来够更远的目标）
- **适用场景**：需要最大化工作空间的动态操作任务

---

## 5. 技术架构详解

### 5.1 仿真设置

| 参数 | 值 |
|------|-----|
| 仿真时间步长 `dt` | 0.005 s (200 Hz) |
| 控制解耦 `decimation` | 4 |
| 控制频率 | 50 Hz |
| Episode 时长 | 20 s (1000 步) |
| 默认并行环境数 `num_envs` | 4096 |
| 环境间距 `env_spacing` | 2.5 m |
| 物理引擎 | PhysX GPU 加速 |

### 5.2 观测空间

观测分为 **Policy（策略）** 和 **Critic（价值函数）** 两组，策略网络拥有 3 步历史堆叠（`history_length = 3`），Critic 使用当前帧的完整信息。

#### Policy 观测（带噪声 + 缩放）

| 观测项 | 维度 | 噪声范围 | 缩放 | 说明 |
|--------|:---:|---------|:----:|------|
| `base_ang_vel` | 3 | [-0.2, 0.2] | 0.2 | 基座角速度（基座坐标系） |
| `projected_gravity` | 3 | [-0.05, 0.05] | — | 重力在基座坐标系下的投影 |
| `joint_pos_rel` | N | [-0.01, 0.01] | — | 关节相对默认位置的角度 |
| `joint_vel_rel` | N | [-1.5, 1.5] | 0.05 | 关节相对默认速度的角速度 |
| `actions` | N | — | — | 上一帧的动作 |
| `velocity_commands` | 3 | — | — | 基座速度指令 (lin_x, lin_y, ang_z) |
| `pos_commands` | 6 | — | — | 末端执行器位姿指令 (pos_xyz + quat_wxyz) |
> N = 12 + 6 = 18（12个腿关节 + 6个机械臂关节）

#### Critic 观测（无噪声，无历史堆叠）

Critic 额外包含 `base_lin_vel`（基座线速度，缩放 2.0），但不添加噪声，因为 Critic 仅在训练阶段使用，负责准确估计状态价值。

### 5.3 动作空间

- **类型**：`JointPositionActionCfg` — 关节位置指令
- **关节列表**：12 条腿关节（FR/FL/RR/RL × hip/thigh/calf）+ 6 条臂关节（joint1-joint6）
- **缩放**：`scale = 0.25`，使用 `use_default_offset = True`
- **数学表达**：`q_des = q_default + scale × tanh(action)`

### 5.4 指令系统

#### 末端执行器位姿指令（`UniformPoseCommand`）

| 参数 | 初始范围 | 最大课程范围 |
|------|---------|------------|
| pos_x | [0.40, 0.45] | [0.40, 0.70] |
| pos_y | [-0.05, 0.05] | [-0.35, 0.35] |
| pos_z | [0.05, 0.05] | [-0.20, 0.50] |
| roll/pitch/yaw | [0, 0] | [0, 0] |
| 重新采样间隔 | 8 ~ 10 s | — |

#### 基座速度指令（`UniformVelocityCommand`）

| 参数 | 初始范围 | 最大课程范围 |
|------|---------|------------|
| lin_vel_x | [-0.2, 0.2] | [-1.0, 1.0] |
| lin_vel_y | [-0.2, 0.2] | [-0.6, 0.6] |
| ang_vel_z | [-0.2, 0.2] | [-1.0, 1.0] |
| heading_command | False | — |
| 重新采样间隔 | 10 s | — |

指令范围的扩展由课程学习模块（`lin_vel_cmd_levels`, `ang_vel_cmd_levels`, `pos_cmd_levels`）根据训练表现自动推进。

### 5.5 奖励函数设计

奖励函数分为四大类，共约 25 个奖励项：

#### 任务跟踪奖励（正向）

| 奖励项 | 权重 | 核函数 | 说明 |
|--------|:---:|--------|------|
| `end_effector_position_tracking_exp` | **+3.0** | Exp, σ²=0.1 | 末端执行器位置跟踪（Body Frame） |
| `track_lin_vel_xy_exp` | **+3.0** | Exp, σ²=0.25 | 基座线速度跟踪（xy 平面） |
| `track_ang_vel_z_exp` | **+1.5** | Exp, σ²=0.25 | 基座偏航角速度跟踪 |
| `track_base_height_exp` | **+1.0** | Exp, σ²=0.02 | 基座目标高度跟踪（默认 0.3m） |
| `end_effector_orientation_tracking` | **-1.5** | L2 Norm | 末端执行器朝向误差惩罚 |
| `feet_air_time` | **+0.5** | Threshold | 足端置空时间奖励（>0.5s） |

#### 运动质量惩罚（负向）

| 奖励项 | 权重 | 说明 |
|--------|:---:|------|
| `lin_vel_z_l2` | -2.0 | 基座 z 轴线速度惩罚（防止弹跳） |
| `flat_orientation_l2` | -1.0 | 基座倾斜惩罚 |
| `ang_vel_xy_l2` | -0.05 | 基座 roll/pitch 角速度惩罚 |

#### 关节与能耗惩罚（负向）

| 奖励项 | 权重 | 说明 |
|--------|:---:|------|
| `dof_torques_l2` | -1e-5 | 关节扭矩 L2 惩罚 |
| `hip_torques_max` | -8e-5 | 髋关节最大扭矩惩罚 |
| `thigh_torques_max` | -8e-5 | 大腿关节最大扭矩惩罚 |
| `calf_torques_max` | -3e-5 | 小腿关节最大扭矩惩罚 |
| `arm_torques_max` | -5e-5 | 机械臂关节最大扭矩惩罚 |
| `dof_acc_l2` | -2.5e-7 | 关节加速度平滑惩罚 |
| `action_rate_l2` | -0.01 | 动作变化率惩罚（防抖） |
| `joint_power` | -2e-5 | 关节功率惩罚（|τ·ω|） |

#### 步态与对称性惩罚

| 奖励项 | 权重 | 说明 |
|--------|:---:|------|
| `feet_slide` | -0.1 | 足端着地滑动惩罚 |
| `feet_long_air` | -0.5 | 单脚悬空过久惩罚（>1.0s） |
| `air_time_variance` | -1.0 | 各脚置空时间方差惩罚 |
| `joint_mirror` | -0.1 | 左右关节对称性 |
| `hip_deviation` | -0.1 | 髋关节偏离默认位置 |
| `joint_deviation` | -0.02 | 腿关节偏离默认位置 |

**设计思路**：任务跟踪奖励权重最高（+3.0），确保策略优先完成操作目标；运动质量惩罚确保行为自然流畅；关节能耗惩罚权重极小，仅作为正则化项。整体奖励函数采用"密集奖励"范式——每一步（50 Hz）都计算所有奖励项。

### 5.6 终止条件

| 终止条件 | 阈值 | 说明 |
|---------|------|------|
| `time_out` | — | Episode 时间超限（20s/1000步） |
| `base_contact` | 接触力 > 0.5 N | 机器人基座碰地 |
| `bad_orientation` | `limit_angle = 1.0 rad` | 基座倾斜过大 |

### 5.7 域随机化

| 随机化项 | 范围 | 模式 | 说明 |
|---------|------|------|------|
| 物理材质 | 静摩擦 [0.5, 1.2]<br>动摩擦 [0.5, 1.2]<br>弹性 [0, 0.1] | startup | 64 个桶，保持一致性 |
| 刚体质量 | × [0.9, 1.1] | startup | 全部 body |
| 基座质心 | ±0.05 m (x, y)<br>±0.01 m (z) | startup | 仅基座 |
| 执行器增益 | × [0.8, 1.2] | reset | 刚度与阻尼 |
| 刚体惯量 | × [0.8, 1.2] | startup | 对角惯量缩放 |
| 基座初始化 | x, y ∈ [-0.5, 0.5]<br>yaw ∈ [-π, π] | reset | 随机初始位姿 |
| 关节初始化 | scale ∈ [0.5, 1.5] | reset | 随机初始关节角度 |
| 推搡事件 | vx, vy ∈ [-0.5, 0.5] | interval (10-15s) | 基座受力扰动 |

### 5.8 课程学习

三种课程同步运行：

1. **`lin_vel_cmd_levels`**：当线速度跟踪奖励达到权重 × 0.8 时，每 episode 将速度指令范围扩大 0.05 m/s
2. **`ang_vel_cmd_levels`**：同理，将角速度指令范围扩大 0.05 rad/s
3. **`pos_cmd_levels`**：将末端执行器位置指令范围扩大 0.05 m，朝向范围扩大 π/18 rad

---

## 6. PPO 训练配置

所有 8 个机器人的 PPO 超参数完全一致（共享同一网络架构），配置如下：

### 网络架构

| 组件 | 配置 |
|------|------|
| Actor 网络 | MLP [512, 256, 128], ELU 激活 |
| Critic 网络 | MLP [512, 256, 128], ELU 激活 |
| 动作分布 | Gaussian, `init_std=1.0`, log-std parametrization |

### PPO 超参数

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 24 |
| `max_iterations` | 10000 |
| `save_interval` | 200 iterations |
| `learning_rate` | 1e-3 |
| `schedule` | adaptive |
| `entropy_coef` | 0.005 |
| `clip_param` | 0.2 |
| `gamma` | 0.99 |
| `lam` (GAE) | 0.95 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `value_loss_coef` | 1.0 |

### 有效训练数据量

每 iteration 的有效样本数：`num_envs × num_steps_per_env = 4096 × 24 = 98,304` 步。

---

## 7. MuJoCo 部署

训练完成的 PyTorch 策略可以导出为 JIT（`policy.pt`）和 ONNX（`policy.onnx`）格式，通过 `mujoco/deploy/deploy_mujoco/` 目录下的脚本部署到 MuJoCo 物理引擎中进行实时推理和验证。每个机器人平台有独立的部署脚本，还包括一个 `keyboard_controller.py` 用于键盘遥控。

---

## 8. 依赖关系

```
LeggedManip_Lab
├── isaaclab (>= 4.5.0)        # Isaac Lab 核心框架
├── isaaclab_assets             # Isaac Lab 资产库
├── isaaclab_mimic              # Isaac Lab Mimic（模仿学习）
├── isaaclab_rl                 # Isaac Lab RL 封装
├── isaaclab_tasks              # Isaac Lab 任务框架
├── rsl-rl-lib                  # RSL-RL PPO 实现
├── gymnasium                   # OpenAI Gymnasium
├── torch                       # PyTorch
└── psutil                      # 系统资源监控
```

---

## 9. 适用场景

- **学术研究**：全身运动-操作协调控制的基准平台，支持对比不同算法在 7 个机器人上的性能
- **Sim-to-Real 迁移**：通过域随机化和课程学习训练的鲁棒策略，可迁移到真实四足机器人
- **机器人设计验证**：快速验证新的四足+机械臂组合的可行性，无需物理样机
- **多任务学习**：在 Flat/WBC 两种模式下交叉训练，研究多任务泛化能力
