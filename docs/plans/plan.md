# RangerBox-CR10-Lidar WBC 训练移植计划

**状态:方案已修订为「底盘 + 机械臂联合 WBC」,可开工** · 拟稿 2026-09-22 ·
决策确认 2026-09-23 · **方案修订 2026-09-23** · 参照平台:GO2-PIPER

把 `rangerboxcr10lidar_description`(轮式移动操作机器人)接入 LocoManip_Lab,
训练 WBC 策略:**由 RL 同时决定 Ranger 底盘运动与 CR10 六个关节**,使末端执行器跟踪目标位姿,
并打通 MuJoCo 部署与实机(agx)链路。

**策略动作(8 维)**:`[vx, wz, cr10_joint1, cr10_joint2, cr10_joint3, cr10_joint4, cr10_joint5, cr10_joint6]`,
其中 `vx ∈ [-0.5, 0.5] m/s`、`wz ∈ [-0.5, 0.5] rad/s`;**不含 `vy`**;
**RL 不直接控制 4 个转向关节与 4 个轮速关节**。

**核心目标**:RL **自主移动 Ranger 底盘以扩大 CR10 的有效工作空间**,并与机械臂**联合**
完成**世界系**末端目标位姿跟踪。

**决策状态**:A/B/C 三层共 15 项已由用户确认(见 §5),其中 **B1 于 2026-09-23 修订** ——
由"底盘速度作外部指令"改为"**RL 同时控制底盘与机械臂**"(§4);**B7/R5 同日确定** ——
EE 目标内部固定在世界系、Policy 观测用 base-relative 量、Reward 用世界系误差(§4.2)。
仿真与实机的接口边界见 §6;剩余待定项(R1、R3、R4)都不阻塞开工,见 §9。
标 **⚠️** = 已核实的障碍或未验证项。

---

## 1. 目标与非目标

**目标**

1. 把该机器人转换/接入 Isaac Lab,注册 `RANGER-CR10-WBC` / `-Play` 与 `RANGER-CR10-Flat` / `-Play`
   (4 个,验收行数 28 → 32 不变)。
   **`-Flat` 变体在新方案下重新定位为"冻结底盘 + 末端位姿跟踪"的固定基座基线**
   —— 它正好充当 §8 里"RL 是否真的用上了底盘"的对照组(§7 阶段 3)。
2. 训练 WBC 策略:**给定末端目标位姿,由 RL 自主协调 Ranger 底盘与 CR10 机械臂**完成跟踪
   —— `(vx, wz)` 不再是外部 command,而是策略输出的一部分。观测与课程架构沿用 GO2-PIPER,
   奖励结构按 §4 调整。
3. MuJoCo 部署验证(复用 `mujoco/deploy/` 架构,但脚本需重写)。
4. 策略的**关节顺序、观测布局、动作语义、控制频率**与 agx 实机对齐,
   且**仿真与实机的底盘动作语义一致**(两边都是 `(vx, wz)` → 底盘执行,§4)。

**非目标**

- **实机上机运行**。agx 里**不存在**任何策略推理代码(全仓库 grep `onnx|torchscript|isaac ?lab|checkpoint`
  零命中),实机侧要新写一个 ROS 节点,而且这个机器人**没有 `/cmd_vel` 看门狗**(见 §6.4)。
  本期只做接口对齐与文档化,上机是独立项目。
- **底盘关节级控制**:RL 只输出 `(vx, wz)`,4 个转向角与 4 个轮速由底盘控制器解算(§4)。
- **蟹行 `vy`**:本期不加入(§4)。
- 抓取(夹爪本次不进动作空间,与 GO2-PIPER 一致)。
- 感知输入(lidar / D435 / D455 只作为模型质量,不进策略)。
- 非平坦地形、轮胎侧滑辨识。

---

## 2. 被移植的机器人:已核实的模型事实

来源:`rangerboxcr10lidar_description/urdf/rangercr10lidar.urdf`(SW2URDF 导出)+
`agx/ranger_ros` 驱动参数。**注意仓库里有两份完全相同的 description**
(`/rangerboxcr10lidar_description/` 与 `agx/rangerboxcr10lidar_description/`,URDF md5 一致),
已定:**保留两份,训练时以仓库根目录那份为准**(§5 C2)。

| 项 | 值 |
|---|---|
| 规模 | 52 link / 51 joint(29 fixed,22 非固定) |
| 根 link | `base_link`(底盘),无 world link → Isaac 里作为浮动基座 |
| 底盘形式 | **4 轮独立转向(4WS/4WD)**,非差速 |
| 底盘关节 | 每角 1 转向(revolute,±1.57 rad)+ 1 驱动(**continuous**,URDF 无 limit) |
| 转向关节名 | `fr_steering_joint` / `fl_steering_wheel_joint` / `rl_steering_wheel_joint` / `rr_steering_wheel_joint`(命名不一致来自导出器) |
| 驱动关节名 | `fr/fl/rl/rr_wheel_joint` |
| 轮距/轴距 | URDF ±0.445 / ±0.28 m;**与驱动参数 `wheelbase=0.90, track=0.56` 吻合 ✅** |
| 机械臂 | **Dobot CR10**,`cr10_joint1..6` ⚠️ 全部 `effort="0" velocity="0"` |
| 夹爪 | **DH AG95**,仅 `gripper_finger1_joint` 独立驱动,**另 7 个是 `<mimic>`** |
| 传感器 | lidar(`lidar_link0`)、D435(在 lidar 上)、**D455(腕部,在 `gripper_base_link`)** |
| 总质量 | **176.0 kg**(底盘 88.8 / 四轮 45.9 / 臂 24.8 / 其余 16.5) |
| 网格 | 27 个 STL,`package://` URI;原始 83 MB / 172.8 万面,**已减面至 19 MB / 38.1 万面**(§5.2) |
| 缺失 | 无 `<dynamics>`(阻尼/摩擦)、臂关节无力矩速度上限、无悬架、无 `<sensor>` |

独立可控自由度 = 15(8 底盘 + 6 臂 + 1 夹爪)。**策略动作不覆盖底盘关节**:底盘由 RL 输出的
`(vx, wz)` 经底盘控制器驱动,因此动作维度 = 2(底盘)+ 6(臂)= **8**(§4)。

### ⚠️ 2.1 转换前必须修的模型缺陷

1. `package://` URI —— Isaac Sim 无法解析,需映射到本地路径或改写为相对路径。
2. 臂关节 `effort=0/velocity=0` —— 力矩与速度上限改由 `ArticulationCfg` 的 actuator 提供。
3. 连续关节无 `<limit>` —— 需显式给定,否则执行器配置异常。
4. 7 个 mimic 关节 —— 转换时**保留这些关节,由 PhysX mimic 约束驱动**
   (`convert_mimic_joints_to_normal_joints: true`)。它们**不是被压成 1 个关节**:
   7 个从动关节在 USD 里依然存在,只是挂上 `PhysxMimicJointAPI`,因此最终是 **22 个关节**
   (§7 阶段 0 已实测验证)。夹爪本次不进动作空间。
5. 23 个纯 TF frame link 无 inertial —— 转换时会被压掉,需确认树结构不受影响。
6. 网格过密(最大单件 30 MB / 63.7 万面)—— **已执行减面**(§5.2):83 MB → 19 MB、172.8 万 → 38.1 万面。

### 2.2 CR10 关节限位:两份文件不一致,已定以本模型为准(2026-09-23)

| 关节 | `agx/TCP-IP-ROS-6AXis/.../cr10_robot.urdf` | `rangercr10lidar.urdf`(**采用**) |
|---|---|---|
| j1 | ±3.14 | **-3.92 … 0.94** ✅ |
| j2 | ±3.14 | **±1.57** ✅ |
| j3 | ±2.861 | ±2.86 ✅ |
| j4 | ±3.14 | ±3.14 ✅ |
| j5 | ±3.14 | ±3.14 ✅ |
| j6 | ±6.28 | **±3.14** ✅ |

**决定:以 `rangercr10lidar.urdf` 的限位为准**(用户确认,2026-09-23)。`ArticulationCfg`、
初始姿态、命令范围与 MuJoCo 模型全部按这套限位实现,不再参考 `cr10_robot.urdf` 的数值。

> 影响面:`UniformPoseWBCCommandCfg` 的 `pos_x/pos_y/pos_z` 与 `roll/pitch/yaw` 可达范围要按
> 这套限位反推(尤其 j2 只有 ±90°,肘部折叠范围比 GO2-PIPER 的 Piper 小),阶段 3 调命令范围时
> 以本表为准。

---

## 3. 与 GO2-PIPER 架构的映射

框架的**平台相关代码很薄**(每平台约 250 行),共享 `mdp/` 与核心 env cfg 可复用。
新增平台需要:

| 新增/修改 | 参照 | 说明 |
|---|---|---|
| `assets/ranger_cr10/`(转换专用 URDF + `config.yaml` + USD 产物) | `assets/go2_piper/` | 转换所需的一切都收在这里,**description 包不动**(§7 阶段 0) |
| `assets/ranger_cr10/ranger_cr10_articulation_cfg.py` | 同目录 | 执行器分组、初始角、USD 路径 |
| `config/ranger_cr10/__init__.py` | `config/go2_piper/__init__.py` | 4 个 `gym.register` |
| `config/ranger_cr10/flat_env_cfg.py` / `wbc_env_cfg.py` | 同目录 | EE 目标范围、奖励权重、EE body 名、**底盘速度动作项与底盘控制器**、`decimation = 20`(§3.3 / §4) |
| `config/ranger_cr10/agents/rsl_rl_ppo_cfg.py` | 同目录 | PPO 超参 + `experiment_name` |
| `mujoco/robots/ranger_cr10/*.xml` | `mujoco/robots/go2_piper/` | MJCF(**需新写**) |
| `mujoco/deploy/deploy_mujoco/ranger_cr10/` | 同目录 | 部署脚本(**需重写**,见 §7 阶段 4) |

注册是自动发现的(`tasks/__init__.py` 用 `import_packages` 遍历 config 子包),
验收标志是 `python scripts/list_envs.py` 行数 **28 → 32**。

### 3.1 平台配置里必须关掉的四足专属项

核心 `leggedmanip_lab_env_cfg.py` 是按四足写的。GO2-PIPER 用 `disable_zero_weight_rewards()`
只能关 weight=0 的,以下需显式置 `None`:

- 奖励:`feet_slide` / `feet_air_time` / `feet_long_air` / `air_time_variance` /
  `hip_torques_max` / `thigh_torques_max` / `calf_torques_max` /
  `hip_deviation` / `joint_deviation` / `joint_mirror`
- 观测:`observations.critic.feet_contact`(`.*_foot` 匹配不到)
- 事件:`events.base_com`、`events.base_external_force_torque`(body 名 `base` 不存在 → 改 `base_link`)
- 终止:`terminations.base_contact`(body 名 `base` 不存在 → 改 `base_link` 或置 None)
- 动作:`actions.joint_pos.joint_names`(写死了 12 个腿关节名)→ 新平台只列 **6 个臂关节**,
  并**新增一个底盘速度动作项**输出 `(vx, wz)`(§4)

### 3.2 ⚠️ 共享代码里硬编码的 body/joint 名

| 位置 | 硬编码 | 影响 |
|---|---|---|
| `mdp/observations.py:117-123` `joint_pos_rel` | 默认参数是 12 个腿关节 + `joint.*` | **新平台观测项必须显式传 `joint_names`**,否则匹配为空 → 观测维度 0,静默出错 |
| `mdp/observations.py:143-149` `joint_vel_rel` | 同上 | 同上 |
| `mdp/observations.py:82-83` `end_effector_link0_relative_pose` | 硬编码 `end_effector` 与 `link0` | **找不到时静默返回全 0**(85-86 行有保护)→ critic 里混入 7 维常量,训练照跑但信息是假的 |
| `mdp/rewards.py:54` `position_command_b_error_exp` | 硬编码 `find_bodies("link0")` | Flat 模式用;需要时加 `link0_name` 参数(1 行) |
| `mdp/rewards.py:71` `position_command_error_exp` | **已有参数** `link0_name="link0"` | WBC 模式可通过 `params` 覆盖,**无需改共享代码 ✅** |

**结论**:WBC 路径基本"只加不改";但两条静默失败通道必须在阶段 1 用**打印实际动作与观测维度**证伪
(AGENTS.md 已明确:exit 0 不算成功)。

### 3.3 动作空间的实际构成(8 维)

| 动作项 | 维度 | 作用对象 | 说明 |
|---|---|---|---|
| `base_vel`(需新写) | 2 | 底盘控制器 | `[vx, wz]`,经底盘控制器解算成 4 转向角 + 4 轮速(§4) |
| `joint_pos` | 6 | `cr10_joint1..6` | 位置目标,`scale=0.25`、`use_default_offset=True`(对齐 GO2-PIPER) |

`observations.policy.actions`(`mdp.last_action`)按动作项声明顺序拼接 = **8 维**。
这是观测维度的关键一项:旧的 18 维假设随动作空间一起失效(§4.3),
阶段 1 打印维度时必须核到 **8**。

---

## 4. 核心决策:底盘怎么进动作空间(2026-09-23 修订为联合 WBC)

原以为是纯设计选择,但实机接口把选项砍掉了一大半。已核实的事实:

> **AgileX Ranger 底盘的唯一输入是 `/cmd_vel`(`geometry_msgs/Twist`)**,驱动内部
> 依 `(vx, vy, wz)` 自动切换运动模式并解算 4 个转向角与 4 个轮速。**没有任何关节级指令接口。**

| 模式 | 触发条件 | 行为 |
|---|---|---|
| `DUAL_ACKERMAN` | 默认 | `radius=|vx|/|wz|`,转向限幅 ±0.6981 rad |
| `PARALLEL`(蟹行) | `vy != 0` | 转向角 = `atan(vy/vx)`,速度 = `sqrt(vx²+vy²)`,限幅 ±1.570 rad |
| `SPINNING` | 转弯半径 < `min_turn_radius`(0.8103 m) | 原地转,`wz` 限幅 ±0.7853 rad/s |

底盘能力上限:`max_linear_speed=2.7 m/s`、`max_angular_speed=0.7853 rad/s`。

**因此各方案的可行性:**

- **方案 A —— 策略输出 8 个底盘关节:🚫 不可行。**
  实机不接受关节级底盘指令,这条路的策略无法上机。已排除。
- **方案 B —— 底盘速度作外部指令,RL 只控 6 个臂关节:保留为 baseline,不再是最终方案。**
  即忠实移植 GO2-PIPER:`(vx, vy, wz)` 是输入 command,动作 = 6 个臂关节。
  链路最短、最稳,阶段 0–2 仍按它验证底盘控制器;但它学不到"用底盘辅助够取",
  能力上限被机械臂的静态工作空间锁死。
- **方案 C′ —— 底盘 + CR10 联合 WBC(✅ 采用,2026-09-23 修订):**
  策略输出 `[vx, wz, 6×臂关节]`,底盘速度由策略自主决定而非外部指令 ——
  即方案 C 去掉 `vy` 的形态,并保留 `/cmd_vel` 语义以便上机。

**决定(2026-09-23 修订):采用方案 C′,方案 B 降级为 baseline。** 具体:

| 子项 | 决定 |
|---|---|
| **策略动作** | `[vx, wz, cr10_joint1..6]`,共 **8 维**;`vx ∈ [-0.5, 0.5] m/s`、`wz ∈ [-0.5, 0.5] rad/s` |
| 底盘关节 | **不进动作空间**;4 个转向角与 4 个轮速由底盘控制器解算 |
| 仿真底盘链路 | RL 输出 `(vx, wz)` → **Ranger 底盘控制器** → 4 个转向角 + 4 个轮速 |
| 实机底盘链路 | RL 输出 `(vx, wz)` → **`/cmd_vel`** → Ranger 驱动(驱动内部自己做 4WS 解算) |
| 语义一致性 | 两条链路对策略**完全等价**:策略只认 `(vx, wz)`,不关心底层如何解算 |
| **`vy` 蟹行** | **不加入**,避开 `PARALLEL` 模式的切换跳变 |
| 任务定义 | 给定末端目标位姿,RL **自主协调底盘与机械臂**完成跟踪(不再有速度跟踪目标) |
| 奖励 | **删除** `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`;保留 EE 位置/姿态跟踪、臂平滑与限位项;**新增底盘速度惩罚与底盘 action rate 惩罚**(防高速、防抖动) |
| 观测 | `last_action` 变为 **8 维**;底盘 4 转向角与 4 轮速**不进 policy observation**(底层轮系由 Ranger 控制器负责) |

底盘速度范围的依据:实机自己的导航栈用的是 `speed_lim_v = 0.5`、`speed_lim_w = 0.5`
(`ranger_bringup/param/velocity_smoother.yaml`),且硬件上限 2.7 m/s / 0.785 rad/s
远超本任务所需。训稳后可再往上放。

### 4.1 底盘控制器的职责(仿真侧)

本期仿真底盘控制器**复刻与 `(vx, wz)` 相关的 `DUAL_ACKERMAN` 与 `SPINNING` 运动学逻辑、速度与转角限幅**;
**不实现 `vy` / `PARALLEL`**;**暂不模拟转向伺服延迟、轮胎侧滑与其他动力学细节**(§5 B5)。

1. 默认 `DUAL_ACKERMAN`:`radius = |vx| / |wz|`,转向角由 `atan(wheelbase / 2 / radius)` 得,
   限幅 **±0.6981 rad**(与驱动一致);
2. 转弯半径 < `min_turn_radius = 0.8103 m` 时切 `SPINNING`:`vx → 0` 原地转,`wz` 限幅 ±0.7853 rad/s;
3. `vx = 0` 且 `wz ≠ 0` 同样走 `SPINNING`;
4. **四轮转向角与轮速按 Ranger 的实际运动学关系计算**,`(vx, wz)` 到 4 个转向角与 4 个轮速的
   **具体分配方式在阶段 2 用实机 `/actuator_state`(或驱动输出)逐项核对**,这一步不写死。

策略每 **0.1 s** 给出一次 `(vx, wz)`(§6.5),控制器在中间的物理步**持续执行最近一次指令**。

### 4.2 EE 目标:内部固定在世界系,观测给 base 相对量(✅ 已定,2026-09-23)

**问题**:现在 `UniformPoseWBCCommandCfg` 的 XY 是**机身(`link0`)系**、Z 是世界系
(`docs/WBC_MIXED_FRAME.md`)。在本方案下这不成立 —— 机身系 XY 目标不因底盘平移而改变:
底盘往前开,末端**相对机身**的位置一点没变。于是"必须移动底盘才能到达的目标"**永远不可能出现**,
RL 学不到用底盘辅助够取,§1 目标 2 直接落空。(机身系下只有 **yaw 旋转**能间接改变末端位置。)

**决定(2026-09-23)**:新写 `UniformPoseWorldCommand`,**只在 `ranger_cr10` 使用**,
**不修改**现有 GO2-PIPER 的 `UniformPoseWBCCommand`。

#### 4.2.1 三层坐标关系(不要把这三者混为一谈)

| 层 | 表示 | 说明 |
|---|---|---|
| **任务目标(内部真值)** | `target_pose_w` | **固定在世界系**,本次 command 生命周期内**不随底盘移动** |
| **Policy 观测** | `target_pose_b = T_base_world × target_pose_w` | 每步由当前 `base_link` 位姿换算,给策略**相对量**,避免让网络去学与 `env_origin` 相关的全局绝对坐标 |
| **Reward** | EE 世界位姿 vs `target_pose_w` | 误差在**世界系**计算,与目标定义一致(§4.2.2) |

```
世界系固定目标 target_pose_w
        ↓  按当前 base_link 位姿转换
base-relative target observation target_pose_b
        ↓
      Policy
        ↓
   [vx, wz, 6×CR10]

底盘向目标移动 → target_pose_b 里的距离逐渐减小
              → 策略能感知到"底盘移动正在帮机械臂接近目标"
```

这样底盘移动**真的**缩短了末端与目标的距离(在世界系里),而策略看到的始终是相对量。
**不要把世界绝对坐标直接作为 Policy 的主要目标观测。**

#### 4.2.2 Reward 同样用世界系误差

末端跟踪奖励按 **世界系 EE 位姿 vs `target_pose_w`** 分别计算 position error 与 orientation error。
**不允许出现"command 是世界系、reward 却按 base/link0 系计算"的错配** —— 那会让训练往错方向收敛。

若现有 `position_command_error_exp`(`rewards.py:66-98`,XY 按 link0 系)无法直接支持世界系目标,
**就为 `ranger_cr10` 单独新增对应 reward function,不改 GO2-PIPER 的默认行为。**

#### 4.2.3 世界目标的采样方式

多环境训练时**不要直接随机全局绝对坐标**。每次 resample 时:

```
target_pose_w = base_pose_w(采样时刻) ⊕ 采样的局部目标 offset
```

即"**以采样时刻的底盘位姿为基准,采样一个局部 offset,再固化到世界系**",此后本次 command
生命周期内不再随底盘移动。这样天然兼容:

- 各并行环境的 `env_origin` 与机器人初始位置差异(目标始终落在环境内的合法位置);
- 课程学习逐步扩大目标距离(扩大的是 offset 的采样范围)。

**第 ③ 类课程目标(必须移动底盘才能到达)必须是真正的世界系固定目标**,
**不能**再用"base-relative 固定目标"生成 —— 后者底盘一开就跟着走,永远够不着(§7 阶段 3)。

### 4.3 连带修改(删掉速度指令后必须同步处理的项)

`(vx, wz)` 从 command 变成 action 之后,原来围绕"底盘速度指令"搭的几项都要一起动;
漏一个就会得到自相矛盾的配置,或直接启动报错:

| 位置 | 现状 | 改为 |
|---|---|---|
| `commands.base_velocity` | `UniformVelocityCommandCfg`,WBC 用它给出速度指令 | **删除**(不再有速度指令) |
| `observations.*.velocity_commands` | `mdp.generated_commands(command_name="base_velocity")` | **删除** —— 引用已不存在的命令会在启动时报错 |
| `curriculum.lin_vel_cmd_levels` / `ang_vel_cmd_levels` | 按训练进度放宽速度指令范围 | **删除**;由目标范围课程替代(§7 阶段 3) |
| `curriculum.pos_cmd_levels` | 放宽末端位姿指令范围 | **保留并扩展** —— 这就是"逐步加入够不到的目标"的落点(§1 目标 2) |
| `rewards.track_lin_vel_xy_exp` / `track_ang_vel_z_exp` | 速度跟踪奖励 | **删除**(§4 决定) |
| `observations.*.actions` | 维度随动作项 | 自动变为 **8 维**(§3.3) |

策略观测维度因此从 GO2-PIPER 的 `3 × 70 = 210` 变为 `3 × (3 + 3 + N + N + 8 + 7)`,
N = 6 个臂关节 → **3 × 33 = 99**(以阶段 1 打印出来的实测值为准,不要照抄这个数)。

> 补充:`Sim2real.md` / `Sim2real_cursor.md` 我核实过,**是文献调研,不是本机器人的操作流程或部署脚本**,
> 且没有引用任何 checkpoint。计划里不把它们当作既有工作。

---

## 5. 已决定的决策点(全部经用户确认,2026-09-23)

| 层 | 事项 | 决定 |
|---|---|---|
| A1 | CR10 关节限位 | 以 **`rangercr10lidar.urdf`** 为准(§2.2) |
| A2 | 末端执行器帧 | **夹爪指尖中心**,且以**虚拟 TCP**方式实现:在 observation/reward 里对真实连杆 `gripper_base_link` 施加固定变换算出该点,**不往 USD 里加 frame**(理由与静默失败风险见 §7 阶段 1) |
| B1 | 底盘方案 | **方案 C′ —— 底盘 + CR10 联合 WBC**(**2026-09-23 修订**,原为方案 B)。动作 = `[vx, wz, 6×臂]`,底盘速度由 RL 自主决定;底盘关节不入动作空间;**不加 `vy` 蟹行**;方案 B 保留为 baseline(§4) |
| B2 | 底盘状态进观测 | **不进 policy observation**(2026-09-23 随 B1 修订而定):底层轮系由 Ranger 控制器负责。原 (a)/(b) 选项作废,§9 R2 关闭 |
| B3 | 无 IMU 的替代 | 走 **(a) 从 `/odom` 推算**,并在仿真里加入**噪声/漂移随机化** |
| B4 | 控制频率 | **统一策略频率 10 Hz**:物理 200 Hz、策略 10 Hz(`decimation = 20`),一次输出 `[vx, wz, 6×臂]`;底盘控制器在中间的物理步**持续执行最近一次 `(vx, wz)`**(§6.5) |
| B6 | 奖励调整 | **删除** `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`(不再有速度跟踪目标);**新增底盘速度惩罚与底盘 action rate 惩罚**(防高速、防抖动);保留 EE 跟踪与臂平滑/限位项(§4) |
| B7 | EE 目标的参考系 | ✅ **已定(2026-09-23)**:新写 `UniformPoseWorldCommand`,只在 `ranger_cr10` 使用,**不改 GO2-PIPER 的 `UniformPoseWBCCommand`**;**目标内部固定在世界系**、**Policy 观测用 base-relative 量**、**Reward 用世界系误差**(§4.2) |
| B5 | 底盘控制器保真度 | 本期**复刻与 `(vx, wz)` 相关的 `DUAL_ACKERMAN` + `SPINNING` 运动学与限幅**;**不实现 `vy`/`PARALLEL`**;**暂不模拟转向伺服延迟、轮胎侧滑与其他动力学细节**(§4.1)。上机前再评估是否加深保真度 |
| C1 | 网格精简 | **改为执行减面**(2026-09-23 修订,原为"先不动"),见 §5.2 |
| C2 | 两份 description 副本 | **(a) 保留两份,训练时以仓库根目录那份为准**;需在文档写明,避免改错文件 |
| C3 | 车轮摩擦辨识 | **(a) 先接受差距,靠域随机化覆盖** |
| — | 网格入库 | 以**普通 Git 对象**入库,不走 LFS(§5.1) |
| — | `agx/` 归属 | 做成 **submodule**,固定 `28beddb` |

### 5.1 网格入库的落地方式

```
.gitattributes
  rangerboxcr10lidar_description/meshes/** -filter -diff -merge -text
  ↑ 放在 *.stl / *.STL 规则之后(后面的规则优先);MuJoCo 资产仍走 LFS

.pre-commit-config.yaml
  check-added-large-files: exclude: ^rangerboxcr10lidar_description/meshes/
```

已用 `git cat-file -s` 逐个核对暂存 blob 是真实字节数(31,841,184 B)而非 130 B 的 LFS 指针。

### 5.2 网格减面(2026-09-23 执行,修订 C1)

**背景**:原始网格 27 个 STL 共 **83 MB / 172.8 万面**,其中 14 个被 URDF 引用且 >2 万面。
对照 GO2-PIPER 的 Isaac 模型 31 MB(约 35 万面),本模型字节数约 1.8 倍、几何密度约 5 倍;
最大单件 `ranger_base_link.STL`(30.37 MB / 63.7 万面)一个就相当于整个 GO2-PIPER 模型。

**方法**:按**连通分量各自保留 15% 面数**(quadric collapse,`fast_simplification`)。

> 为什么不用全局面预算:该网格是 **495 个独立连通分量**组成的 CAD 装配体(最大一块仅占 9%),
> 而 quadric 误差是绝对量纲,全局预算会把小组件整体删掉 —— 实测有零件直接消失、反向误差达 25 cm。
> 按分量各自按比例分配后该问题消失。

**结果**

| | 减面前 | 减面后 |
|---|---|---|
| 网格总体积 | 83 MB | **19 MB** |
| 总面数 | 1,727,785 | **381,437**(22%) |
| 改动文件 | — | 14 个(URDF 引用且 >2 万面) |
| 未改动 | — | 13 个(CR10 臂、AG95 夹爪等小网格,原样保留) |

**质量**(逐文件测"新顶点到原网格最近顶点的距离"):中位 **0.04 mm**、p99.9 约 1.5 cm;
底盘 `ranger_base_link` 偏离 >1 cm 的顶点占 **0.206%**(106 / 51,551),最大 4.7 cm;
其余文件极值 6–13 mm。该极值与面数预算**无关**(底盘按 40% 重做,极值纹丝不动),
来自少量退化/细小特征,不是减面引入的。

**代价(必须明说)**:这是有损压缩。常规观察距离下不可见,但**高分辨率特写下会有差异**;
判据是绝对误差 1 cm 量级,不是"无损"。原始网格可从 git 历史或 `agx/` 子模块取回。

**库存变化**:减面**就地改写了仓库根目录那份 description**,`agx/` 子模块里那份保持原始,
两份从此不再逐字节相同 —— §5 C2 的"以根目录为准"现在同时意味着"以减面版为准"。

仍未定的见 §9。

---

## 6. 实机接口:已核实的事实与缺口

> 这一节是后续部署与"接口对齐"的依据,全部来自 agx 仓库实读。

### 6.1 控制接口

| 部件 | 接口 | 单位/范围 | 频率 |
|---|---|---|---|
| 底盘 | `/cmd_vel` `geometry_msgs/Twist` | m/s, rad/s;只用 `linear.x/y`、`angular.z` | 50 Hz |
| 臂 | `/cr10_robot/joint_controller/follow_joint_trajectory`(MoveIt 用) | **rad** | 控制频率上限 **33 Hz**,实测可达 20 Hz,当前默认实机参数 **10 Hz** |
| 臂(备选) | `/dobot_v4_bringup/srv/ServoJ` 服务 | **度(deg)**,一次性 | 单次调用 |
| 夹爪 | `/gripper/ctrl` `dh_gripper_msgs/GripperCtrl` | **position 0–1000 原始计数,force/speed 百分比** | 50 Hz |

> 频率数据的口径(用户确认 2026-09-23):控制器 **30004 端口**输出 1440 字节二进制 RealTimeData,
> 周期 **8 ms = 125 Hz**,这是**状态反馈**的上限。**控制**频率上限 33 Hz(实测 20 Hz),
> 当前默认实机参数为 10 Hz。本计划决定训练与部署都按 **10 Hz** 对齐(§5 B4)。
> 先前记录里"`servoj` 2.5 Hz"指的是 `cr5_v4_robot.cpp` 里 MoveIt action 内部 `SERVOJ_DURATION=0.4`
> 的分段流式实现,不是控制器能力上限,不作为设计依据。

⚠️ `ServoJ` 服务是**度**,而 trajectory action 是**弧度**,内部帮你换算 —— 混用是 57.3 倍误差。
⚠️ `JointMovJ.srv` 在 v4 包里**不存在**,但 `agx/CLAUDE.md` 和 `README.md` 都让人调它 —— 文档是错的。

### 6.2 状态反馈

`/odom`(50 Hz)、`/system_state`、`/motion_state`、`/actuator_state`、`/battery_state`、
`/joint_states`(臂,**10 Hz**)、`/gripper/joint_states`(50 Hz)、`/dobot_v4_bringup/msg/ToolVectorActual`(TCP 位姿,10 Hz)。

- ⚠️ **全仓库没有任何 IMU 话题**(`agx/debugs/bug_ranger.md` 自己也在问"IMU 的 launch 在哪")。
  而框架的观测里有 `base_ang_vel` 与 `projected_gravity`(通常来自 IMU)→
  **实机侧只能从 `/odom` 推算**。已定:走这条路,并在仿真里加入噪声/漂移随机化(§5 B3)。
- ✅ `/actuator_state` 的填充循环原先有混叠 bug(8 个执行器返回同一份数值),
  **用户确认已修复,`/cmd_vel` 现在可以正常驱动底盘**(2026-09-23)。本节此前记录的该 bug 作废。

### 6.3 关节命名/顺序(部署时最要命的一环)

- 臂:驱动 `/joint_states` 发的是 **`joint1..joint6`**,而整合 URDF 里叫 **`cr10_joint1..cr10_joint6`**
  → 名字不匹配,`robot_state_publisher` 不动。
- 线协议顺序:`ServoJ(a,b,c,d,e,f)` 其中 a=j1 … f=j6,**单位度**。
- 底盘:`fr / fl / rl / rr` 各 `(steering, wheel)`,注意 `fr_steering_joint` 少一个 `wheel` 后缀。
- 夹爪:驱动把原始计数 `0–1000` 线性映射到 `gripper_finger1_joint` 的 `0–0.637 rad`,**且是反向的**
  (`msg.position[0] = (1000-raw)/1000.0 * 0.637`)。

### 6.4 ⚠️ 部署前必须补的安全项(不影响训练,影响上机安全)

**驱动里没有 `/cmd_vel` 看门狗** —— 一旦停止发布,最后一条速度指令会**永久保持**。
部署节点的循环必须自己实现超时归零。

### 6.5 控制频率(2026-09-23 修订:物理 200 Hz / 策略 10 Hz)

底盘与机械臂现在**同属一个 policy action**,因此不再拆成两个 RL 频率,统一按策略频率更新。

| 回路 | 实机能力 | 本计划采用 |
|---|---|---|
| 物理仿真 | — | **200 Hz**(`sim.dt = 0.005`,核心配置既有值,不改) |
| **策略** | — | **10 Hz**(`decimation = 20`) |
| 底盘 `/cmd_vel` | 50 Hz | 驱动侧 50 Hz;`(vx, wz)` 由策略 10 Hz 给出,控制器/驱动在中间的步**持续执行最近一次指令** |
| 臂状态反馈 | **125 Hz**(30004 端口,8 ms) | 10 Hz |
| 臂控制 | 上限 33 Hz,实测 20 Hz,默认 10 Hz | **10 Hz**(与实机默认参数一致) |

**决定**:一次策略推理同时输出 `[vx, wz, 6×臂]`;臂按 10 Hz 执行,
底盘在两个策略步之间**零阶保持**。仿真与实机的动作频率由此一致,**不需要插值层**。

**实现方式**(阶段 1/3 落地):

- 核心配置是 `decimation = 4`(→ 50 Hz);新平台在 `wbc_env_cfg.py` 里覆盖为 **`decimation = 20`**;
- 动作项仍是两个(`base_vel` 2 维 + `joint_pos` 6 维),二者在**同一个 env step** 更新,
  即同为 10 Hz;`last_action` 按动作项顺序拼成 8 维(§3.3);
- 底盘控制器**不额外降频**:它在每个**物理步**都按最近一次 `(vx, wz)` 解算 4 转向角与 4 轮速。

> 与旧方案的区别:旧方案是"底盘 50 Hz、臂 10 Hz 两套 action";新方案**只有一个 10 Hz 的 policy**,
> 底盘不再拥有自己的 RL 频率。这是本计划被修订掉的重点之一。

---

## 7. 实施步骤(每阶段有可验证产物)

> 验收遵循 AGENTS.md:**不认 exit 0,只认正证据**(行数/维度、生成文件、指标)。

### 阶段 0 — URDF→USD 转换(风险最高,先做最小验证)

**资产目录边界(2026-09-23 定)**:为转换而做的一切改动都落在 `assets/ranger_cr10/`,
**description 包本身不再动** —— 它已经因为减面(§5.2)与 `agx/` 里那份分叉了,
不再叠加新的改动来源。

1. 建 `assets/ranger_cr10/`,内含两样东西:
   - **一份转换专用 URDF 副本**。唯一改动:把
     `package://rangerboxcr10lidar_description/...` 改写为指向 description 包的相对路径。
     其余内容与原 URDF **逐字一致**,便于日后 diff 确认"改动只有路径"。
   - `config.yaml`,照抄 GO2-PIPER 的字段(`fix_base: false`、`merge_fixed_joints: true`、
     `collider_type: convex_hull`),但 **`convert_mimic_joints_to_normal_joints` 取 `true`**
     —— 与 go2_piper 相反,理由见 §2.1 与下面阶段 0 的实测结论。
2. 用 Isaac Lab 自带工具转换:`IsaacLab5/scripts/tools/convert_urdf.py`。
   网格用**减面后**的版本(§5.2),转换时不要再动网格。
3. **验收(必须打印证据,不能只看 exit code)**:
   - 打印 USD 的关节表 —— 关节数、名称、类型、limit;
   - 确认底盘 8(4 转向 + 4 驱动)+ 臂 6 全部在列,根 link 是 `base_link`;
   - mimic 关节按预期处理 —— **7 个从动关节保留在关节表里并挂上 `PhysxMimicJointAPI`**,
     不是"压成 1 个"(需在 USD 里核对 `PhysxMimicJointAPI` 确实存在);
   - **臂关节 limit 逐项等于 `rangercr10lidar.urdf` 的值**(§2.2 那张表),
     任何一项不一致都说明转换没做完。

#### ✅ 阶段 0 已完成(2026-09-23)

产物落在 `assets/ranger_cr10/`:`ranger_cr10.usd`(1.5 KB 包装)+ `configuration/`(网格)+
`ranger_cr10_converted.urdf` + `config.yaml`,共 **22 MB**。转换驱动脚本
`assets/ranger_cr10/convert_ranger_cr10.py`(Isaac Lab 自带的 `convert_urdf.py` 只暴露部分开关,
所以这里直接调 `UrdfConverter` API,读同目录的 `config.yaml`)。

**验收结果**:打印出 22 个关节,父 link 归属正确(base_link → 4 转向 + `cr10_joint1`;
steering_link → wheel;`cr10_Link1..5` → `cr10_joint2..6`;`cr10_Link6` → 夹爪)。
臂关节限位逐项吻合(USD 的单位是**度**):
`-224.59944 … 53.85803`(URDF -3.92…0.94 rad)、`±89.95437`(±1.57)、`±163.86592`(±2.86)、
`±179.90874`(±3.14)。

**过程中确认的四件事(与 go2_piper 不同,已写进配置)**:

1. **`convert_mimic_joints_to_normal_joints` 必须是 `true`**(go2_piper 是 `false`)。
   原因:AG95 有 7 个 mimic 关节,`false` 时它们会变成**互相独立的自由铰链**,
   重力下会散架;`true` 时导入器建立 `PhysxMimicJointAPI:rotY`(带 gearing/offset),
   实测 7 个从动关节全部挂上该 API。go2_piper 用 `false` 是因为它的 URDF 里
   **根本没有夹爪** —— 它的 USD 只有 18 个关节(12 腿 + 6 臂),我们的有 22 个。
2. **USD 里的关节限位是「度」**,go2_piper 的 USD 也一样(±60.0001° = ±1.0472 rad)。
   这是本框架的正常状态,Isaac Lab 侧按弧度处理,不需要额外转换。
3. 导入器会**回写 `config.yaml`**(绝对路径 + `Generated by UrdfConverter` 页脚),
   所以写在里面的注释保不住 —— 转换参数的理由记在本计划里,不要只写在 yaml 注释里。
4. 根 prim 名是 `/rangercr10lidar`,`base_link` 在其下;`merge_fixed_joints: true`
   已把 29 个固定关节(含 23 个无惯量的 TF frame link)压掉。

### 阶段 1 — 接入与注册

4. 写 `ranger_cr10_articulation_cfg.py`:
   - 关节限位**严格取自 `rangercr10lidar.urdf`**(§2.2);
   - `effort_limit` 按 CR10 规格填(URDF 里是 0);`stiffness`/`damping` 起步值参照 GO2-PIPER 的 Piper;
   - 初始角:转向归零,臂取肘部朝前的姿态(按 j2 只有 ±90° 的实际可达区间选,别照抄 Piper)。
5. 写 `config/ranger_cr10/` 四件套 + agents:
   - **末端 TCP 用"虚拟 TCP"实现,不要往 USD 里加 frame**:

     ```
     cr10_Link6(真实 articulation rigid body,CR10 法兰)
             ↓ 固定 TCP offset(两指尖中点相对它的常值变换)
     virtual fingertip-center TCP
     ```

     ⚠️ **锚点是 `cr10_Link6`,不是 `gripper_base_link`**:后者在转换时被
     `merge_fixed_joints: true` 合并进了 `cr10_Link6`,**在 USD 里根本不存在**
     (实测 23 个 body 里没有它)。这是本计划早先版本的错误假设。

     **为什么不能靠新加一个 frame**:
     `Articulation.find_bodies()` 是拿名字去匹配 `self.body_names`,而它是
     `root_physx_view.shared_metatype.link_names` —— **PhysX 的 articulation 连杆名**。
     新加的普通 Xform 不是 articulation 连杆,不在这个列表里;叠加 `merge_fixed_joints: true`
     会合并固定结构,这条路更容易踩空。而 `position_command_error_exp` / `end_effector_link0_relative_pose`
     都是按 articulation body 查的 —— 查不到时 `end_effector_link0_relative_pose` **静默返回全 0**
     (`observations.py:85-86`),训练照跑但信息是假的(§3.2),很难发现。

     **实测偏移**(闭合夹爪、臂稳定姿态下测):`TCP_OFFSET_POS = (0.0, 0.00003, 0.14389)`,
     模长 **0.1439 m**,方向几乎就在 Link6 的 z 轴上。写法见 `mdp/tcp.py`。

     该 offset 在**夹爪不动作、保持固定开合度**时是常值 —— 夹爪本次不进动作空间,所以成立。
     若将来要把夹爪纳入动作空间,这个"固定"变换就要改成随指关节角变化。

     **除非实测确认** `find_bodies()` 能返回新加的 frame,否则不要走加 frame 的路。

     **受影响的两个函数**:reward 的 `position_command_error_exp` 与 critic 观测的
     `end_effector_link0_relative_pose` 都是按 articulation body 取末端位姿的 ——
     需要为 `ranger_cr10` 写 TCP 版本(对 `gripper_base_link` 施加 offset),
     或在共享实现里加 `tcp_offset` 参数并**默认不启用**。**不改 GO2-PIPER 的默认行为。**
   - **动作空间按 §3.3 组装**:`base_vel`(2 维,新写的动作项)+ `joint_pos`(6 个臂关节);
     `base_vel` 输出范围即 `vx ∈ [-0.5, 0.5]`、`wz ∈ [-0.5, 0.5]`,**`vy` 恒为 0**;
   - **底盘控制器**(§4.1)先按纯运动学实现,由 `base_vel` 动作项在**每个物理步**驱动;
   - **`decimation = 20`**(§6.5,策略 10 Hz);
   - 按 §4.3 删掉 `commands.base_velocity`、`velocity_commands` 观测项与两条速度课程项;
   - **命令项**:新写 `UniformPoseWorldCommand`(§4.2),目标在内部固定为世界系 `target_pose_w`,
     采样方式按 §4.2.3(采样时刻的底盘位姿 ⊕ 局部 offset,然后固化到世界系);
   - **观测**:给 Policy 的是 **base-relative 的 `target_pose_b`**(每步按当前 `base_link` 位姿换算),
     **不要把世界绝对坐标直接喂给策略**(§4.2.1);
   - **奖励**:按**世界系** EE 误差计算(§4.2.2);必要时为 `ranger_cr10` 单独新增 reward function,
     不改共享实现的默认行为;
   - `link0_name` 仍指向 `base_link`(§3.2),供其它需要机身系的项使用。
6. **验收**:`list_envs.py` 行数 **28 → 32**;`zero_agent.py --task RANGER-CR10-WBC` 不报错
   并**打印 policy 动作维度 = 8、观测维度 = 实测值**(预期 99,见 §4.3)。
   打印这两个数是用来证伪 §3.2 那两条静默失败通道的,不能跳过。

#### ⚠️ 阶段 1 实施中实测到的六件事(均与计划原文的假设不同)

1. **`gripper_base_link` 在 USD 里不存在** —— 被 `merge_fixed_joints` 合并进了 `cr10_Link6`
   (23 个 body 里没有它)。TCP 的锚点因此必须是 `cr10_Link6`,见上面的虚拟 TCP 说明。
2. **mimic 约束单独撑不住连杆**。导入器写的是 `naturalFrequency = 25`、`dampingRatio = 0.005`
   (几乎无阻尼),实测 7 个从动关节里有 2 个漂到 **0.72 rad** 和 **2.79 rad**(并行连杆闭环,
   约束求解器压不住)。两道措施一起上:
   - 把约束调硬到 `natFreq = 200 / dampingRatio = 1.0` —— **写进转换驱动的后处理**
     (`stiffen_mimic_constraints()`),否则重新转换就丢了;
   - 给 7 个从动关节加**保持驱动**(目标 = 0)。它们与 mimic 关系
     `q_i = gearing × 0 + 0 = 0` 完全一致,所以驱动和约束不会互相打架。
   改完最大残差 **0.0124 rad**。**mimic 关节是保留的,不是压成 1 个**(§2.1)。
3. **臂的执行器刚度不能照抄 go2_piper**。那套 50–80 是给轻得多的 Piper 臂的。PD 是弹簧:
   24.8 kg 的 CR10 在肩部约需 68 N·m,`kp = 80` 对应稳态误差 `68/80 ≈ 0.85 rad` ——
   实测关节直接顶到限位(偏差 0.97 rad)。提到 **1500–4000**(肩部最高)后残差降到 0.0072 rad。
4. **底盘高度**:几何推算 0.4114 m,实际轮胎受力后稳定在 **0.4071 m**,按后者生成初始位姿。
5. **臂的"正前方"对应 `cr10_joint1 ≈ 0.94`(它的上限)**,此时 TCP 在
   `(0.549, -0.001, 0.470) m`;而 `j1 = 0` 时 TCP 偏到右侧 24 cm。初始姿态取
   `j1 = 0.6`(TCP 约在 `(0.52, -0.07, 0.47)`),留出 `reset_joints_by_scale(0.5, 1.5)`
   不越过 j1 上限的余量。目标采样范围也据此以"臂的实际工作位置"为中心,而不是假设在 y = 0。
6. **`list_envs.py` 通过 32 行不能证明配置可用**:它只做 gym 注册,entry point 是字符串、
   懒解析,**配置类根本没被导入**。真正验证必须建环境——这也是为什么验收要求打印维度。

### 阶段 2 — 底盘单独跑通(先于联合训练)

7. 先把**底盘控制器**做完:RL 输出的 `(vx, wz)` 经 §4.1 的解算变成 4 个转向角 + 4 个轮速,
   驱动底盘直行、转向、原地旋转。**不做蟹行**(§4 不加 `vy`),但单独验证一次
   "`vx = 0, wz ≠ 0` 时确实走 `SPINNING` 分支"。
8. **验收**:令臂关节固定,给 `(vx, wz)` 一组台阶/斜坡指令,确认
   - `|vx| ≤ 0.5`、`|wz| ≤ 0.5` 全范围内底盘都能跟住;
   - **转向角限幅与实机驱动一致**(Ackermann ±0.6981 rad、原地转 `wz` 限幅 ±0.7853 rad/s);
   - 转弯半径 < 0.8103 m 时确实切到原地转,而不是继续 Ackermann。
   这一步是"仿真 `(vx, wz)` 语义 == 实机 `/cmd_vel` 语义"的证据(§4)。

### 阶段 3 — 底盘 + CR10 联合 WBC 训练

9. 打开臂动作与末端位姿奖励,奖励权重从 GO2-PIPER 的起步(位置 4.5 / 姿态 -4.0),按实测调。
   - **一个 10 Hz 的 policy 同时输出 `[vx, wz, 6×臂]`**(§6.5),不再有分开的两套频率;
   - 删除两条速度跟踪奖励,**新增底盘速度惩罚与底盘 action rate 惩罚**(§4);
     两者的权重不能大到让策略干脆不用底盘 —— 这正是 §8 里新增的两条风险;
   - 从 `/odom` 推算的 `base_ang_vel` / `projected_gravity` 带噪声与漂移随机化(§5 B3)。
10. **训练目标范围必须分层**(否则学不到协同,§4.2):让 `pos_cmd_levels` 课程按顺序扩大目标,
    依次覆盖
    ① CR10 静止底盘时可达的目标 → ② 工作空间边缘的目标 →
    ③ **必须移动底盘才能到达的目标**。
    其中 **③ 必须是真正的世界系固定目标**(§4.2.3):底盘开过去之后,该目标相对 `base_link`
    的距离会自然缩短并进入 CR10 可达范围。**不能再用 base-relative 固定目标生成第 ③ 类** ——
    那种目标会跟着底盘走,永远够不着。
11. **验收**:`Metrics/ee_pose/position_error` 与 `orientation_error` 收敛到与 GO2-PIPER 同量级
    (它训到 0.065 m / 0.095);**出图 + 视频确认策略真的在用底盘**:面对第 ③ 类目标时底盘发生
    位移,且这类目标的成功率显著高于"冻结底盘"的对照组(对照做法:把 `base_vel` 动作置零重跑
    `play.py`)。只看总误差无法区分"用了底盘"和"没用到但碰巧够着"。

### 阶段 4 — MuJoCo 部署
12. 建 MJCF,写 `deploy_mujoco/ranger_cr10/`。**GO2-PIPER 那份不能照抄**:它写死了 18 维观测
    和 12 个腿关节的 `ISAAC_TO_MUJOCO` 重排表。新脚本要:
    - 观测装配改为 **6 个臂关节 + 8 维动作**的布局(§3.3 / §4.3);
    - **策略输出的前两维 `(vx, wz)` 必须真的驱动底盘** —— MJCF 里用与 §4.1 同一套运动学解算,
      而不是像 GO2-PIPER 那样把底盘速度当成外部输入。
13. **验收**:`play.py` 导出 `policy.pt`;键盘遥操作时给一个远端目标,
    确认**底盘会动起来**去够 —— 这是联合方案在 MuJoCo 里的最小证据。

### 阶段 5 — 实机接口对齐(不上机)
14. 产出**"仿真 ↔ 实机"对照表**:每个观测分量/动作分量 → agx 的话题与字段,含单位、
    频率、命名映射。用 `--export_io_descriptors`(`mdp/observations.py` 的 `generic_io_descriptor`
    与 `record_joint_names`)导出的布局描述作为抓手。
15. **验收**:对照表与 agx 代码逐项核对通过;列出实机部署节点待办清单 ——
    **把策略前两维 `(vx, wz)` 发布到 `/cmd_vel`**(零阶保持)、
    关节名重映射(`joint1..6` ↔ `cr10_joint1..6`)、度/弧度换算、**臂指令 10 Hz**、
    `/cmd_vel` 看门狗(驱动无超时,停发即保持最后指令)、夹爪 0–1000 原始计数与反向映射、
    无 IMU 时用 `/odom` 推算姿态。

---

## 8. 风险清单

| 风险 | 状态 | 缓解 |
|---|---|---|
| ~~CR10 限位两份文件矛盾~~ | ✅ 已定(§2.2,以 `rangercr10lidar.urdf` 为准) | — |
| ~~控制频率~~ | ✅ 已定(§6.5:物理 200 Hz / **策略 10 Hz**,单一 8 维动作,底盘零阶保持) | — |
| ~~实机无 IMU~~ | ✅ 已定(§5 B3,`/odom` 推算 + 仿真噪声随机化) | — |
| ~~URDF 转换质量~~ | ✅ 已完成(§7 阶段 0,22 关节验收通过) | — |
| ~~底盘关节状态是否进观测~~ | ✅ 已定(§5 B2:不进) | — |
| ~~EE 目标参考系~~ | ✅ 已定(§4.2:新写 `UniformPoseWorldCommand`;目标固世界系、观测给 base 相对量、奖励按世界系误差) | — |
| ⚠️ **RL 完全不用底盘**,退化成"固定底盘 + 纯机械臂" | 联合方案特有风险 | ① 目标范围课程必须包含第 ③ 类(§7 阶段 3);② 底盘速度惩罚权重别大到压制底盘使用;③ 阶段 3 用"冻结底盘"对照组量化底盘的真实贡献 |
| ⚠️ **RL 过度依赖底盘**(靠开过去代替用臂,或对近目标也乱动底盘) | 联合方案特有风险 | ① 底盘速度惩罚 + 底盘 action rate 惩罚(§5 B6);② 检查近目标成功率不因冻结底盘而下降;③ 评估时三类目标的误差分开统计 |
| ⚠️ **`(vx, wz)` 高频抖动** | 新增 | action rate 惩罚;**10 Hz 策略 + 零阶保持**本身就抑制高频(§6.5);验收要看底盘速度的**时间序列**,不能只看均方误差 |
| ⚠️ **仿真底盘响应与实机 `/cmd_vel` 响应不一致** | 新增 | 仿真控制器按 §4.1 复刻驱动的分支与限幅;阶段 2 用同一组台阶/斜坡指令核对;转向伺服延迟与轮胎摩擦未辨识(§5 C3),先用域随机化覆盖 |
| ⚠️ `/odom` 推算角速度/姿态的精度 | 未知,需实测 | 阶段 5 前用真机跑一段直行+旋转,对比 IMU 或外部真值;仿真里按实测噪声标定 |
| ⚠️ 176 kg + 8 维动作 + 22 MB 资产 | 训练吞吐可能远低于 GO2-PIPER 的 8192 环境 | 阶段 2 先实测吞吐再定规模 |
| 静默失败(观测为空/全零) | 已知有两处通道(§3.2) | 阶段 1 强制打印**动作维度 8 与观测维度** |
| 车轮摩擦/侧滑未辨识(§5 C3 决定先不做) | 已知 sim-to-real 主要误差源 | 靠域随机化覆盖;上机后按实测回填 |
| 无 `/cmd_vel` 看门狗 | 未处理 | 部署节点自建超时归零(阶段 5 清单已列) |
| 关节名不匹配(driver `joint1..6` vs URDF `cr10_joint1..6`) | 未处理 | 阶段 5 映射表覆盖 |
| 末端帧 = 指尖中心(以**虚拟 TCP** 实现),与控制器 TCP 是否一致 | **未确认** | 见 §9 R1 —— 上机前必须核对,否则位姿指令系统性偏移 |
| 虚拟 TCP 的固定变换取值(两指尖中点相对 `gripper_base_link`) | 未计算 | 阶段 1 从 URDF 里按初始开合度算出并写成常量;写错会得到一个"差一点"的末端点,误差不在指标里体现 |

---


## 9. 剩余待定项(其余问题已全部锁定,见 §5)

### R1. 末端帧与控制器 TCP 是否一致?(上机前必须核对)

已定仿真里的末端点 = **夹爪指尖中心**,且以**虚拟 TCP** 方式实现(对 `gripper_base_link` 施加固定
变换,不往 USD 加 frame,§5 A2 / §7 阶段 1)。但这只在仿真侧成立 ——
实机 `ToolVectorActual` 报的是 **Dobot 控制器里配置的工具坐标系(TCP)**。

- 如果控制器 TCP 也配在指尖中心 → 两侧一致,直接可用;
- 如果配在法兰或其他位置 → 部署节点必须做一次固定偏移变换,否则位姿指令系统性偏一个常数。

**需要你确认或实测**:控制器当前的 TCP 偏移(Tx/Ty/Tz)。
不确认也能先训(仿真内部自洽),但阶段 5 的对齐表必须解决它。

### ~~R2. 底盘关节状态要不要进观测?~~ ✅ 已关闭(2026-09-23)

随 §4 的方案修订一并确定:**不进 policy observation** —— 底盘由 Ranger 控制器解算,
策略只认 `(vx, wz)`(§5 B2)。原 (a)/(b) 两种选项作废。

### R3. "训练效果好"的判据

沿用 GO2-PIPER 的指标(末端位置/姿态误差收敛到 0.065 m / 0.095 量级)+ 人工看视频。
但**联合方案下这还不够**:总误差好,分不清"用了底盘"和"没用到但碰巧够着"。建议补两条:

- 三类目标(§7 阶段 3 的 ①②③)的成功率与误差**分开统计**;
- **"冻结底盘"对照组**(把 `base_vel` 动作置零重跑 `play.py`):第 ③ 类目标的成功率必须
  明显低于正常策略,否则说明底盘根本没被用上。

你还有别的判据吗?这会影响评估脚本怎么写、以及阶段 3 何时算通过。

### R4. 近期是否上机?

- 若**近期要上机**:我会在阶段 1/2 就顺带落实部署侧清单(关节名重映射、度/弧度换算、
  `/cmd_vel` 看门狗、`/odom` 推算姿态),不留到阶段 5。
- 若**暂不上机**:阶段 5 只产出对照表文档,不写实机节点。

### ~~R5. EE 目标参考系怎么改?~~ ✅ 已关闭(2026-09-23)

**决定:采用新写 `UniformPoseWorldCommand` 的方案,只在 `ranger_cr10` 使用,
不修改现有 GO2-PIPER 的 `UniformPoseWBCCommand`。** 配套三条同时生效(§4.2):

1. 目标**内部固定在世界系**(`target_pose_w`),本次 command 生命周期内不随底盘移动;
2. **Policy 观测用 base-relative target**(`target_pose_b`),不给世界绝对坐标;
3. **Reward 按世界系 EE 误差**计算;必要时为 `ranger_cr10` 单独新增 reward function,
   不改共享实现的默认行为。

---

## 10. 当前无阻断项

A 层两个阻断项都已解决(§2.2 限位已定、§5 A2 末端帧已定),**阶段 0 已完成**(§7),
**R5 已关闭**(§9:采用 `UniformPoseWorldCommand`)。剩余待定项都不阻塞开工:

| 项 | 影响阶段 | 默认动作 |
|---|---|---|
| R1 末端帧 vs 控制器 TCP | 阶段 5 | 先按指尖中心做,阶段 5 前核对 |
| R3 成功判据 | 阶段 3 验收 | GO2-PIPER 量级 + 三类目标分开统计 + 冻结底盘对照组 |
| R4 是否上机 | 阶段 1/5 的深度 | 默认按"暂不上机" |

**下一步直接进入阶段 1(接入与注册)。** 命令与观测按 §4.2 的三层坐标关系搭:
世界系固定目标 → base-relative 观测 → Policy → `[vx, wz, 6×CR10]`,奖励用世界系 EE 误差。
