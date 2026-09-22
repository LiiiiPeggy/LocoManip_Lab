# RangerBox-CR10-Lidar WBC 训练移植计划

**状态:决策已锁定,可开工** · 拟稿 2026-09-22 · 决策确认 2026-09-23 · 参照平台:GO2-PIPER

把 `rangerboxcr10lidar_description`(轮式移动操作机器人)接入 LocoManip_Lab,
训练 WBC 策略:**底盘按速度指令行走,同时末端执行器跟踪位姿指令**,
并打通 MuJoCo 部署与实机(agx)链路。

**决策状态**:A/B/C 三层共 15 项已全部由用户确认,见 §5;仿真与实机的接口边界见 §6;
剩余待定项(R1–R4)都不阻塞开工,见 §9。
标 **⚠️** = 已核实的障碍或未验证项。

---

## 1. 目标与非目标

**目标**

1. 把该机器人转换/接入 Isaac Lab,注册 `RANGER-CR10-WBC`(及 `-Flat`)。
2. 训练 WBC 策略:末端位姿跟踪 + 底盘速度跟踪,沿用 GO2-PIPER 的观测/奖励/课程架构。
3. MuJoCo 部署验证(复用 `mujoco/deploy/` 架构,但脚本需重写)。
4. 策略的**关节顺序、观测布局、动作语义、控制频率**与 agx 实机对齐。

**非目标**

- **实机上机运行**。agx 里**不存在**任何策略推理代码(全仓库 grep `onnx|torchscript|isaac ?lab|checkpoint`
  零命中),实机侧要新写一个 ROS 节点,而且这个机器人**没有 `/cmd_vel` 看门狗**(见 §6.4)。
  本期只做接口对齐与文档化,上机是独立项目。
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

独立可控自由度 = 15(8 底盘 + 6 臂 + 1 夹爪),但**底盘不能按关节控制**(见 §4)。

### ⚠️ 2.1 转换前必须修的模型缺陷

1. `package://` URI —— Isaac Sim 无法解析,需映射到本地路径或改写为相对路径。
2. 臂关节 `effort=0/velocity=0` —— 力矩与速度上限改由 `ArticulationCfg` 的 actuator 提供。
3. 连续关节无 `<limit>` —— 需显式给定,否则执行器配置异常。
4. 7 个 mimic 关节 —— Isaac Lab 不支持 URDF mimic;建议**压成 1 个关节**(夹爪本次不进动作空间)。
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
| `config/ranger_cr10/flat_env_cfg.py` / `wbc_env_cfg.py` | 同目录 | 命令范围、奖励权重、EE body 名 |
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
- 动作:`actions.joint_pos.joint_names`(写死了 12 个腿关节名)

### 3.2 ⚠️ 共享代码里硬编码的 body/joint 名

| 位置 | 硬编码 | 影响 |
|---|---|---|
| `mdp/observations.py:117-123` `joint_pos_rel` | 默认参数是 12 个腿关节 + `joint.*` | **新平台观测项必须显式传 `joint_names`**,否则匹配为空 → 观测维度 0,静默出错 |
| `mdp/observations.py:143-149` `joint_vel_rel` | 同上 | 同上 |
| `mdp/observations.py:82-83` `end_effector_link0_relative_pose` | 硬编码 `end_effector` 与 `link0` | **找不到时静默返回全 0**(85-86 行有保护)→ critic 里混入 7 维常量,训练照跑但信息是假的 |
| `mdp/rewards.py:54` `position_command_b_error_exp` | 硬编码 `find_bodies("link0")` | Flat 模式用;需要时加 `link0_name` 参数(1 行) |
| `mdp/rewards.py:71` `position_command_error_exp` | **已有参数** `link0_name="link0"` | WBC 模式可通过 `params` 覆盖,**无需改共享代码 ✅** |

**结论**:WBC 路径基本"只加不改";但两条静默失败通道必须在阶段 1 用**打印实际观测维度**证伪
(AGENTS.md 已明确:exit 0 不算成功)。

---

## 4. 核心决策:底盘怎么进动作空间(已被实机证据收敛,方案 B 已定)

原以为是纯设计选择,但实机接口把选项砍掉了一大半。已核实的事实:

> **AgileX Ranger 底盘的唯一输入是 `/cmd_vel`(`geometry_msgs/Twist`)**,驱动内部
> 依 `(vx, vy, wz)` 自动切换运动模式并解算 4 个转向角与 4 个轮速。**没有任何关节级指令接口。**

| 模式 | 触发条件 | 行为 |
|---|---|---|
| `DUAL_ACKERMAN` | 默认 | `radius=|vx|/|wz|`,转向限幅 ±0.6981 rad |
| `PARALLEL`(蟹行) | `vy != 0` | 转向角 = `atan(vy/vx)`,速度 = `sqrt(vx²+vy²)`,限幅 ±1.570 rad |
| `SPINNING` | 转弯半径 < `min_turn_radius`(0.8103 m) | 原地转,`wz` 限幅 ±0.7853 rad/s |

底盘能力上限:`max_linear_speed=2.7 m/s`、`max_angular_speed=0.7853 rad/s`。

**因此三个方案的可行性:**

- **方案 A —— 策略输出 8 个底盘关节(我原先的推荐之一):🚫 不可行。**
  实机不接受关节级底盘指令,这条路的策略无法上机。已排除。
- **方案 B —— 底盘速度作为输入指令(推荐,忠实移植 GO2-PIPER):**
  动作 = 6 个臂关节。`(vx, vy, wz)` 是**输入指令**,仿真里由与 AgileX 驱动同构的控制器执行
  (复现上表三种模式与限幅)。奖励:**去掉 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`**
  —— 底盘被完美执行时这两项恒等于 1,只会稀释回报、不产生梯度。
  任务本质变成"底盘在按指令移动时,末端跟踪位姿",即 GO2-PIPER WBC 的完整语义。
- **方案 C —— 策略输出 `(vx, vy, wz) + 臂关节`(你 `agx/Sim2real.md` 里写的形态):**
  底盘速度由策略决定而非外部指令。可行,但**任务与奖励需要重新设计**
  (没有速度跟踪目标了,得换成到达/NBV/能量等),不能直接套 GO2-PIPER 的奖励结构。

**决定(2026-09-23):采用方案 B,本期不做方案 C。** 具体:

| 子项 | 决定 |
|---|---|
| 底盘接口 | `(vx, vy, wz)` 作为**输入指令**,由仿真里与驱动同构的控制器执行;动作 = 6 个臂关节 |
| **速度指令范围** | **保守值**:`lin_vel_x = ±0.5 m/s`、`ang_vel_z = ±0.5 rad/s` |
| **`vy` 蟹行** | **不加入**(`lin_vel_y = 0`),避开 `PARALLEL` 模式的切换跳变 |
| 奖励 | 去掉 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`(底盘被完美执行时恒等于 1) |
| 方案 C(`RangerArmReachEnv`) | 不在本期范围;作为后续独立课题(见 `agx/Sim2real.md`) |

保守范围的依据:实机自己的导航栈用的是 `speed_lim_v = 0.5`、`speed_lim_w = 0.5`
(`ranger_bringup/param/velocity_smoother.yaml`),且硬件上限 2.7 m/s / 0.785 rad/s
远超 WBC 任务所需。训稳后可再往上放。

> 补充:`Sim2real.md` / `Sim2real_cursor.md` 我核实过,**是文献调研,不是本机器人的操作流程或部署脚本**,
> 且没有引用任何 checkpoint。计划里不把它们当作既有工作。

---

## 5. 已决定的决策点(全部经用户确认,2026-09-23)

| 层 | 事项 | 决定 |
|---|---|---|
| A1 | CR10 关节限位 | 以 **`rangercr10lidar.urdf`** 为准(§2.2) |
| A2 | 末端执行器帧 | **夹爪指尖中心**(`gripper_finger1_finger_tip_link` / `gripper_finger2_finger_tip_link` 之间的中点,见 §7 阶段 1) |
| B1 | 底盘方案 | **方案 B**;速度范围 **保守值 ±0.5 m/s / ±0.5 rad/s**;**不加 `vy` 蟹行**;本期不做方案 C(§4) |
| B2 | 底盘状态进观测 | 原阻塞原因(agx 的 `/actuator_state` 混叠 bug)**已由用户确认为已修复**,`/cmd_vel` 现在可正常驱动底盘。是否纳入底盘关节状态见 §9 |
| B3 | 无 IMU 的替代 | 走 **(a) 从 `/odom` 推算**,并在仿真里加入**噪声/漂移随机化** |
| B4 | 臂控制频率 | **训练与部署都降到 10 Hz**。实机能力:控制器 30004 端口 8 ms(**125 Hz**)真实反馈;控制频率实测可达 20 Hz,最高 33 Hz,当前默认实机参数 10 Hz |
| B5 | 底盘控制器保真度 | 先 **(b) 只做运动学正确的全向 swerve 解算**,不模拟驱动的模式切换;上机前再评估是否切到完全复现 |
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

### 6.5 控制频率对齐(已定:全链路 10 Hz)

| 回路 | 能力 | 本计划采用 |
|---|---|---|
| 底盘 `/cmd_vel` | 50 Hz | 50 Hz(底盘驱动侧不变) |
| 臂状态反馈 | **125 Hz**(30004 端口,8 ms) | 10 Hz |
| 臂控制 | 上限 33 Hz,实测 20 Hz,默认 10 Hz | **10 Hz** |
| 框架训练步长 | `dt=0.005 × decimation=4` = 50 Hz | 臂动作**降频到 10 Hz** |

**决定**:臂动作在训练与部署两端都按 **10 Hz** 对齐(与当前实机默认参数一致),
这样仿真里学到的动作频率和实机能执行的完全一致,**不需要插值层**。
底盘仍按 50 Hz 接受指令(驱动侧本来就是 50 Hz)。

实现方式(阶段 3 落地):把 `actions` 拆成两个动作项 —— 底盘/需要高频的项按 50 Hz,
臂关节位置项按 10 Hz(用 `decimation` 或让臂动作项每 5 个仿真步才更新一次),
观测里的 `last_action` 会自动按动作项顺序拼接。

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
     `collider_type: convex_hull`、`convert_mimic_joints_to_normal_joints: false`)。
2. 用 Isaac Lab 自带工具转换:`IsaacLab5/scripts/tools/convert_urdf.py`。
   网格用**减面后**的版本(§5.2),转换时不要再动网格。
3. **验收(必须打印证据,不能只看 exit code)**:
   - 打印 USD 的关节表 —— 关节数、名称、类型、limit;
   - 确认底盘 8(4 转向 + 4 驱动)+ 臂 6 全部在列,根 link 是 `base_link`;
   - mimic 关节按预期处理(压成 1 个);
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
   - `end_effector` 帧取 **夹爪指尖中心**——URDF 里没有现成 link,
     需要在转换后的 USD 上加一个 fixed frame,位置为 `gripper_finger1_finger_tip_link` 与
     `gripper_finger2_finger_tip_link` 的中点(两指闭合时即为夹持中心);
   - WBC 的 `link0_name` 参数指向 `base_link`(§3.2,零改动共享代码);
   - 速度指令范围 `lin_vel_x = ±0.5`、`ang_vel_z = ±0.5`、**`lin_vel_y = 0`**。
6. **验收**:`list_envs.py` 行数 **28 → 32**;`zero_agent.py --task RANGER-CR10-WBC` 不报错
   并**打印 policy 观测维度**(预期 3×(3+3+N+N+N+3+7),N = 实际进观测的关节数)。

### 阶段 2 — 底盘单独跑通
7. 固定臂关节,只按速度指令驱动底盘:直行、转向、原地旋转。
   **不做蟹行**(§4 决定不加 `vy`),但要单独验证一次"`vy=0` 时驱动确实走 Ackermann 分支"。
8. **验收**:速度跟踪误差收敛;`|vx| ≤ 0.5`、`|wz| ≤ 0.5` 全范围内都能跟住。

### 阶段 3 — WBC 联合训练
9. 打开臂动作与末端位姿奖励,从 GO2-PIPER 的权重起步(位置 4.5 / 姿态 -4.0),按实测调。
   - **臂动作降到 10 Hz**(§6.5):动作项按 50 Hz / 10 Hz 拆分,与实机默认参数一致;
   - 命令范围沿用阶段 1 的保守值,别训超出硬件能力的动作;
   - 从 `/odom` 推算的 `base_ang_vel` / `projected_gravity` 要带噪声与漂移随机化(§5 B3)。
10. **验收**:`Metrics/ee_pose/position_error` 与 `orientation_error` 收敛到与 GO2-PIPER 同量级
    (它训到 0.065 m / 0.095);出图 + 视频人工确认"边走边够"。

### 阶段 4 — MuJoCo 部署
11. 建 MJCF,写 `deploy_mujoco/ranger_cr10/`。**GO2-PIPER 那份不能照抄**:它写死了 18 维观测
    和 12 个腿关节的 `ISAAC_TO_MUJOCO` 重排表。
12. **验收**:`play.py` 导出 `policy.pt`,键盘遥操作复现 Isaac 行为。

### 阶段 5 — 实机接口对齐(不上机)
13. 产出**"仿真 ↔ 实机"对照表**:每个观测分量/动作分量 → agx 的话题与字段,含单位、
    频率、命名映射。用 `--export_io_descriptors`(`mdp/observations.py` 的 `generic_io_descriptor`
    与 `record_joint_names`)导出的布局描述作为抓手。
14. **验收**:对照表与 agx 代码逐项核对通过;列出实机部署节点待办清单 ——
    关节名重映射(`joint1..6` ↔ `cr10_joint1..6`)、度/弧度换算、**臂指令 10 Hz**、
    `/cmd_vel` 看门狗(驱动无超时,停发即保持最后指令)、夹爪 0–1000 原始计数与反向映射、
    无 IMU 时用 `/odom` 推算姿态。

---

## 8. 风险清单

| 风险 | 状态 | 缓解 |
|---|---|---|
| ~~CR10 限位两份文件矛盾~~ | ✅ 已定(§2.2,以 `rangercr10lidar.urdf` 为准) | — |
| ~~控制频率不匹配(50 Hz vs 臂)~~ | ✅ 已定(§6.5,全链路臂 10 Hz,无需插值层) | — |
| ~~实机无 IMU~~ | ✅ 已定(§5 B3,`/odom` 推算 + 仿真噪声随机化) | — |
| ⚠️ `/odom` 推算角速度/姿态的精度 | 未知,需实测 | 阶段 5 前用真机跑一段直行+旋转,对比 IMU 或外部真值;仿真里按实测噪声标定 |
| URDF 转换质量(无 dynamics/limit、mimic、`package://`) | 未验证 | 阶段 0 最小可加载优先,逐项修 |
| 176 kg + 15 自由度 + 31 MB 网格 | 训练吞吐可能远低于 GO2-PIPER 的 8192 环境 | 阶段 2 先实测吞吐再定规模 |
| 静默失败(观测为空/全零) | 已知有两处通道(§3.2) | 阶段 1 强制打印观测维度 |
| 车轮摩擦/侧滑未辨识(§5 C3 决定先不做) | 已知 sim-to-real 主要误差源 | 靠域随机化覆盖;上机后按实测回填 |
| 无 `/cmd_vel` 看门狗 | 未处理 | 部署节点自建超时归零(阶段 5 清单已列) |
| 关节名不匹配(driver `joint1..6` vs URDF `cr10_joint1..6`) | 未处理 | 阶段 5 映射表覆盖 |
| 末端帧 = 指尖中心,与控制器 TCP 是否一致 | **未确认** | 见 §9 R1 —— 上机前必须核对,否则位姿指令系统性偏移 |

---


## 9. 剩余待定项(其余问题已全部锁定,见 §5)

### R1. 末端帧与控制器 TCP 是否一致?(上机前必须核对)

已定仿真里的 `end_effector` = **夹爪指尖中心**(§5 A2)。但这只在仿真侧成立 ——
实机 `ToolVectorActual` 报的是 **Dobot 控制器里配置的工具坐标系(TCP)**。

- 如果控制器 TCP 也配在指尖中心 → 两侧一致,直接可用;
- 如果配在法兰或其他位置 → 部署节点必须做一次固定偏移变换,否则位姿指令系统性偏一个常数。

**需要你确认或实测**:控制器当前的 TCP 偏移(Tx/Ty/Tz)。
不确认也能先训(仿真内部自洽),但阶段 5 的对齐表必须解决它。

### R2. 底盘关节状态要不要进观测?(§5 B2 的遗留)

你已确认 agx 的 `/actuator_state` 混叠 bug 已修复、`/cmd_vel` 可正常驱动底盘,
所以原先阻塞选项 (b) 的原因不存在了。两种设计现在都可行:

- **(a) 不含**(我仍倾向这个):底盘当黑盒,策略只看到速度指令与自身实际速度。
  与"底盘由驱动接管"的实机分工一致,观测维度也更小。
- **(b) 含**:把 4 个转向角 + 4 个轮速放进 policy/critic 观测。信息更全,
  但要求实机侧稳定提供这 8 个量,且仿真里的转向角要与实机同尺度。

**需要你定**。选 (b) 的话我会在阶段 1 就把它加进观测配置;不回复则默认按 (a) 走。

### R3. "训练效果好"的判据

目前只知道沿用 GO2-PIPER 的指标(末端位置/姿态误差收敛到 0.065 m / 0.095 量级)+ 人工看视频。
你还有别的判据吗?例如"行进中末端抖动小于某阈值"、"能走到指定位置并够到目标"。
这会影响评估脚本怎么写、以及阶段 3 何时算通过。

### R4. 近期是否上机?

- 若**近期要上机**:我会在阶段 1/2 就顺带落实部署侧清单(关节名重映射、度/弧度换算、
  `/cmd_vel` 看门狗、`/odom` 推算姿态),不留到阶段 5。
- 若**暂不上机**:阶段 5 只产出对照表文档,不写实机节点。

---

## 10. 当前无阻断项

A 层两个阻断项都已解决(§2.2 限位已定、§5 A2 末端帧已定)。R1–R4 都不阻塞**开工**:

| 项 | 影响阶段 | 默认动作 |
|---|---|---|
| R1 末端帧 vs 控制器 TCP | 阶段 5 | 先按指尖中心做,阶段 5 前核对 |
| R2 底盘状态进观测 | 阶段 1 | 默认按 (a) 不含 |
| R3 成功判据 | 阶段 3 验收 | 默认用 GO2-PIPER 的量级 |
| R4 是否上机 | 阶段 1/5 的深度 | 默认按"暂不上机" |

**可以开始阶段 0(URDF→USD 转换)了。**
