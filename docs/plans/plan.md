# RangerBox-CR10-Lidar WBC 训练移植计划

**状态:待审阅** · 拟稿 2026-09-22 · 参照平台:GO2-PIPER

把 `rangerboxcr10lidar_description`(轮式移动操作机器人)接入 LocoManip_Lab,
训练 WBC 策略:**底盘按速度指令行走,同时末端执行器跟踪位姿指令**,
并打通 MuJoCo 部署与实机(agx)链路。

标 **❓** = 需要你拍板;标 **⚠️** = 已核实的障碍;标 **🚫** = 阻断项,不解决不能开训。

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
需要定一个单一来源以免漂移(见 §5.5)。

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
| 网格 | 27 个 STL,共 83 MB,`package://` URI,最大单件 31 MB |
| 缺失 | 无 `<dynamics>`(阻尼/摩擦)、臂关节无力矩速度上限、无悬架、无 `<sensor>` |

独立可控自由度 = 15(8 底盘 + 6 臂 + 1 夹爪),但**底盘不能按关节控制**(见 §4)。

### ⚠️ 2.1 转换前必须修的模型缺陷

1. `package://` URI —— Isaac Sim 无法解析,需映射到本地路径或改写为相对路径。
2. 臂关节 `effort=0/velocity=0` —— 力矩与速度上限改由 `ArticulationCfg` 的 actuator 提供。
3. 连续关节无 `<limit>` —— 需显式给定,否则执行器配置异常。
4. 7 个 mimic 关节 —— Isaac Lab 不支持 URDF mimic;建议**压成 1 个关节**(夹爪本次不进动作空间)。
5. 23 个纯 TF frame link 无 inertial —— 转换时会被压掉,需确认树结构不受影响。
6. 31 MB 底盘网格 —— 训练加载慢,建议简化(见 §5.4)。

### 🚫 2.2 阻断项:CR10 关节限位两份文件互相矛盾

| 关节 | `agx/TCP-IP-ROS-6AXis/dobot_description/urdf/cr10_robot.urdf` | `rangercr10lidar.urdf`(本模型) |
|---|---|---|
| j1 | ±3.14 | **-3.92 … 0.94** |
| j2 | ±3.14 | **±1.57** |
| j3 | ±2.861 | ±2.86 ✅ |
| j4 | ±3.14 | ±3.14 ✅ |
| j5 | ±3.14 | ±3.14 ✅ |
| j6 | **±6.28** | **±3.14** |

用错限位的后果:策略会在仿真里学到超出实机可达范围的姿态,上机直接撞限位。
**必须先确认真机实际的关节限位**(查 Dobot 控制器返回,或实测),再定 `ArticulationCfg`。
在此之前不要开始阶段 3。

---

## 3. 与 GO2-PIPER 架构的映射

框架的**平台相关代码很薄**(每平台约 250 行),共享 `mdp/` 与核心 env cfg 可复用。
新增平台需要:

| 新增/修改 | 参照 | 说明 |
|---|---|---|
| `assets/ranger_cr10/*.usd` + `config.yaml` | `assets/go2_piper/` | URDF→USD 产物(§7 阶段 0) |
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

## 4. ❓ 核心决策:底盘怎么进动作空间(已被实机证据收敛)

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

**我的建议**:先做 **B**(链路最短、奖励结构可直接复用、策略可部署),把
`RangerArmReachEnv`(方案 C)作为下一个独立课题——你的 `Sim2real.md` 里那张 6 步计划图
本来就把它排在后面。

> 补充:`Sim2real.md` / `Sim2real_cursor.md` 我核实过,**是文献调研,不是本机器人的操作流程或部署脚本**,
> 且没有引用任何 checkpoint。计划里不把它们当作既有工作。

---

## 5. 已决定的决策点(2026-09-23)

| 事项 | 决定 | 落地方式 |
|---|---|---|
| 网格入库 | **以普通 Git 对象入库,不走 LFS** | `.gitattributes` 加路径级例外 `rangerboxcr10lidar_description/meshes/** -filter ...`,放在 `*.stl`/`*.STL` 规则之后(后者仍让 MuJoCo 资产走 LFS);`.pre-commit-config.yaml` 给 `check-added-large-files` 加同路径 `exclude` |
| `agx/` 归属 | **做成 submodule** | 固定到 `28beddb`(master);git dir 已收进 `.git/modules/agx` |
| 转换前的模型缺陷 | 按 §2.1 逐项修 | — |

仍未定的见 §9。

---

## 6. 实机接口:已核实的事实与缺口

> 这一节是后续部署与"接口对齐"的依据,全部来自 agx 仓库实读。

### 6.1 控制接口

| 部件 | 接口 | 单位/范围 | 频率 |
|---|---|---|---|
| 底盘 | `/cmd_vel` `geometry_msgs/Twist` | m/s, rad/s;只用 `linear.x/y`、`angular.z` | 50 Hz |
| 臂 | `/cr10_robot/joint_controller/follow_joint_trajectory`(MoveIt 用) | **rad** | `servoj` 内部 **2.5 Hz** 流式 |
| 臂(备选) | `/dobot_v4_bringup/srv/ServoJ` 服务 | **度(deg)**,一次性 | 单次调用 |
| 夹爪 | `/gripper/ctrl` `dh_gripper_msgs/GripperCtrl` | **position 0–1000 原始计数,force/speed 百分比** | 50 Hz |

⚠️ `ServoJ` 服务是**度**,而 trajectory action 是**弧度**,内部帮你换算 —— 混用是 57.3 倍误差。
⚠️ `JointMovJ.srv` 在 v4 包里**不存在**,但 `agx/CLAUDE.md` 和 `README.md` 都让人调它 —— 文档是错的。

### 6.2 状态反馈

`/odom`(50 Hz)、`/system_state`、`/motion_state`、`/actuator_state`、`/battery_state`、
`/joint_states`(臂,**10 Hz**)、`/gripper/joint_states`(50 Hz)、`/dobot_v4_bringup/msg/ToolVectorActual`(TCP 位姿,10 Hz)。

- ⚠️ **全仓库没有任何 IMU 话题**(`agx/debugs/bug_ranger.md` 自己也在问"IMU 的 launch 在哪")。
  而框架的观测里有 `base_ang_vel` 与 `projected_gravity`(通常来自 IMU)→
  **实机侧只能从 `/odom` 推算**,这是必须设计的替代路径。
- ⚠️ `/actuator_state` 的填充循环有 bug,8 个执行器返回的是同一份数值(复制的是数组而非元素)。

### 6.3 关节命名/顺序(部署时最要命的一环)

- 臂:驱动 `/joint_states` 发的是 **`joint1..joint6`**,而整合 URDF 里叫 **`cr10_joint1..cr10_joint6`**
  → 名字不匹配,`robot_state_publisher` 不动。
- 线协议顺序:`ServoJ(a,b,c,d,e,f)` 其中 a=j1 … f=j6,**单位度**。
- 底盘:`fr / fl / rl / rr` 各 `(steering, wheel)`,注意 `fr_steering_joint` 少一个 `wheel` 后缀。
- 夹爪:驱动把原始计数 `0–1000` 线性映射到 `gripper_finger1_joint` 的 `0–0.637 rad`,**且是反向的**
  (`msg.position[0] = (1000-raw)/1000.0 * 0.637`)。

### 6.4 🚫 部署前必须补的安全项

**驱动里没有 `/cmd_vel` 看门狗** —— 一旦停止发布,最后一条速度指令会**永久保持**。
部署节点的循环必须自己实现超时归零。

### 6.5 控制频率不匹配(必须在训练时就考虑)

| 回路 | 频率 |
|---|---|
| 底盘 `/cmd_vel` | 50 Hz |
| 臂状态反馈 | 10 Hz |
| 臂 `servoj` 流式 | **2.5 Hz** |
| 框架训练步长 | 50 Hz(`dt=0.005 × decimation=4`,**所有关节同一频率**) |

策略若以 50 Hz 输出臂关节位置,实机只能以 ~2.5 Hz 平滑接受 →
**需要一个插值/保持层**,或干脆把臂动作降频。这是设计约束,不是实现细节。

---

## 7. 实施步骤(每阶段有可验证产物)

> 验收遵循 AGENTS.md:**不认 exit 0,只认正证据**(行数/维度、生成文件、指标)。

### 阶段 0 — URDF→USD 转换(风险最高,先做最小验证)
1. 建 `assets/ranger_cr10/`,照抄 GO2-PIPER 的 `config.yaml` 字段
   (`fix_base: false`、`merge_fixed_joints: true`、`collider_type: convex_hull`、
   `convert_mimic_joints_to_normal_joints: false`)。
2. 用 Isaac Lab 自带工具转换:`IsaacLab5/scripts/tools/convert_urdf.py`。先解决 `package://` URI。
3. **验收**:打印 USD 关节表 —— 关节数、名称、类型、limit;确认底盘 8 + 臂 6 在列,
   mimic 已按预期处理,根 link 是 `base_link`。

### 阶段 1 — 接入与注册
4. 写 `ranger_cr10_articulation_cfg.py`:填 `effort_limit`(URDF 给的是 0,按 CR10 规格填,
   **并与 §2.2 的限位矛盾一并解决**)、`stiffness`/`damping`、初始角(转向归零、臂取肘部朝前姿态)。
5. 写 `config/ranger_cr10/` 四件套 + agents。
6. **验收**:`list_envs.py` 行数 **28 → 32**;`zero_agent.py --task RANGER-CR10-WBC` 不报错
   并**打印 policy 观测维度**(预期 3×(3+3+N+N+N+3+7),N=实际进观测的关节数)。

### 阶段 2 — 底盘单独跑通
7. 固定臂关节,只按速度指令驱动底盘:直行、蟹行(`vy≠0`)、原地旋转,并复现 §4 的三种模式与限幅。
8. **验收**:速度跟踪收敛;**侧向速度指令能被跟踪** —— 这是 4WIS 建模正确性的关键证据。

### 阶段 3 — WBC 联合训练
9. 打开臂动作与末端位姿奖励,从 GO2-PIPER 的权重起步(位置 4.5 / 姿态 -4.0),按实测调。
   命令范围按实机能力设上限(`|vx| ≤ ~1.0 m/s`,`|wz| ≤ 0.7853 rad/s`),别训超出硬件的动作。
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
14. **验收**:对照表与 agx 代码逐项核对通过;列出实机部署节点缺失的功能清单
    (关节名重映射、度/弧度换算、臂动作插值、看门狗、夹爪原始计数、无 IMU 的替代)。

---

## 8. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| 🚫 CR10 限位两份文件矛盾 | 策略学到不可达姿态,上机撞限位 | 阶段 3 前查真机实际限位 |
| URDF 转换质量(无 dynamics/limit、mimic、`package://`) | 阶段 0 反复 | 最小可加载优先,逐项修 |
| 31 MB 网格 + 176 kg + 15 自由度 | 训练吞吐可能远低于 GO2-PIPER 的 8192 环境 | 阶段 2 先实测吞吐再定规模 |
| 静默失败(观测为空/全零) | 白训一整天 | 阶段 1 强制打印观测维度 |
| 控制频率不匹配(50 Hz vs 臂 2.5 Hz) | 上机动作抖动/被拒 | 阶段 5 设计插值/降频层 |
| 实机无 IMU | 观测里两项无来源 | 用 `/odom` 推算,并评估精度 |
| 无 `/cmd_vel` 看门狗 | 通信中断即失控 | 部署节点自建超时归零 |
| 关节名不匹配(driver `joint1` vs URDF `cr10_joint1`) | TF 不动 | 阶段 5 映射表覆盖 |

---

## 9. 需要你决策的问题(详细版)

按"是否阻塞开工"分三层。A 层不解决不能开训;B 层影响训练设计与成败;C 层可以边做边定。

### A 层 — 阻断项

#### A1. CR10 的关节限位到底是多少?(对应 §2.2)

**为什么是阻断项**:`ArticulationCfg` 里的关节限位决定策略能学到什么姿态。用错限位 → 策略
在仿真里学会伸到实机去不了的位置 → 上机撞限位/报错。而且这不是"精度问题",是**能不能跑**的问题。

**冲突现状**(两个文件对 j1/j2/j6 说法不同):

| 关节 | `dobot_description/urdf/cr10_robot.urdf` | `rangercr10lidar.urdf` | 差异 |
|---|---|---|---|
| j1 | ±3.14 rad(±180°) | -3.92 … 0.94 rad(-224.6° … 53.9°) | **差 2.14 rad** |
| j2 | ±3.14 | ±1.57 | **差 1.57 rad** |
| j6 | ±6.28(±360°) | ±3.14 | **差 3.14 rad** |
| j3/j4/j5 | ±2.861 / ±3.14 / ±3.14 | 同 | 一致 |

j2 的差异最可疑:±90° 的肘关节对 CR10 这种协作臂很常见,但 ±180° 也并非不可能。
j1 的 -224.6°…53.9° 这个不对称区间不像是官方规格,更像是**特定安装姿态下的可达范围**。

**怎么解决(按优先级)**:

1. **读控制器**(首选):Dobot 控制器里有实际限位参数,可用 DobotStudio 查看,或走
   TCP 接口查询。这是唯一权威来源。
2. **查 CR10 官方规格书**:确认出厂限位,再判断两份文件哪个是被改过的。
3. **慢速实测**:手动 jog 到各关节边界,读角度。

**你能接触到真机吗?** 这是 A1 能否解决的前提。

**如果都拿不到,我可以做的保守兜底**:取两组限位的**交集**
(j1: -3.14…0.94,j2: ±1.57,j6: ±3.14)。代价是牺牲部分工作空间,但保证不会训出
任何一份文档认为超限的姿态。**需要你同意我才这么做** —— 因为它会永久影响这一期策略的能力上限。

#### A2. 仿真里的末端帧要和 Dobot 的 TCP 对齐

**为什么是阻断项**:框架要求 body 名 `end_effector`,URDF 里没有。但更关键的是:
实机的 `ToolVectorActual` 报的是 **Dobot 控制器里配置的工具坐标系(TCP)** 的位姿。
如果仿真里的 EE 帧和控制器里的 TCP 不一致,那么"末端到 (0.5, 0, 0.6)"这句话在两边的
物理含义就不同 —— 训练出来的策略上机就会系统性偏一个固定量。

**需要你确认**:控制器里当前配置的 TCP 偏移是多少(Tx/Ty/Tz 与姿态)?是夹爪指尖中心,
还是法兰面?

**兜底**:用 `gripper_base_link` 作 EE 帧,并在文档里明确写"仿真 EE 帧 = 法兰侧",
上机前再统一到真正的 TCP。可用但不对齐。

### B 层 — 设计选择(影响训练能否成功)

#### B1. 底盘在动作空间里的位置(§4 的 B/C 分叉)

**背景**:实机底盘唯一入口是 `/cmd_vel`,驱动内部解算 4WS 并自动切换
`DUAL_ACKERMAN` / `PARALLEL`(蟹行)/ `SPINNING` 三种模式。"策略学 8 个底盘关节"已确认不可行。

**方案 B —— 底盘速度作输入指令(我推荐)**

- 动作 = 6 个臂关节;`(vx, vy, wz)` 是任务输入,仿真里由与驱动同构的控制器执行。
- 奖励:去掉 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`(被完美执行时恒等于 1,只稀释回报)。
- 语义 = "底盘按指令走的同时,末端跟踪位姿",与 GO2-PIPER WBC 完全同构。

**方案 B 还需要你定两件事**:

- **B1a. 速度指令的范围**。
  - (i) 按硬件上限:`|vx| ≤ 2.7 m/s`,`|wz| ≤ 0.785 rad/s`;
  - (ii) 保守:实机自己的导航栈用的是 `speed_lim_v = 0.5`、`speed_lim_w = 0.5`,
    我建议从这里起步(约 `±0.5 m/s`、`±0.5 rad/s`),训稳了再往上放。
  - 理由:指令是"任务要求",超出底盘能力时策略只能学出无意义的胳膊代偿。
- **B1b. 要不要把 `vy`(蟹行)放进指令**。
  - 放:能体现这台底盘的全向能力,但 `vy ≠ 0` 会触发驱动的模式切换,切换瞬间有跳变;
  - 不放:只训 `vx + wz`,避开模式切换,链路最短。
  - 我建议**先不放**,等 B 跑通再开 vy 做对比实验。

**方案 C —— 策略输出 `(vx, vy, wz) + 臂关节`(你 `agx/Sim2real.md` 里写的形态)**

- 底盘速度由策略决定,不再有外部速度指令。
- **必须重新设计任务与奖励**:原来的"跟踪速度指令"没有目标了。
  - 那么任务是什么?① 底盘走到某个目标位姿 + 末端跟踪位姿?② 末端够到目标点、底盘自由?
    (你文档里提到的 NBV 属于第三类。)
- 我建议作为 B 之后的独立课题,不要一次做两件事。

#### B2. 观测里要不要含底盘状态(转向角 / 轮速)

- 实机的底盘关节状态来自 `/actuator_state`,但**驱动里这个填充循环有 bug**
  (8 个执行器返回同一份数值,复制的是数组而非元素)。
- (a) **不含**(推荐):把底盘当黑盒,策略只知道速度指令和自身实际速度 —— 这与
  "底盘由驱动接管"的实机分工一致。
- (b) 含:需要**先修 agx 的驱动 bug**(另一个仓库的改动),并且要确认修的版本能上机。
- 需要你定;选 (b) 的话我会把"修 agx"列成前置任务。

#### B3. 没有 IMU,`base_ang_vel` 和 `projected_gravity` 从哪来

- 框架的观测里有这两项(通常来自 IMU),而**实机没有任何 IMU 话题**。
- (a) **从 `/odom` 推算**(推荐):但轮式里程计的角速度在打滑时误差大,
  姿态更是积分漂移。若选这条,我建议在仿真里加入与之匹配的噪声/漂移随机化。
- (b) 加装 IMU:硬件改动,超出本期。
- (c) 从观测里去掉这两项:最诚实,但策略接口与 GO2-PIPER 不再一致,且失去姿态反馈。
- **需要你定**;若选 (a),还需要你评估一下 `/odom` 的实际质量(有没有跑过、漂不漂)。

#### B4. 臂的控制频率(50 Hz 训练 vs 2.5 Hz 实机)

- 实机 `servoj` 流式只有 **2.5 Hz**;臂状态反馈 10 Hz;而仿真里所有关节是 50 Hz。
- (a) **训练时就把臂动作降频**(推荐):让仿真与实机能力一致,避免学到高频抖动。
  具体降到多少需要你定(10 Hz?5 Hz?),我建议先按 10 Hz。
- (b) 训练 50 Hz + 部署时插值:仿真里能学到的动态更丰富,但上机要靠插值层掩盖差异。

#### B5. 底盘控制器的保真度

- 仿真的底盘控制器要复现 AgileX 驱动的**三种模式切换与限幅**吗?
- (a) 完全复现:一致性高,但把驱动的 quirks(跳变、边界行为)带进训练;
- (b) **只做运动学正确的全向 swerve 解算**(推荐起步):平滑、可微,先训出行为;
- (c) 分阶段:先 (b),上机前再评估是否切到 (a)。
- 注意:无论选哪个,**限幅必须与实机一致**(Ackermann 模式 ±0.6981 rad、
  蟹行 ±1.570 rad、最小转弯半径 0.8103 m),否则策略会依赖仿真里才有的大转向。

### C 层 — 工程细节(可以边做边定)

#### C1. 网格要不要精简

- 31 MB 的底盘网格(`ranger_base_link.STL`)每次启动都要加载。
- 补充事实:**碰撞体在转换时本来就会被 `collider_type: convex_hull` 替换**,
  所以碰撞精度不取决于原网格;抽稀只影响**外观保真**。
- (a) 不动;(b) 只对视觉网格抽稀(如减到 30–50%),碰撞体保持凸包。
- 我建议 (b),但需要你确认可以接受外观变化。

#### C2. 两份 description 副本

- 根目录 `<repo>/rangerboxcr10lidar_description/` 与 `agx/rangerboxcr10lidar_description/`
  是同一份模型(URDF md5 一致,已核)。
- 现在两份都在版本控制里(一份在 LocoManip_Lab,一份在 agx 子模块),**会各自漂移**。
- (a) 保留两份,文档里写明"isaac 侧以根目录那份为准";(b) 删掉 agx 内那份(但要改 agx 仓库);
  (c) 让根目录那份改成指向子模块的软链。
- 需要你定;我倾向 (a) + 文档说明,因为改 agx 会牵动另一个仓库。

#### C3. 车轮摩擦/滚阻的 sim-to-real

- URDF 没有 `<dynamics>`,场景摩擦是 `1.0/1.0`,而轮式机器人的侧滑与滚阻是
  sim-to-real 的最大误差源。
- (a) 先接受差距,靠域随机化覆盖;(b) 做一次真机辨识实验(直线加速/制动/原地旋转)后标定。
- 需要你定本期是否包含辨识(需要真机可动)。

#### C4. 训练规模

- GO2-PIPER 是 8192 环境 × 2500 轮 ≈ 64 分钟。本机器人自由度更多、质量更大(176 kg)、
  网格更重,吞吐未知。
- 我建议按流程:**阶段 2 先测吞吐 → 再定 `--num_envs`**(和上次一样,先跑 1024/4096/8192 对比)。
  这条只是通知你,不需要决策,除非你有别的偏好。

#### C5. "训练效果好"的判据

- 我默认沿用 GO2-PIPER 的指标(末端位置/姿态误差收敛到同量级:0.065 m / 0.095)
  加人工看视频。
- **你还有别的成功判据吗?** 例如"行进中末端抖动小于 X"、"能走到指定位置并够到目标"。
  这会影响我怎么设计评估脚本。

#### C6. 近期是否上机

- 若近期要上机,我会在**设计阶段**就把部署节点的清单纳入(关节名重映射、度/弧度换算、
  臂动作插值、`/cmd_vel` 看门狗、夹爪原始计数、无 IMU 替代),而不是留到最后补。
- 若不上机,阶段 5 只产出对照表文档。

---

## 10. 最小答复集(如果上面太长)

只回这五条我就能开工:

1. **A1**:能不能确认 CR10 真实限位?不能的话,是否同意取两组限位的交集兜底?
2. **A2**:Dobot 控制器的 TCP 偏移是多少?不知道的话,是否同意先用 `gripper_base_link`?
3. **B1**:底盘走方案 **B** 还是 **C**?走 B 的话,速度指令范围取保守(±0.5)还是硬件上限?`vy` 先不开可以吗?
4. **B3**:没有 IMU,`base_ang_vel`/`projected_gravity` 走"从 `/odom` 推算 + 仿真加噪声"可以吗?
5. **B4**:臂动作频率降到 10 Hz 可以吗?
