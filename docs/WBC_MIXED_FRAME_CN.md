# WBC 混合坐标系说明

WBC（Whole-Body Control）模式下的末端执行器位姿指令采用**混合坐标系**，即位置 XY、位置 Z、姿态分别使用不同的参考坐标系：

## 坐标系定义

| 指令分量 | 参考坐标系 | 说明 |
|---------|-----------|------|
| `pos_x` | **link0 坐标系** | 末端相对于机身 link0 的前后偏移 |
| `pos_y` | **link0 坐标系** | 末端相对于机身 link0 的左右偏移 |
| `pos_z` | **世界坐标系** | 末端的绝对世界高度 |
| 姿态 (roll/pitch/yaw) | **link0 坐标系** | 末端朝向，由目标方向自动解算 |

## 设计动机

### 为什么 XY 用 link0 坐标系？

- **机身运动无关性**：机械臂末端相对于机身的 XY 指令不受机器人行走、转向的影响。无论机身朝向哪里，"向前 0.5m"始终在机身正前方。
- **训练稳定性**：策略只需学习末端与机身的相对关系，无需额外补偿机身位姿变化。

### 为什么 Z 用世界坐标系？

- **绝对参照物**：地面、桌面等高度是固定的，世界坐标系下的 Z 指令可以直接对应到"离地 0.6m"这样的物理目标。
- **简化推理**：策略无需根据机身高度波动来推算目标高度。

### 为什么姿态自动解算？

- 姿态的 pitch 和 yaw 由 link0 到目标位置的方向向量自动计算（确保末端"朝向"目标点），roll 随机采样。
- 所有欧拉角限制在 `[-π/4, π/4]` 范围内，避免不可达姿态。


## 相关代码

- 指令生成：[`pose_command_wbc.py`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/pose_command_wbc.py)
- 位姿误差计算：[`rewards.py`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/rewards.py) 中的 `position_command_error_exp`
