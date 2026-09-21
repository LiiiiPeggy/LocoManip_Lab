# WBC Mixed Coordinate Frame

The WBC (Whole-Body Control) mode uses a **mixed coordinate frame** for end-effector pose commands — position XY, position Z, and orientation each use different reference frames:

## Coordinate Frame Definitions

| Component | Reference Frame | Description |
|-----------|----------------|-------------|
| `pos_x` | **link0 frame** | Forward/backward offset from the robot body |
| `pos_y` | **link0 frame** | Left/right offset from the robot body |
| `pos_z` | **World frame** | Absolute height in the world |
| Orientation (roll/pitch/yaw) | **link0 frame** | End-effector orientation, auto-computed from target direction |

## Design Rationale

### Why XY in link0 frame?

- **Body-motion invariance**: The XY command relative to the body is unaffected by robot locomotion or turning. "0.5m forward" always means forward relative to the robot.
- **Training stability**: The policy only needs to learn the relative relationship between the end-effector and the body, without compensating for body pose changes.

### Why Z in world frame?

- **Absolute references**: The ground, tables, etc. have fixed heights. A world-frame Z command directly maps to physical targets like "0.6m above ground".
- **Simpler inference**: The policy doesn't need to infer target height from fluctuating body height.

### Why auto-computed orientation?

- The pitch and yaw are automatically calculated from the direction vector from link0 to the target position, ensuring the end-effector naturally points toward the target. Roll is randomly sampled.
- All Euler angles are clamped to `[-π/4, π/4]` to avoid unreachable orientations.


## Related Code

- Command generator: [`pose_command_wbc.py`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/pose_command_wbc.py)
- Pose error computation: `position_command_error_exp` in [`rewards.py`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/rewards.py)
