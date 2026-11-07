# Copyright (c) 2025-2026, Junjie Zhu.
# Apache-2.0 header omitted here? (this script uses the project header in the repo standard)

from __future__ import annotations
import json, os
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import gymnasium as gym, torch, LeggedManip_Lab.tasks
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import RangerCr10TeleopSimpleCollisionEnvCfg
cfg=RangerCr10TeleopSimpleCollisionEnvCfg(); cfg.scene.num_envs=1
cfg.commands.ee_pose.debug_vis=False; cfg.commands.ee_pose.resampling_time_range=(1e6,1e6); cfg.commands.ee_pose.curriculum_enabled=False
cfg.events.reset_robot_joints=None; cfg.events.randomize_actuator_gains=None; cfg.actions.joint_pos.use_default_offset=False
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions=True
env=gym.make('RANGER-CR10-TELEOP-SIMPLECOLL',cfg=cfg); u=env.unwrapped; robot=u.scene['robot']; scale=cfg.actions.joint_pos.scale
idx=[robot.joint_names.index(f'cr10_joint{i}') for i in range(1,7)]
resets=[]; base_s=[]; joint_s=[]
for r in range(5):
 env.reset(); start=robot.data.joint_pos[:,idx].clone()
 act=torch.zeros(u.num_envs,u.action_manager.total_action_dim,device=u.device); act[:,2:]=start/scale
 for _ in range(10): env.step(act)
 resets.append(dict(reset=r,arm_drift=(robot.data.joint_pos[:,idx]-start).abs().max().item(),base_z=robot.data.root_pos_w[0,2].item()))
# base motion with arm target held
env.reset(); hold=robot.data.joint_pos[:,idx].clone(); base0=robot.data.root_pos_w.clone()
act=torch.zeros(u.num_envs,u.action_manager.total_action_dim,device=u.device); act[:,0]=0.2; act[:,2:]=hold/scale
for _ in range(40): env.step(act)
base1=robot.data.root_pos_w
# box local offset under base is fixed; verify base transform consistency for the two box centrelines
base_s.append(dict(base_displacement=(base1-base0)[0].tolist(),base_z=base1[0,2].item(),arm_drift=(robot.data.joint_pos[:,idx]-hold).abs().max().item()))
out=dict(reset=resets,base_motion=base_s,note='box_link colliders are merged into base_link, so they follow the base transform; local offsets are authored in the flat USD')
os.makedirs('outputs/stage413/b/simple_collision_validation',exist_ok=True)
json.dump(out,open('outputs/stage413/b/simple_collision_validation/reset_base.json','w'),indent=1)
print(json.dumps(out,indent=1),flush=True); env.close()
