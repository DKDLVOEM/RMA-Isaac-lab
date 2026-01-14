from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def track_position_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(cmd - joint_pos), dim=1)
    return torch.exp(-error / 0.25)

def track_velocity_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    target_vel = cmd_term.get_command_vel()
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(target_vel - joint_vel), dim=1)
    return torch.exp(-error / 0.5)
