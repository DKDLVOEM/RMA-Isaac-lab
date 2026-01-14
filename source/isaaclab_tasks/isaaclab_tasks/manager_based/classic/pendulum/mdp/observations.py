from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Imu

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def imu_orientation(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Orientation of the IMU sensor in world frame (quaternion)."""
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.quat_w

def imu_angular_velocity(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Angular velocity of the IMU sensor in sensor frame."""
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ang_vel_b

def imu_linear_acceleration(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linear acceleration of the IMU sensor in sensor frame."""
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.lin_acc_b