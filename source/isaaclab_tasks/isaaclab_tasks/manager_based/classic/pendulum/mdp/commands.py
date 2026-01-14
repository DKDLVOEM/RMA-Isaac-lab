from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TrajectoryCommand(CommandTerm):
    """Command generator for trajectory tracking (Sine wave or Step input)."""

    cfg: TrajectoryCommandCfg

    def __init__(self, cfg: TrajectoryCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 1, device=self.device) # Target Position
        self.command_vel = torch.zeros(self.num_envs, 1, device=self.device) # Target Velocity
        
        self.amplitudes = torch.zeros(self.num_envs, device=self.device)
        self.frequencies = torch.zeros(self.num_envs, device=self.device)
        self.phases = torch.zeros(self.num_envs, device=self.device)
        self.step_targets = torch.zeros(self.num_envs, device=self.device)
        
        # Metric 계산을 위한 로봇 객체 가져오기
        self._robot = env.scene[cfg.asset_name]
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, command_dim)."""
        return self._command

    # def _resample_command(self, env_ids: Sequence[int]):
    #     self.amplitudes[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.amplitude_range)
    #     self.frequencies[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.frequency_range)
    #     self.phases[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-3.14, 3.14)
    #     self.step_targets[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-1.5, 1.5)

    def _resample_command(self, env_ids: Sequence[int]):
        self.amplitudes[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(0.5, 0.5)
        self.frequencies[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(0.5, 0.5)
        self.phases[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 0.0)
        self.step_targets[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(0.7, 0.7)


    def _update_command(self):
        current_time = self._env.sim.current_time
        
        if self.cfg.trajectory_type == "sine":
            t = current_time + self.phases
            omega = 2 * torch.pi * self.frequencies
            self._command[:, 0] = self.amplitudes * torch.sin(omega * t)
            self.command_vel[:, 0] = self.amplitudes * omega * torch.cos(omega * t)

            
        elif self.cfg.trajectory_type == "step":
            # self._command[:, 0] = self.step_targets
            self._command[:, 0] = 0.5
            self.command_vel[:, 0] = 0.0

    def _update_metrics(self):
        """Update the metrics based on the current state."""
        # 현재 관절 위치 가져오기 (0번 인덱스가 revolute joint)
        current_pos = self._robot.data.joint_pos[:, 0]
        # 오차 누적 (reset 시 자동으로 평균이 기록되고 0으로 초기화됨)
        self.metrics["position_error"] += torch.abs(self._command[:, 0] - current_pos)
    
    def get_command_vel(self) -> torch.Tensor:
        return self.command_vel

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics."""
        # Resolve env_ids
        if env_ids is None:
            env_ids = slice(None)
        
        # Calculate average position error in degrees
        # Get accumulated error (sum of absolute errors in radians)
        acc_error = self.metrics["position_error"][env_ids]
        # Get episode length (number of steps)
        ep_len = self._env.episode_length_buf[env_ids].float()
        # Avoid division by zero
        ep_len = torch.where(ep_len > 0.0, ep_len, torch.ones_like(ep_len))
        
        # Average error per step in radians -> convert to degrees
        avg_error_deg = (acc_error / ep_len) * (180.0 / torch.pi)
        
        # Prepare log dict
        extras = {
            "position_error_deg": torch.mean(avg_error_deg).item()
        }
        
        # Reset the metric buffer
        self.metrics["position_error"][env_ids] = 0.0
        
        # Reset command counter and resample (logic from base CommandTerm)
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        
        return extras

@configclass
class TrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the trajectory command generator."""
    class_type: type[CommandTerm] = TrajectoryCommand
    asset_name: str = "robot"
    trajectory_type: str = "sine"  # "sine" or "step"
    amplitude_range: tuple[float, float] = (0.5, 1.5)  # rad
    frequency_range: tuple[float, float] = (0.5, 2.0)  # Hz
