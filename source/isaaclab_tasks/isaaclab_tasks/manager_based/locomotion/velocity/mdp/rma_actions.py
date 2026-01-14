from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class RMAJointPositionAction(joint_actions.JointPositionAction):
    cfg: "RMAJointPositionActionCfg"

    def __init__(self, cfg: "RMAJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._delay_steps = max(int(cfg.delay_steps), 0)
        self._motor_strength_key = cfg.motor_strength_key
        if self._delay_steps > 0:
            self._action_history = torch.zeros(
                self.num_envs, self._delay_steps + 1, self.action_dim, device=self.device
            )
        else:
            self._action_history = None

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        raw = self._raw_actions

        if self._action_history is not None:
            self._action_history = self._action_history.roll(1, dims=1)
            self._action_history[:, 0] = raw
            raw = self._action_history[:, -1]

        processed = raw * self._scale + self._offset
        if self.cfg.clip is not None:
            processed = torch.clamp(processed, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        motor_strength = self._resolve_motor_strength()
        if motor_strength is not None:
            processed = processed * motor_strength

        self._processed_actions = processed

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self._action_history is not None and env_ids is not None:
            self._action_history[env_ids] = 0.0

    def _resolve_motor_strength(self) -> torch.Tensor | None:
        buffers = getattr(self._env, "_rma_buffers", None)
        if buffers is None:
            return None
        strength = getattr(buffers, self._motor_strength_key, None)
        if strength is None:
            return None
        if isinstance(self._joint_ids, slice):
            return strength
        return strength[:, self._joint_ids]


@configclass
class RMAJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = RMAJointPositionAction
    delay_steps: int = 0
    motor_strength_key: str = "motor_strength"
