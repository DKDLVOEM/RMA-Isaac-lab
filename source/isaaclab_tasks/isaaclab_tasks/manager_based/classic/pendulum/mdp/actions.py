from __future__ import annotations

import torch
from typing import Sequence

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

class ActuatorPositionAction(JointPositionAction):
    """Joint position action with simulated delay and low-pass filter."""

    cfg: ActuatorPositionActionCfg

    def __init__(self, cfg: ActuatorPositionActionCfg, env):
        super().__init__(cfg, env)
        self._delay_steps = max(int(cfg.delay_steps), 0)
        self._lpf_alpha = torch.tensor(cfg.lpf_alpha, device=self.device).clamp(0.0, 1.0)
        
        # Buffer for delay
        if self._delay_steps > 0:
            self._action_history = torch.zeros(
                self.num_envs, self._delay_steps + 1, self.action_dim, device=self.device
            )
        else:
            self._action_history = None
            
        # Buffer for LPF state (previous output)
        self._prev_processed_action = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

    def process_actions(self, actions: torch.Tensor):
        # 1. Store raw actions
        self._raw_actions[:] = actions
        raw = self._raw_actions.clone()

        # 2. Apply Delay
        if self._action_history is not None:
            # Shift history: drop oldest, add newest at index 0
            self._action_history = self._action_history.roll(1, dims=1)
            self._action_history[:, 0] = raw
            # The delayed action is at the end of the buffer
            delayed_action = self._action_history[:, -1]
        else:
            delayed_action = raw

        # 3. Apply Scale & Offset (Affine Transform)
        target_pos = delayed_action * self._scale + self._offset

        # 4. Apply Clip
        if self.cfg.clip is not None:
            target_pos = torch.clamp(target_pos, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        # 5. Apply LPF
        # y[t] = (1 - alpha) * y[t-1] + alpha * x[t]
        # Note: If alpha=1.0, y[t] = x[t] (No filter)
        filtered_action = (1.0 - self._lpf_alpha) * self._prev_processed_action + self._lpf_alpha * target_pos
        
        # Update state
        self._prev_processed_action[:] = filtered_action
        self._processed_actions[:] = filtered_action

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            
        if self._action_history is not None:
            self._action_history[env_ids] = 0.0
        
        # Reset LPF state to current joint positions to avoid jumps
        # We assume the robot starts at a valid state.
        # Ideally, we would read the current joint pos, but 0.0 is safe for relative/absolute if reset to 0.
        self._prev_processed_action[env_ids] = 0.0 

@configclass
class ActuatorPositionActionCfg(JointPositionActionCfg):
    """Configuration for the actuator position action term with delay and LPF."""
    class_type: type[ActionTerm] = ActuatorPositionAction
    
    delay_steps: int = 0
    """Number of steps to delay the action."""
    
    lpf_alpha: float = 1.0
    """Low-pass filter coefficient (0.0 to 1.0). 1.0 means no filtering (pass-through).
    y[t] = (1 - alpha) * y[t-1] + alpha * x[t]
    """
