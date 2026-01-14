# NEW add: aggressiveness scalar g (add all code)



from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

# NEW add: aggressiveness scalar g
# class AggressivenessAction(ActionTerm):
#     """Reads 1-D action and stores scalar g into env for later use (AYRO)."""

#     # def __init__(self, cfg, env):
#     def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:

#         super().__init__(cfg, env)

#         # 스케일/오프셋 
#         # [-1,1] -> [0,1] 로 맵핑할 scale / offset

#         self.scale = 0.5
#         self.offset = 0.5

#         # g를 저장해 둘 버퍼를 env 쪽에 만들어 둔다.
#         # (이미 있다면 생략 가능)
#         if not hasattr(env, "ayro_g"):
#             # [num_envs] shape 텐서
#             env.ayro_g = torch.zeros(env.num_envs, device=env.device)

#         self._env = env

#     def apply(self, env, action: torch.Tensor):
#         """Called every step with sliced action tensor of shape [num_envs, 1]."""
#         # [-1,1] → 실수 g 로 스케일링
#         g = self.scale * action.squeeze(-1) + self.offset
#         # 0~1로 클램핑(원하면)
#         g = torch.clamp(g, 0.0, 1.0)

#         # env 버퍼에 저장 → path_command 에서 사용
#         self._env.ayro_g = g



class AggressivenessAction(ActionTerm):
    r"""1-D aggressiveness scalar g를 받아서 [0,1] 범위로 변환하고
    env.ayro_g 버퍼에 저장하는 액션 term.

    - policy output: [-1, 1]
    - 내부 매핑: g = 0.5 * action + 0.5  →  [0,1]
    """

    cfg: actions_cfg.AggressivenessActionCfg

    def __init__(self, cfg: actions_cfg.AggressivenessActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env = env

        # --------------------------
        # 1) 액션 버퍼 준비
        # --------------------------
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # --------------------------
        # 2) scale / offset 설정
        #    - cfg.scale / cfg.offset 을 쓰되,
        #    - 안 넣었으면 기본값 0.5, 0.5로 [-1,1]→[0,1]
        # --------------------------
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        else:
            # 기본값: [-1,1] → 0.5 * a + 0.5
            self._scale = 0.5

        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        else:
            self._offset = 0.5

        # (clip 필요하면 cfg.clip 추가해서 쓰면 됨. 지금은 [0,1] 클램프만 사용)

        # --------------------------
        # 3) env 쪽 g 버퍼 준비
        # --------------------------
        if not hasattr(env, "ayro_g"):
            env.ayro_g = torch.zeros(self.num_envs, device=self.device)

    # ==========================
    #  Properties (필수)
    # ==========================
    @property
    def action_dim(self) -> int:
        # 이 term이 차지하는 액션 차원: g 1개
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # 여기서는 이미 [0,1]로 매핑된 g
        return self._processed_actions

    # ==========================
    #  Operations (필수)
    # ==========================
    def process_actions(self, actions: torch.Tensor):
        """ActionManager가 step마다 넘겨주는 슬라이스 액션.

        actions: [num_envs, 1] in [-1, 1]
        """
        # # 1) raw 저장
        # self._raw_actions[:] = actions
        # print("_raw_actions ayro: ", self._raw_actions)

        # # 2) affine 변환: g = scale * a + offset # TODO
        # g = self._raw_actions * self._scale + self._offset  # [-1,1] → [0,1] 기본
        # # NEW add: aggressiveness scalar g
        # # print("is this working?")
        # # print("actions? ", actions)
        # # print("g scaled: ", g)
        # # print("_scale: ", self._scale)
        # # print("_offset: ", self._offset)

        # # 3) [0,1]로 클램핑
        # g = torch.clamp(g, 0.0, 1.0)


        # NEW add: aggressiveness scalar g
        self._raw_actions[:] = actions                 # actor output (혹시라도 [-1,1] 벗어나도 방어)
        # g_raw = torch.tanh(self._raw_actions)          # 확실하게 [-1,1]로 자르기
        # g = g_raw * self._scale + self._offset  # [-1,1] → [0,1] 기본
        g = self._raw_actions * self._scale + self._offset  # [-1,1] → [0,1] 기본
        g = torch.clamp(g, 0.0, 1.0)                   # 수치 안전용



        # 4) processed 버퍼에 저장
        self._processed_actions[:] = g





        # 5) env 전역 버퍼에 [num_envs] 형태로 저장
        #    path_command._update_command() 에서 self.env.ayro_g로 사용
        self._env.ayro_g[:] = g.squeeze(-1)
        # print("g in action process: ", self._env.ayro_g[:])









    def apply_actions(self):
        """JointAction과 달리 실제 조인트 명령은 없고,
        g는 path command 쪽에서만 사용되므로 여기서는 아무것도 안 해도 됨.
        """
        pass

    def reset(self, env_ids=None) -> None:
        """에피소드 리셋 시 g도 0으로 초기화(선택 사항이지만 깔끔하게)."""
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._env.ayro_g.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._env.ayro_g[env_ids] = 0.0



# class JointPositionAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as position commands."""

#     cfg: actions_cfg.JointPositionActionCfg
#     """The configuration of the action term."""

#     def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint positions as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

#     def apply_actions(self):
#         # set position targets
#         self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)