from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class randomize_rigid_body_com(ManagerTermBase):
    """Randomize the Center of Mass (COM) of rigid bodies."""
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # Resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # Sample random COM offsets
        range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

        # Apply offsets to default COMs
        # Note: We assume we want to perturb the *current* or *default* COM. 
        # Here we get current, which is fine for reset.
        coms = self.asset.root_physx_view.get_coms().clone()
        rand_samples = rand_samples.to(coms.device)
        
        # Apply to specific bodies for specific envs
        # coms shape: (num_envs, num_bodies, 7)
        # rand_samples shape: (num_envs, 1, 3) -> broadcast to bodies if needed, or (num_envs, num_selected_bodies, 3)
        
        # Simple implementation: apply same offset logic to all selected bodies
        for i, body_idx in enumerate(body_ids):
             coms[env_ids, body_idx, :3] += rand_samples[:, 0, :]

        self.asset.root_physx_view.set_coms(coms, env_ids)
