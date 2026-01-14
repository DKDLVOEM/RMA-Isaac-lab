from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


@dataclass
class RmaBuffers:
    friction: torch.Tensor
    mass_scale: torch.Tensor
    com_offset: torch.Tensor
    motor_strength: torch.Tensor


def _get_rma_buffers(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> RmaBuffers:
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    num_joints = asset.num_joints
    device = env.device
    if not hasattr(env, "_rma_buffers"):
        env._rma_buffers = RmaBuffers(
            friction=torch.ones(num_envs, 1, device=device),
            mass_scale=torch.ones(num_envs, 1, device=device),
            com_offset=torch.zeros(num_envs, 3, device=device),
            motor_strength=torch.ones(num_envs, num_joints, device=device),
        )
        return env._rma_buffers

    buffers: RmaBuffers = env._rma_buffers
    if buffers.motor_strength.shape != (num_envs, num_joints):
        buffers.motor_strength = torch.ones(num_envs, num_joints, device=device)
    if buffers.friction.shape != (num_envs, 1):
        buffers.friction = torch.ones(num_envs, 1, device=device)
    if buffers.mass_scale.shape != (num_envs, 1):
        buffers.mass_scale = torch.ones(num_envs, 1, device=device)
    if buffers.com_offset.shape != (num_envs, 3):
        buffers.com_offset = torch.zeros(num_envs, 3, device=device)
    return buffers


def rma_priv_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _get_rma_buffers(env, asset_cfg).friction


def rma_priv_mass_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _get_rma_buffers(env, asset_cfg).mass_scale


def rma_priv_com_offset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _get_rma_buffers(env, asset_cfg).com_offset


def rma_priv_motor_strength(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _get_rma_buffers(env, asset_cfg).motor_strength


def rma_feet_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (contacts > threshold).float()


class rma_randomize_rigid_body_material(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"RMA material randomization not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "RMA material randomization failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            self.num_shapes_per_body = None

        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        materials = self.asset.root_physx_view.get_material_properties()

        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids] = material_samples[:]

        self.asset.root_physx_view.set_material_properties(materials, env_ids)

        buffers = _get_rma_buffers(env, asset_cfg)
        friction = material_samples[:, :, 0].mean(dim=1, keepdim=True)
        buffers.friction[env_ids.to(env.device)] = friction.to(env.device)


class rma_randomize_rigid_body_mass(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1e-6,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        masses = self.asset.root_physx_view.get_masses()
        masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[env_ids[:, None], body_ids].clone()

        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
        )
        masses = torch.clamp(masses, min=min_mass)
        self.asset.root_physx_view.set_masses(masses, env_ids)

        if recompute_inertia:
            ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                inertias[env_ids[:, None], body_ids] = (
                    self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
                )
            else:
                inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
            self.asset.root_physx_view.set_inertias(inertias, env_ids)

        buffers = _get_rma_buffers(env, asset_cfg)
        ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
        mass_scale = ratios.mean(dim=1, keepdim=True)
        buffers.mass_scale[env_ids.to(env.device)] = mass_scale.to(env.device)


class rma_randomize_rigid_body_com(ManagerTermBase):
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

        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

        coms = self.asset.root_physx_view.get_coms().clone()
        coms[env_ids[:, None], body_ids, :3] += rand_samples
        self.asset.root_physx_view.set_coms(coms, env_ids)

        buffers = _get_rma_buffers(env, asset_cfg)
        buffers.com_offset[env_ids.to(env.device)] = rand_samples[:, 0].to(env.device)


class rma_randomize_motor_strength(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        strength_range: tuple[float, float],
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        else:
            env_ids = env_ids.to(env.device)

        buffers = _get_rma_buffers(env, asset_cfg)
        strength = math_utils.sample_uniform(
            strength_range[0], strength_range[1], (len(env_ids), buffers.motor_strength.shape[1]), device=env.device
        )
        buffers.motor_strength[env_ids] = strength
