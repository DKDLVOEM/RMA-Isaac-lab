# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlRmaActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic config for RMA teacher (privileged encoder + action MLP)."""

    class_name: str = "RmaActorCritic"

    base_obs_dim: int = MISSING
    priv_obs_dim: int = MISSING
    geom_obs_dim: int = 0
    geom_num_slices: int = 0
    geom_obs_start: int | None = None
    geom_reverse: bool = True
    prop_latent_dim: int = 8
    geom_latent_dim: int = 1


@configclass
class AnymalDRoughRMAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Teacher policy with privileged observations for RMA."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough_rma_teacher"
    obs_groups = {"policy": ["policy", "privileged"], "critic": ["policy", "privileged"]}
    policy = RslRlRmaActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        base_obs_dim=48,
        priv_obs_dim=204,
        geom_obs_dim=160,
        geom_num_slices=1,
        geom_obs_start=0,
        geom_reverse=False,
        prop_latent_dim=8,
        geom_latent_dim=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalDFlatRMAPPORunnerCfg(AnymalDRoughRMAPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_rma_teacher"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        self.policy.priv_obs_dim = 44
        self.policy.geom_obs_dim = 0
        self.policy.geom_num_slices = 0
        self.policy.geom_latent_dim = 0
        self.policy.geom_obs_start = None
