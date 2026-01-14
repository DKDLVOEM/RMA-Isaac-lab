# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class RslRlRmaStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Student/teacher config for RMA distillation (latent imitation)."""

    class_name: str = "RmaStudentTeacher"

    base_obs_dim: int = MISSING
    history_obs_dim: int = MISSING
    history_length: int = MISSING
    geom_obs_dim: int = 0
    geom_num_slices: int = 0
    geom_obs_start: int | None = None
    geom_reverse: bool = True
    prop_latent_dim: int = 8
    geom_latent_dim: int = 1
    history_obs_first: bool = False
    freeze_student_actor: bool = True
    deterministic_actions: bool = True
    copy_teacher_action_mlp: bool = True


@configclass
class AnymalDRoughRMADistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Student distillation with history observations for RMA."""

    num_steps_per_env = 120
    max_iterations = 300
    save_interval = 50
    experiment_name = "anymal_d_rough_rma_student"
    obs_groups = {"policy": ["policy", "history"], "teacher": ["policy", "privileged"]}
    policy = RslRlRmaStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_hidden_dims=[128, 128, 128],
        teacher_hidden_dims=[128, 128, 128],
        activation="elu",
        base_obs_dim=48,
        history_obs_dim=52,
        history_length=50,
        geom_obs_dim=160,
        geom_num_slices=1,
        geom_obs_start=0,
        geom_reverse=False,
        prop_latent_dim=8,
        geom_latent_dim=1,
        history_obs_first=False,
        freeze_student_actor=True,
        deterministic_actions=True,
        copy_teacher_action_mlp=True,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class AnymalDFlatRMADistillationRunnerCfg(AnymalDRoughRMADistillationRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_rma_student"
        self.policy.geom_obs_dim = 0
        self.policy.geom_num_slices = 0
        self.policy.geom_latent_dim = 0
        self.policy.geom_obs_start = None
