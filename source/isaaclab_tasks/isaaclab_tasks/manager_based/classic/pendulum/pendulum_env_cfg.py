import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.pendulum.mdp as mdp

@configclass
class PendulumSceneCfg(InteractiveSceneCfg):
    """Configuration for the pendulum scene."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.join(os.path.dirname(__file__), "assets/pendulum.urdf"),
            fix_base=True,
            joint_drive=None,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"revolute_joint": 0.0},
        ),
        actuators={
            "servo": ImplicitActuatorCfg(
                joint_names_expr=["revolute_joint"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=0.0, # Controlled by action term or PD
                damping=0.0,   # Viscous friction (randomized later)
            ),
        },
    )

    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pendulum_link",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, -0.5)),
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class CommandsCfg:
    trajectory = mdp.TrajectoryCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        trajectory_type="sine",
    )

@configclass
class ActionsCfg:
    # Custom Action Term with Delay and LPF
    joint_pos = mdp.ActuatorPositionActionCfg(
        asset_name="robot", 
        joint_names=["revolute_joint"], 
        scale=1.0, 
        use_default_offset=False,
        # delay_steps=2,      # Actuator Delay
        # lpf_alpha=0.8,      # Actuator LPF (1.0 = no filter)
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "trajectory"})
        imu_quat = ObsTerm(func=mdp.imu_orientation, params={"sensor_cfg": SceneEntityCfg("imu_sensor")})
        imu_ang_vel = ObsTerm(func=mdp.imu_angular_velocity, params={"sensor_cfg": SceneEntityCfg("imu_sensor")})
        imu_lin_acc = ObsTerm(func=mdp.imu_linear_acceleration, params={"sensor_cfg": SceneEntityCfg("imu_sensor")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for domain randomization events."""

    # # 1. Link Mass Randomization
    # randomize_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="pendulum_link"),
    #         "mass_distribution_params": (0.8, 1.2), # Scale 0.8x to 1.2x
    #         "operation": "scale",
    #     },
    # )

    # # 2. Link COM Randomization
    # randomize_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="pendulum_link"),
    #         "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.05, 0.05)}, # Meters offset
    #     },
    # )

    # 3. Joint Friction & Damping (Viscous Friction) Randomization
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="revolute_joint"),
            "friction_distribution_params": (0.0, 0.1), # Static/Dynamic Friction (Coulomb)
            "operation": "abs", # Set absolute value
            "distribution": "uniform",
        },
    )
    
    # 4. Armature (Simulated via Link Inertia Scaling)
    # Since direct armature randomization is complex, we scale inertia to simulate rotor inertia effects.
    randomize_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass, # This function also scales inertia if recompute_inertia=True (default)
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pendulum_link"),
            "mass_distribution_params": (1.0, 1.0), # Keep mass same
            "operation": "scale",
            # Note: To strictly randomize ONLY inertia (armature effect) without mass, 
            # one would need a custom function. But scaling mass/inertia together is a good proxy for "heavier/lighter link".
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,  # scale 대신 offset 사용
        mode="reset",
        params={"position_range": (-0.5, 0.5), "velocity_range": (-0.5, 0.5)},
    )

    # push_link = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(2.0, 4.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="pendulum_link"),
    #         "force_range": (-50.0, 50.0),
    #         "torque_range": (-5.0, 5.0),
    #     },
    # )

@configclass
class RewardsCfg:
    track_pos = RewTerm(
        func=mdp.track_position_error,
        # weight=100.0,
        weight=10.0,
        params={"command_name": "trajectory", "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute_joint"])}
    )
    track_vel = RewTerm(
        func=mdp.track_velocity_error,
        weight=1.0,
        params={"command_name": "trajectory", "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute_joint"])}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class PendulumEnvCfg(ManagerBasedRLEnvCfg):
    scene: PendulumSceneCfg = PendulumSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
