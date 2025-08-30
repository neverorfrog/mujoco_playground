"""Dataclass-based configuration for the Booster T1 joystick task.

Mirrors the structure used in the 12-DoF variant for consistency and
stronger type checking. Provides helpers to convert to/from plain dicts
for ml_collections compatibility.
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Tuple, Dict, Any

@dataclass
class RewardScales:
    # Tracking related rewards.
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5

    # Base related rewards.
    lin_vel_z: float = 0.0
    ang_vel_xy: float = -0.15
    orientation: float = -1.0
    base_height: float = 0.0

    # Energy related rewards.
    torques: float = 0.0
    action_rate: float = 0.0
    energy: float = -1.0e-3
    dof_acc: float = -1.0e-7
    dof_vel: float = -1.0e-4

    # Feet related rewards.
    feet_clearance: float = 0.0
    feet_air_time: float = 2.0
    feet_slip: float = -0.25
    feet_height: float = 0.0
    feet_phase: float = 1.0

    # Other rewards.
    stand_still: float = 0.0
    alive: float = 0.25
    termination: float = 0.0

    # Pose related rewards.
    joint_deviation_knee: float = -0.1
    joint_deviation_hip: float = -0.1
    dof_pos_limits: float = -1.0
    pose: float = -1.0
    feet_distance: float = -1.0
    collision: float = -1.0


@dataclass
class RewardConfig:
    scales: RewardScales = field(default_factory=RewardScales)
    tracking_sigma: float = 0.25
    max_foot_height: float = 0.12
    base_height_target: float = 0.665


@dataclass
class NoiseScales:
    joint_pos: float = 0.03
    joint_vel: float = 1.5
    gravity: float = 0.05
    linvel: float = 0.1
    gyro: float = 0.2


@dataclass
class NoiseConfig:
    level: float = 1.0  # Set to 0.0 to disable noise.
    scales: NoiseScales = field(default_factory=NoiseScales)


@dataclass
class PushConfig:
    enable: bool = False
    interval_range: Tuple[float, float] = (5.0, 10.0)
    magnitude_range: Tuple[float, float] = (0.1, 1.0)


@dataclass
class JoystickConfig:
    # Simulation parameters.
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    episode_length: int = 1000
    action_repeat: int = 1
    action_scale: float = 1.0
    history_len: int = 1
    soft_joint_pos_limit_factor: float = 0.95

    # Sub-configurations.
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    push_config: PushConfig = field(default_factory=PushConfig)

    # Command velocity ranges.
    lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: Tuple[float, float] = (-0.8, 0.8)
    ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "JoystickConfig":
        # Rebuild nested dataclasses if present.
        if "noise_config" in config_dict:
            n = config_dict["noise_config"]
            if isinstance(n, dict):
                if "scales" in n and isinstance(n["scales"], dict):
                    n["scales"] = NoiseScales(**n["scales"])
                config_dict["noise_config"] = NoiseConfig(**n)
        if "reward_config" in config_dict:
            r = config_dict["reward_config"]
            if isinstance(r, dict):
                if "scales" in r and isinstance(r["scales"], dict):
                    r["scales"] = RewardScales(**r["scales"])
                config_dict["reward_config"] = RewardConfig(**r)
        if "push_config" in config_dict:
            p = config_dict["push_config"]
            if isinstance(p, dict):
                config_dict["push_config"] = PushConfig(**p)
        return cls(**config_dict)

    def update(self, **kwargs) -> "JoystickConfig":
        return replace(self, **kwargs)
