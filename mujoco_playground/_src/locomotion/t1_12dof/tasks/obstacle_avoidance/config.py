"""
Dataclass-based configuration for the Booster T1 joystick task.
"""

from dataclasses import dataclass, field, asdict, replace
from typing import List, Tuple, Dict, Any
import jax.numpy as jp

@dataclass
class SceneConfig:
    obstacle_positions: jp.array = field(default_factory=lambda: 
        jp.array([(1.75, 0.25), (-0.02, 0.1), (0.12, 0.17)]))
    goal_position: jp.array = field(default_factory=lambda: jp.array([2.5, 0.0, 0.0]))

@dataclass
class RewardScales:
    """Reward scaling factors for different reward components."""
    # Tracking related rewards
    tracking_lin_vel_x: float = 1.0
    tracking_lin_vel_y: float = 1.0
    tracking_ang_vel: float = 2.0
    cost_to_goal_distance: float = 0.0
    cost_to_goal_orientation: float = 0.0
    reward_abstract_map: float = 0.1

    # Base related rewards
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.2
    orientation: float = -5.0
    base_height: float = -20.0
    
    # Energy related rewards
    torque_tiredness: float = -0.5e-2
    torques: float = -1.0e-3
    action_rate: float = -0.5
    power: float = -1.0e-3
    dof_acc: float = -1.0e-7
    dof_vel: float = -1.0e-4
    
    # Feet related rewards
    feet_slip: float = -0.1
    feet_air_time: float = 2.0
    feet_distance: float = -10.0
    feet_yaw_diff: float = -1.0
    feet_yaw_mean: float = -1.0
    feet_roll: float = -10.0
    feet_swing: float = 3.0
    feet_collision: float = -10.0
    
    # Other rewards
    survival: float = 0.25
    root_acc: float = -1.0e-4
    dof_pos_limits: float = -1.0
    goal_reached: float = 1.0
    
@dataclass
class CurriculumConfig:
    ramp_steps: int = 30000
    tracking_lin_vel_x: float = 1.0
    tracking_lin_vel_y: float = 1.0
    cost_to_goal_distance: float = 0.0
    cost_to_goal_orientation: float = 0.0


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    scales: RewardScales = field(default_factory=RewardScales)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    tracking_sigma: float = 0.25
    base_height_target: float = 0.68
    swing_period: float = 0.2
    

@dataclass
class NoiseScales:
    """Noise scaling factors for different sensor types."""
    joint_pos: float = 0.03
    joint_vel: float = 1.5
    gravity: float = 0.05
    linvel: float = 0.1
    gyro: float = 0.2


@dataclass
class NoiseConfig:
    """Configuration for sensor noise."""
    level: float = 1.0  # Set to 0.0 to disable noise
    scales: NoiseScales = field(default_factory=NoiseScales)


@dataclass
class PushConfig:
    """Configuration for external push disturbances."""
    enable: bool = False
    interval_range: Tuple[float, float] = (5.0, 10.0)
    magnitude_range: Tuple[float, float] = (0.1, 1.0)


@dataclass
class ObstacleAvoidanceConfig:
    """Main configuration for the obstacle avoidance task."""
    # Simulation parameters
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    episode_length: int = 500
    action_repeat: int = 1
    action_scale: float = 1.0
    history_len: int = 1
    soft_joint_pos_limit_factor: float = 0.95
    
    # Sub-configurations
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    push_config: PushConfig = field(default_factory=PushConfig)
    
    # Command velocity ranges
    lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: Tuple[float, float] = (-0.8, 0.8)
    ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)

    def to_dict(self) -> dict:
        """Convert to dictionary format for compatibility with ml_collections."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ObstacleAvoidanceConfig":
        # Rebuild nested dataclasses if present.
        if "scene_config" in config_dict:
            s = config_dict["scene_config"]
            if isinstance(s, dict):
                config_dict["scene_config"] = SceneConfig(**s)
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
                if "curriculum" in r and isinstance(r["curriculum"], dict):
                    r["curriculum"] = CurriculumConfig(**r["curriculum"])
                config_dict["reward_config"] = RewardConfig(**r)
        if "push_config" in config_dict:
            p = config_dict["push_config"]
            if isinstance(p, dict):
                config_dict["push_config"] = PushConfig(**p)
        return cls(**config_dict)

    def update(self, **kwargs) -> "ObstacleAvoidanceConfig":
        return replace(self, **kwargs)
