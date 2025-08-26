"""Configuration dataclasses for T1 locomotion tasks."""

from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class RewardScales:
    """Reward scaling factors for different reward components."""
    # Tracking related rewards
    tracking_lin_vel_x: float = 1.0
    tracking_lin_vel_y: float = 1.0
    tracking_ang_vel: float = 2.0
    
    # Base related rewards
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.2
    orientation: float = -5.0
    base_height: float = -20.0
    
    # Energy related rewards
    torque_tiredness: float = -0.5e-2
    torques: float = -1.0e-4
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
    
    # Other rewards
    survival: float = 0.25
    root_acc: float = -1.0e-4
    dof_pos_limits: float = -1.0
    collision: float = -10.0
    
@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    scales: RewardScales = field(default_factory=RewardScales)
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
    enable: bool = True
    interval_range: Tuple[float, float] = (5.0, 10.0)
    magnitude_range: Tuple[float, float] = (0.1, 1.0)


@dataclass
class JoystickConfig:
    """Main configuration for the Joystick locomotion task."""
    # Simulation parameters
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    episode_length: int = 500
    action_repeat: int = 1
    action_scale: float = 1.0
    history_len: int = 1
    soft_joint_pos_limit_factor: float = 0.95
    
    # Sub-configurations
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
    def from_dict(cls, config_dict: dict) -> 'JoystickConfig':
        """Create from dictionary (useful for loading from config files)."""
        # Handle nested configs
        if 'noise_config' in config_dict:
            noise_dict = config_dict['noise_config']
            if 'scales' in noise_dict:
                noise_dict['scales'] = NoiseScales(**noise_dict['scales'])
            config_dict['noise_config'] = NoiseConfig(**noise_dict)
        
        if 'reward_config' in config_dict:
            reward_dict = config_dict['reward_config']
            if 'scales' in reward_dict:
                reward_dict['scales'] = RewardScales(**reward_dict['scales'])
            config_dict['reward_config'] = RewardConfig(**reward_dict)
        
        if 'push_config' in config_dict:
            config_dict['push_config'] = PushConfig(**config_dict['push_config'])
        
        return cls(**config_dict)

    def update(self, **kwargs) -> 'JoystickConfig':
        """Return a new config with updated parameters."""
        from dataclasses import replace
        return replace(self, **kwargs)