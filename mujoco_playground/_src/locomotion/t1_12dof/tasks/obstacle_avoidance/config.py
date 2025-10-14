"""
Dataclass-based configuration for the Booster T1 joystick task.
"""

from dataclasses import dataclass, field, replace
from typing import Literal, Tuple, Dict, Any
import jax.numpy as jp
from enum import IntEnum

class Foot(IntEnum):
    LEFT = 0
    RIGHT = 1
@dataclass
class FootstepPlannerConfig:
    """
    Configuration for the footstep planner.
    """
    P: int = 1000
    """Number of timesteps to plan ahead. TODO: should be the same as episode length? """
    W: int  = 50
    """Number of steps in the preview window for ZMP trajectory generation (and reward heuristics)"""
    dt: float = 0.02
    """Time step duration for the planner. TODO: should be the same as episode step length? """
    Tp: float = P * dt
    """Total planning horizon in seconds."""
    max_steps: int = 50
    """Maximum steps to plan ahead. Should be >= Tp * step_frequency"""
    first_swing: Foot = Foot.LEFT
    """Which foot is the first swing foot"""
    step_width: float = 0.1
    """Lateral foot separation from the pelvis. Depends on the robot dimensions"""
    swing_percentage: float = 0.4
    """Percentage duration of the swing phase for each step."""
    peak_height: float = 0.05
    """Desired height of each foot at peak phase."""
    warmup_ds_factor: float = 0.6
    """Number of warmup steps for the double support phase"""
    theta_max: float = 0.25
    """Maximum foot rotation angle during the swing phase [rad]"""

@dataclass
class SceneConfig:
    scenario: Literal["A", "B", "C"] = field(default="Random")
    bins: int = 39
    bin_size: float = 0.25
    abs_gamma: float = 0.91
    goal: jp.ndarray = field(init=False)
    obstacles: jp.ndarray = field(init=False)
    num_obstacles: int = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)
    origin: jp.ndarray = field(init=False)
    
    def __post_init__(self):
        self.width = self.bins * self.bin_size
        self.height = self.bins * self.bin_size
        self.origin = jp.array([ -self.width / 2, -self.height / 2 ])
        
        if self.scenario == "A":
            self.obstacles = jp.array([[6, -1], [6, 0]])
            self.goal = jp.array([14, 0])
        elif self.scenario == "B":
            self.obstacles = jp.array([[5, -3], [5, -2], [5, -1], [10, 3], [10, 2], [10, 1]]) 
            self.goal = jp.array([12, 2])
        else:
            self.obstacles = jp.array([[6, 3], [6, 2], [6, 1], [6, 0], [6, -1], [6, -2], [6, -3], [6, -4], [6, -5], [6, -6], [5, 3], [4, 3], [3, 3], [2, 3], [5, -6], [4, -6], [3, -6], [2, -6]]) # SCENARIO C
            self.goal = jp.array([8, 4])
        
        self.num_obstacles = self.obstacles.shape[0]
        
    def map_to_world(self, map_pos: jp.ndarray, center: bool = True) -> jp.ndarray:
        offset = 0.5 if center else 0.0  # fraction of bin_size
        world_x = (map_pos[0] + offset) * self.bin_size
        world_y = (map_pos[1] + offset) * self.bin_size
        return jp.array([world_x, world_y])
    
    @property
    def obstacle_positions(self):
        """Convert obstacle indices to world positions at tile centers."""
        positions = []
        for idx in self.obstacles:
            world_pos = self.map_to_world(idx, center=True)
            world_x, world_y = world_pos[0], world_pos[1]
            positions.append(jp.array([world_x, world_y, 0.0]))
        return jp.array(positions)
    
    @property
    def goal_position(self):
        """Convert goal index to world position at tile center."""
        world_pos = self.map_to_world(self.goal, center=True)
        world_x, world_y = world_pos[0], world_pos[1]
        return jp.array([world_x, world_y, 0.0])
    
@dataclass
class RewardScales:
    """Reward scaling factors for different reward components."""
    # Velocity tracking
    tracking_lin_vel_x: float = 1.5
    tracking_lin_vel_y: float = 1.5
    tracking_ang_vel: float = 2.0
    
    # Abstract Map
    cost_collision: float = -10.0
    reward_map: float = 1.0
    goal_distance: float = 0.01
    goal_proximity: float = 1.0
    torso_velocity_alignment: float = 0.0
    
    # Feet Trajectories
    feet_swing: float = 3.0
    feet_air_time: float = 2.0
    
    # Base related rewards
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.2
    orientation: float = -5.0
    base_height: float = -20.0
    cost_linvel_rate: float = -0.02
    cost_command_rate: float = 0.0
    
    # Energy related rewards
    torque_tiredness: float = -0.5e-2
    torques: float = -1.0e-3
    action_rate: float = -0.5
    power: float = -1.0e-3
    dof_acc: float = -1.0e-7
    dof_vel: float = -1.0e-4
    
    # Feet kinematics
    feet_slip: float = -0.1
    feet_distance: float = -15.0
    feet_yaw_diff: float = -1.0
    feet_yaw_mean: float = -1.0
    feet_roll: float = -10.0
    feet_collision: float = -10.0
    
    # Other rewards
    survival: float = 0.05
    root_acc: float = -1.0e-4
    dof_pos_limits: float = -1.0
    episode_failed: float = 1.0
    
    
@dataclass
class CurriculumConfig:
    ramp_steps: int = 10_000
    tracking_lin_vel_x: float = 2.0
    tracking_lin_vel_y: float = 2.0
    tracking_ang_vel: float = 2.0
    velocity_direction_alignment: float = 0.0
    torso_velocity_alignment: float = 1.0
    planner_com_x: float = 0.0
    planner_com_y: float = 0.0


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    scales: RewardScales = field(default_factory=RewardScales)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    tracking_sigma: float = 0.5
    base_height_target: float = 0.68
    swing_period: float = 0.2
    min_command_magnitude: float = 0.05
    

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
    episode_length: int = 1000
    action_repeat: int = 1
    action_scale: float = 1.0
    history_len: int = 1
    soft_joint_pos_limit_factor: float = 0.95
    
    # Sub-configurations
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    push_config: PushConfig = field(default_factory=PushConfig)
    planner_config: FootstepPlannerConfig = field(default_factory=FootstepPlannerConfig)
    
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
