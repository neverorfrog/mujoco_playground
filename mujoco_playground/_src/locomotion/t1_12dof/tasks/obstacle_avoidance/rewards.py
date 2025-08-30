"""Reward functions for Booster T1 locomotion tasks"""

# TODO: should i use pose related rewards?

"""
Reward functions for Booster T1 ObstacleAvoidance task
"""

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src import gait
from mujoco_playground._src.collision import geoms_colliding
from .config import RewardConfig

if TYPE_CHECKING:  # pragma: no cover
    from .env import ObstacleAvoidance
    
class ObstacleAvoidanceRewards:
    """Collection of reward functions for T1 joystick locomotion tasks"""
    
    def __init__(self, env: 'ObstacleAvoidance'):
        self.env = env
        self.config = RewardConfig()
        
    def get(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics
        del done
        cmd = info["command"]
        lin_f = info["filtered_linvel"]
        ang_f = info["filtered_angvel"]

        return {
            # Tracking rewards.
            "tracking_lin_vel_x": self._reward_tracking_lin_vel_axis(0, cmd, lin_f),
            "tracking_lin_vel_y": self._reward_tracking_lin_vel_axis(1, cmd, lin_f),
            "tracking_ang_vel": self._reward_tracking_ang_vel(cmd, ang_f),
            "tracking_goal": self._reward_tracking_goal(data, info),
            
            # Base-related rewards.
            "lin_vel_z": self._cost_lin_vel_z(lin_f),
            "ang_vel_xy": self._cost_ang_vel_xy(ang_f),
            "orientation": self._cost_orientation(self.env.get_gravity(data)),
            "base_height": self._cost_base_height(data),
            
            # Energy related rewards
            "torque_tiredness": self._cost_torque_tiredness(data.actuator_force),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "power": self._cost_energy(data.qvel[6:], data.actuator_force),
            "dof_acc": self._cost_dof_acc(data.qacc[6:]),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
            
            # Feet related rewards.
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_air_time": self._reward_feet_air_time(
                info["feet_air_time"], first_contact, info["command"]
            ),
            "feet_distance": self._cost_feet_distance(data, info),
            "feet_swing": self._reward_feet_swing(info["phase"], contact),
            "feet_roll": self._cost_feet_roll(data),
            "feet_yaw_diff": self._cost_feet_yaw_diff(data),
            "feet_yaw_mean": self._cost_feet_yaw_mean(data),
            "collision": self._cost_collision(data),
            
            # Pose related rewards
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "root_acc": self._cost_root_acc(data),
            
            # Other rewards
            "survival": jp.array(1.0),
        }
        
    # Tracking rewards
    def _reward_tracking_lin_vel_axis(
        self, axis: int, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        """
        Axis wise linear velocity tracker (matches Isaac Gym x & y trackers). 
        Negative exp of the squared error.
        """
        err = jp.square(command[axis] - local_linvel[axis])
        return jp.exp(-err / self.config.tracking_sigma)  
    
    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        local_angvel: jax.Array,
    ) -> jax.Array:
        """Tracks angular velocity by doing a negative exponential of the squared error"""
        ang_vel_error = jp.square(commands[2] - local_angvel[2])
        return jp.exp(-ang_vel_error / self.config.tracking_sigma)

    def _reward_tracking_goal(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Tracks the goal position by doing a negative exponential of the squared error"""
        goal_error = jp.square(info["goal"] - data.qpos[:2])
        return jp.exp(-goal_error)[0]

    # Base related rewards
    def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
        """Penalty for z-axis linear velocity."""
        return jp.square(local_linvel[2])

    def _cost_ang_vel_xy(self, local_angvel) -> jax.Array:
        """Penalty for xy-plane angular velocity."""
        return jp.sum(jp.square(local_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        """Penalty for torso upwards orientation."""
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _cost_base_height(self, data: mjx.Data) -> jax.Array:
        """Penalty for deviating from base height."""
        h = data.qpos[2]
        return jp.square(h - self.config.base_height_target)
    
    # Energy related rewards.
    def _cost_torque_tiredness(self, torques: jax.Array) -> jax.Array:
        # Σ (τ / τ_max)²  – clipped at 1 so the term stays O(1)
        frac = jp.clip(jp.abs(torques) / self.env._torque_limits, 0.0, 1.0)
        return jp.sum(jp.square(frac))
    
    def _cost_torques(self, torques: jp.ndarray) -> jp.ndarray:
        """Guess what"""
        return jp.sum(jp.square(torques))
    
    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        """Penalty for action values changing too quickly."""
        del last_last_act  # Unused.
        c1 = jp.sum(jp.square(act - last_act))
        return c1
    
    def _cost_energy(self, qvel: jp.ndarray, qfrc_actuator: jp.ndarray) -> jp.ndarray:
        """Calculate energy cost based on positive power consumption."""
        power = qvel * qfrc_actuator
        # Only count positive power (energy being consumed/dissipated)
        return jp.sum(jp.where(power > 0.0, power, 0.0))
    
    def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qacc))

    def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qvel))
    
    # Feet related rewards
    def _cost_feet_slip(
        self,
        data: mjx.Data,
        contact: jp.ndarray,
        info: dict[str, Any],
    ) -> jp.ndarray:
        """Penalty for foot slipping. If the foot is in contact, there should be no xy velocity."""
        del info  # Unused here.
        v = data.sensordata[self.env._foot_linvel_sensor_adr]  # shape (2, 3)
        speed2 = jp.sum(jp.square(v), axis=-1)  # per‑foot |v|²
        return jp.sum(speed2 * contact)
    
    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
    ) -> jax.Array:
        
        cmd_norm = jp.linalg.norm(commands)
        air_time = (air_time - threshold_min) * first_contact
        air_time = jp.clip(air_time, max=threshold_max - threshold_min)
        reward = jp.sum(air_time)
        reward *= cmd_norm > 0.1  # No reward for zero commands.
        return reward
    
    def _cost_feet_distance(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        left_foot_pos = data.site_xpos[self.env._feet_site_id[0]]
        right_foot_pos = data.site_xpos[self.env._feet_site_id[1]]
        base_xmat = data.site_xmat[self.env._site_id]
        base_yaw = jp.arctan2(base_xmat[1, 0], base_xmat[0, 0])
        feet_distance = jp.abs(
            jp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1])
            - jp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        return jp.clip(0.2 - feet_distance, min=0.0, max=0.1)
    
    def _reward_feet_swing(
        self, phase: jp.ndarray, feet_contact: jp.ndarray
    ) -> jp.ndarray:
        gait = jp.fmod(phase[0] + jp.pi, 2 * jp.pi) / (2 * jp.pi)  # scalar ∈[0,1)
        half_window = 0.5 * self.config.swing_period

        left_swing = jp.abs(gait - 0.25) < half_window
        right_swing = jp.abs(gait - 0.75) < half_window

        # Reward when the corresponding foot is **not** in contact
        return (left_swing & ~feet_contact[0]) + (right_swing & ~feet_contact[1])
    
    def _cost_feet_roll(self, data: mjx.Data) -> jax.Array:
        """Penalty for feet roll angles (should be close to 0)."""
        roll, _ = self.env._feet_roll_yaw(data)
        return jp.sum(jp.square(roll))
    
    def _cost_feet_yaw_diff(self, data: mjx.Data) -> jax.Array:
        """Calculate squared cost based on yaw angle difference between feet."""
        _, yaw = self.env._feet_roll_yaw(data)
        diff = jp.fmod(yaw[1] - yaw[0] + jp.pi, 2 * jp.pi) - jp.pi
        return jp.square(diff)

    def _cost_feet_yaw_mean(self, data: mjx.Data) -> jax.Array:
        """
        Computes a cost penalty based on the difference between the robot's base yaw
        and the mean yaw angle of its feet. This encourages the robot to maintain
        alignment between its body and foot orientations.
        """
        _, feet_yaw = self.env._feet_roll_yaw(data)
        base_R = data.site_xmat[self.env._site_id]
        base_yaw = jp.arctan2(base_R[1, 0], base_R[0, 0])
        mean_yaw = jp.mean(feet_yaw)
        err = jp.fmod(base_yaw - mean_yaw + jp.pi, 2 * jp.pi) - jp.pi
        return jp.square(err)
    
    def _cost_collision(self, data: mjx.Data) -> jax.Array:
        """
            Checks if the left and right foot geometry boxes are colliding
            with each other, which would be undesirable behavior (feet shouldn't overlap).
        """
        return geoms_colliding(
            data, self.env._left_foot_box_geom_id, self.env._right_foot_box_geom_id
        )
    
    # Pose related rewards
    def _cost_root_acc(self, data: mjx.Data) -> jax.Array:
        # Root‑link 6‑D acceleration² (free‑joint entries of qacc)
        return jp.sum(jp.square(data.qacc[:6]))
    
    def _cost_joint_pos_limits(self, qpos: jp.ndarray) -> jp.ndarray:
        below = qpos < self.env._soft_lowers
        above = qpos > self.env._soft_uppers
        return jp.sum((below | above).astype(jp.float32))
    
    
        