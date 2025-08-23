"""Reward functions for Booster T1 joystick locomotion task.

This mirrors the refactor done for the 12-DoF variant where reward logic
is isolated from the environment class for clarity and reuse.
"""

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src import gait
from mujoco_playground._src.collision import geoms_colliding

if TYPE_CHECKING:  # pragma: no cover
    from .joystick import Joystick


class JoystickRewards:
    """Collection of reward / cost terms for the T1 joystick task.

    The get(...) method returns unscaled terms; the environment applies the
    scaling factors from config.reward_config.scales and dt.
    """

    def __init__(self, env: "Joystick"):
        self.env = env

    # ------------------------------------------------------------------ public
    def get(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],  # unused presently (kept for parity)
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            # Tracking rewards.
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], info["filtered_linvel"]
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], info["filtered_angvel"]
            ),
            # Base-related rewards.
            "lin_vel_z": self._cost_lin_vel_z(info["filtered_linvel"]),
            "ang_vel_xy": self._cost_ang_vel_xy(info["filtered_angvel"]),
            "orientation": self._cost_orientation(self.env.get_gravity(data)),
            "base_height": self._cost_base_height(data, info),
            # Energy related rewards.
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "dof_acc": self._cost_dof_acc(data.qacc[6:]),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
            # Feet related rewards.
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_clearance": self._cost_feet_clearance(data, info),
            "feet_height": self._cost_feet_height(
                info["swing_peak"], first_contact, info
            ),
            "feet_air_time": self._reward_feet_air_time(
                info["feet_air_time"], first_contact, info["command"]
            ),
            "feet_phase": self._reward_feet_phase(
                data,
                info["phase"],
                self.env._config.reward_config.max_foot_height,
                info["command"],
            ),
            # Other rewards.
            "alive": self._reward_alive(),
            "termination": self._cost_termination(done),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
            "collision": self._cost_collision(data),
            # Pose related rewards.
            "joint_deviation_hip": self._cost_joint_deviation_hip(
                data.qpos[7:], info["command"]
            ),
            "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "pose": self._cost_pose(data.qpos[7:]),
            "feet_distance": self._cost_feet_distance(data, info),
        }

    # ---------------------------------------------------------- tracking terms
    def _reward_tracking_lin_vel(
        self, commands: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_linvel[:2]))
        return jp.exp(
            -lin_vel_error / self.env._config.reward_config.tracking_sigma
        )

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, local_angvel: jax.Array
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - local_angvel[2])
        return jp.exp(
            -ang_vel_error / self.env._config.reward_config.tracking_sigma
        )

    # ------------------------------------------------------------- base costs
    def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
        return jp.square(local_linvel[2])

    def _cost_ang_vel_xy(self, local_angvel) -> jax.Array:
        return jp.sum(jp.square(local_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _cost_base_height(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        base_height = data.qpos[2]
        return jp.square(
            base_height - self.env._config.reward_config.base_height_target
        )

    # ----------------------------------------------------------- energy costs
    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(torques))

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qvel * qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))

    def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qacc))

    def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qvel))

    # -------------------------------------------------------------- other misc
    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        # replicate original logic (non‑squared violation sum)
        out_of_limits = -jp.clip(qpos - self.env._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self.env._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _cost_stand_still(self, commands: jax.Array, qpos: jax.Array) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self.env._default_pose)) * (cmd_norm < 0.1)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _reward_alive(self) -> jax.Array:
        return jp.array(1.0)

    def _cost_collision(self, data: mjx.Data) -> jax.Array:
        return geoms_colliding(
            data, self.env._left_foot_box_geom_id, self.env._right_foot_box_geom_id
        )

    # ------------------------------------------------------------ pose related
    def _cost_joint_deviation_hip(self, qpos: jax.Array, cmd: jax.Array) -> jax.Array:
        cost = jp.sum(
            jp.abs(qpos[self.env._hip_indices] - self.env._default_pose[self.env._hip_indices])
        )
        cost *= jp.abs(cmd[1]) > 0.1
        return cost

    def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(
            jp.abs(qpos[self.env._knee_indices] - self.env._default_pose[self.env._knee_indices])
        )

    def _cost_pose(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qpos - self.env._default_pose) * self.env._weights)

    # ------------------------------------------------------------ feet related
    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        body_vel = self.env.get_global_linvel(data)[:2]
        return jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)

    def _cost_feet_clearance(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        feet_vel = data.sensordata[self.env._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self.env._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self.env._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(
        self, swing_peak: jax.Array, first_contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        error = swing_peak / self.env._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact)

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

    def _reward_feet_phase(
        self,
        data: mjx.Data,
        phase: jax.Array,
        foot_height: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        del commands  # Unused.
        foot_pos = data.site_xpos[self.env._feet_site_id]
        foot_z = foot_pos[..., -1]
        rz = gait.get_rz(phase, swing_height=foot_height)
        error = jp.sum(jp.square(foot_z - rz))
        return jp.exp(-error / 0.01)

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
