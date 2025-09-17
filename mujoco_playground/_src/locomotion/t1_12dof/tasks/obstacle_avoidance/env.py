# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Navigation task for Booster T1."""

import copy
import re
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

import xml.etree.ElementTree as ET

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.t1_12dof import base as t1_base
from mujoco_playground._src.locomotion.t1_12dof import t1_constants as consts
from .rewards import ObstacleAvoidanceRewards
from .config import ObstacleAvoidanceConfig
from .abstract_map import AbstractMap

def _to_config_dict(obj):
    if isinstance(obj, dict):
        return config_dict.ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_config_dict(v) for v in obj)
    return obj

def default_config() -> config_dict.ConfigDict:
    return _to_config_dict(ObstacleAvoidanceConfig().to_dict())

class ObstacleAvoidance(t1_base.T1LowDimEnv):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        xml_path = consts.task_to_xml(task).as_posix()
        xml_content = self._setup_scene(xml_path, config)
        
        super().__init__(
            xml_path=xml_path,
            xml_content=xml_content,
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()
        self.rewards = ObstacleAvoidanceRewards(self)
        self.abstract_map = AbstractMap(bins=15, bin_size=0.5)
        
    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

        # Note: First joint is freejoint.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        hip_indices = []
        hip_joint_names = ["Hip_Roll", "Hip_Yaw"]
        for side in ["Left", "Right"]:
            for joint_name in hip_joint_names:
                hip_indices.append(
                    self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7
                )
        self._hip_indices = jp.array(hip_indices)

        knee_indices = []
        for side in ["Left", "Right"]:
            knee_indices.append(self._mj_model.joint(f"{side}_Knee_Pitch").qposadr - 7)
        self._knee_indices = jp.array(knee_indices)

        # fmt: off
        self._weights = jp.array([
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Left leg.
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Right leg.
        ])
        # fmt: on

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._left_feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS]
        )
        self._right_feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._left_foot_box_geom_id = self._mj_model.geom("left_foot").id
        self._right_foot_box_geom_id = self._mj_model.geom("right_foot").id

        force_range = self._mj_model.actuator_forcerange  # (nact, 2)
        force_limited = self._mj_model.actuator_forcelimited  # (nact,)
        hi = jp.array(force_range[:, 1])
        #   unlimited → treat as very large so the penalty goes to zero
        self._torque_limits = jp.where(force_limited, hi, jp.full_like(hi, 1e6))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-0.2, maxval=0.2)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(key, (12,), minval=0.8, maxval=1.2)
        )

        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Phase, freq=U(1.25, 1.75)
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.75)
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        phase = jp.array([0, jp.pi])

        rng, goal_rng = jax.random.split(rng)
        goal = jp.array(self._config.scene_config.goal_position)
        goal += jax.random.uniform(goal_rng, shape=goal.shape, minval=-0.1, maxval=0.1)
        goal = jp.array(goal[:2])
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.get_command(goal, cmd_rng)
        
        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)
        
        info = {
            "abs_goal": jp.array([goal[0], goal[1]]), # ENV
            "rel_goal": jp.array([goal[0], goal[1]]) - data.qpos[:2],
            "global_step": jp.array(0, dtype=jp.int32),
            "abstract_map": self.abstract_map.reset(goal, np.array([])),
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Phase related.
            "phase_dt": phase_dt,
            "phase": phase,
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "filtered_linvel": jp.zeros(3),
            "filtered_angvel": jp.zeros(3),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        left_feet_contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._left_feet_geom_id
            ]
        )
        right_feet_contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._right_feet_geom_id
            ]
        )
        contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])

        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        linvel = self.get_local_linvel(data)
        state.info["filtered_linvel"] = (
            linvel * 1.0 + state.info["filtered_linvel"] * 0.0
        )
        angvel = self.get_gyro(data)
        state.info["filtered_angvel"] = (
            angvel * 1.0 + state.info["filtered_angvel"] * 0.0
        )

        left_feet_contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._left_feet_geom_id
            ]
        )
        right_feet_contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._right_feet_geom_id
            ]
        )
        contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        done, goal_reached = self._get_termination(data, state.info)

        rewards = self.rewards.get(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards["goal_reached"] = jp.where(goal_reached, jp.array(200.0), jp.array(0.0))
        
        # curriculum_weights = self._get_curriculum_weights(state.info)       
        for k, v in rewards.items():
            base_scale = self._config.reward_config.scales[k]
            # if(k in curriculum_weights.keys()):
                # rewards[k] *= curriculum_weights[k]
            # else:
            rewards[k] *= base_scale

        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["push"] = push
        state.info["step"] += 1
        state.info["global_step"] += 1
        state.info["push_step"] += 1
        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["phase"] = jp.where(
            jp.linalg.norm(state.info["command"]) > 0.01,
            state.info["phase"],
            jp.ones(2) * jp.pi,
        )
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["step"] = jp.where(
            done,
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        goal_termination = jp.linalg.norm(info["rel_goal"]) < 0.25 # ENV
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | goal_termination, goal_termination

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        cos = jp.cos(info["phase"])
        sin = jp.sin(info["phase"])
        phase = jp.concatenate([cos, sin])

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )
        
        # ENV: including feedback to generate command
        # info["command"] is the desired (x, y, yaw) velocity
        # convert goal to be relative to robot's position
        info["rel_goal"] = info["abs_goal"] - data.qpos[:2]

        state = jp.hstack(
            [
                noisy_linvel, # 3
                noisy_gyro,  # 3
                noisy_gravity,  # 3
                info["abs_goal"],  # ENV 2
                noisy_joint_angles - self._default_pose,  # 12
                noisy_joint_vel,  # 12
                info["last_act"],  # 12
                phase,  # 4
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        privileged_state = jp.hstack(
            [
                state,
                info["command"],  # 3
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_pose,
                joint_vel,
                root_height,  # 1
                data.actuator_force,
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }


    def get_command(self, goal: jax.Array, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        goal_dist = jp.linalg.norm(goal) + 1e-8
        goal_dir = goal / goal_dist
        return jp.array([goal_dir[0], goal_dir[1], 0.0])

        
    def _setup_scene(self, xml_path: str, config: dict) -> str:
        """Setup the Mujoco environment by modifying the XML file"""
        with open(xml_path, "r") as file:
            xml = file.read()

        # Overwrite the target and obstacle position in the XML file
        tree = copy.deepcopy(ET.ElementTree(ET.fromstring(xml)))
        root = tree.getroot()
        obstacle_pattern = re.compile(r"obstacle_\d+")
        goal_pattern = re.compile(r"goal")

        obstacle_positions = config.scene_config.obstacle_positions
        goal_position = config.scene_config.goal_position

        for geom in root.findall(".//geom"):
            if obstacle_pattern.match(geom.get("name", "")):
                print(f"Found obstacle: {geom.get('name')}")
                obstacle_index = int(geom.get("name").replace("obstacle_", ""))
                try:
                    new_obstacle_pos = np.array(
                        [*obstacle_positions[obstacle_index], 0.0]
                    )
                    obstacle_positions[obstacle_index] = new_obstacle_pos
                    geom.set("pos", " ".join(map(str, new_obstacle_pos)))
                    print(
                        f"Set new position for {geom.get('name')}: {new_obstacle_pos}"
                    )
                except IndexError:
                    new_obstacle_pos = np.array([2.0, 0.0, 0.0])
                    geom.set("pos", " ".join(map(str, new_obstacle_pos)))

        for site in root.findall(".//site"):
            if goal_pattern.match(site.get("name", "")):
                site.set("pos", " ".join(map(str, goal_position)))

        return ET.tostring(root, encoding="unicode")
    
    def _get_curriculum_weights(self, info: Dict[str, Any]):
        alpha = jp.clip(
            info["global_step"] / self._config.reward_config.curriculum["ramp_steps"], 0.0, 1.0
        )
        
        # Weights departing from normal config and ending in curriculum config
        tracking_lin_vel_x = self._config.reward_config.scales["tracking_lin_vel_x"] - alpha * (
            self._config.reward_config.curriculum["tracking_lin_vel_x"] - self._config.reward_config.scales["tracking_lin_vel_x"]
        )
        tracking_lin_vel_y = self._config.reward_config.scales["tracking_lin_vel_y"] - alpha * (
            self._config.reward_config.curriculum["tracking_lin_vel_y"] - self._config.reward_config.scales["tracking_lin_vel_y"]
        )
        
        # Weights departing from curriculum config and ending in normal config
        cost_to_goal_distance = self._config.reward_config.curriculum["cost_to_goal_distance"] + alpha * (
            self._config.reward_config.scales["cost_to_goal_distance"] - self._config.reward_config.curriculum["cost_to_goal_distance"]
        )
        cost_to_goal_orientation = self._config.reward_config.curriculum["cost_to_goal_orientation"] + alpha * (
            self._config.reward_config.scales["cost_to_goal_orientation"] - self._config.reward_config.curriculum["cost_to_goal_orientation"]
        )
        
        return {
            "tracking_lin_vel_x": tracking_lin_vel_x,
            "tracking_lin_vel_y": tracking_lin_vel_y,
            "cost_to_goal_distance": -cost_to_goal_distance,
            "cost_to_goal_orientation": -cost_to_goal_orientation
        }
        


    # ----- feet kinematics ----------------------------------------------------
    def _feet_site_xmat(self, data: mjx.Data) -> jax.Array:
        """Return the (2,3,3) rotation matrices of the foot *sites*."""
        return data.site_xmat[self._feet_site_id].reshape(2, 3, 3)

    def _feet_roll_yaw(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Return (roll, yaw) angles of both feet, radians in [‑π, π]."""
        R = self._feet_site_xmat(data)
        # roll  = atan2(R32, R33)           (x‑rotation)
        # yaw   = atan2(R21, R11)           (z‑rotation)
        roll = jp.arctan2(R[:, 2, 1], R[:, 2, 2])
        yaw = jp.arctan2(R[:, 1, 0], R[:, 0, 0])
        return roll, yaw