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

from random import choice
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.t1_12dof import base as t1_base
from mujoco_playground._src.locomotion.t1_12dof import t1_constants as consts
from .rewards import ObstacleAvoidanceRewards
from .config import ObstacleAvoidanceConfig, SceneConfig
from .map import Map
from .planner import FootstepPlanner
from .lqr import LipDynamics, preview_control

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
        self.randomize_scenario = config.scene_config.scenario == "Random"
        self.rewards = ObstacleAvoidanceRewards(self)
        self.lip = LipDynamics(N=75, dt=config.ctrl_dt, zc=config.reward_config.base_height_target)
        
        if self.randomize_scenario:
            self.scenarios = ["A", "B", "C"]
            self.maps = {}
            self.goals = {}
            self.obstacle_geom_ids = {}
            self.planners = {}
            
            # Find scenario with most obstacles to use for unified XML
            max_obstacles = 0
            max_scenario = None
            for scenario in self.scenarios:
                scene_cfg = SceneConfig(scenario=scenario)
                if scene_cfg.num_obstacles > max_obstacles:
                    max_obstacles = scene_cfg.num_obstacles
                    max_scenario = scenario
            self.max_obstacles = max_obstacles
            
            # Create unified XML with all possible obstacles
            unified_scene_cfg = SceneConfig(scenario=max_scenario)
            unified_map = Map(unified_scene_cfg)
            xml_content = unified_map._xml
            self.padded_obstacles = {}
                
            for scenario in self.scenarios:
                scene_cfg = SceneConfig(scenario=scenario)
                map_instance = Map(scene_cfg)
                self.maps[scenario] = map_instance
                self.goals[scenario] = jp.array(scene_cfg.goal_position[:2])
                planner_instance = FootstepPlanner(map_instance)
                planner_instance.config.dt = config.ctrl_dt
                self.planners[scenario] = planner_instance
                
                # Pad obstacles to max_obstacles size
                obstacles = jp.array(map_instance.obstacles)  # Shape: (num_obstacles, 3)
                num_obstacles = obstacles.shape[0]
                if num_obstacles < max_obstacles:
                    # Pad with zeros (or dummy values far away)
                    padding = jp.zeros((max_obstacles - num_obstacles, 3))
                    padded = jp.concatenate([obstacles, padding], axis=0)
                else:
                    padded = obstacles
                self.padded_obstacles[scenario] = padded
                
            self.current_scenario = self.scenarios[0]
            self.scene_cfg = SceneConfig(scenario=self.current_scenario)
            self.map = self.maps[self.current_scenario]
            self.goal = self.goals[self.current_scenario]
            
        else:
            self.scenarios = []
            self.current_scenario = config.scene_config.scenario
            self.scene_cfg = SceneConfig(scenario=self.current_scenario)
            self.map = Map(self.scene_cfg)
            self.goal = jp.array(self.scene_cfg.goal_position[:2])
            xml_content = self.map._xml
            self.planner = FootstepPlanner(self.map)
            self.planner.config.dt = config.ctrl_dt
            self.max_obstacles = self.scene_cfg.num_obstacles
        
        super().__init__(
            xml_path=xml_path,
            xml_content=xml_content,
            config=config,
            config_overrides=config_overrides,
        )
    
        jax.debug.print("RANDOMIZE SCENARIO: {}", self.randomize_scenario)
        jax.debug.print("GOAL POSITION: {}", self.scene_cfg.goal)
        jax.debug.print("OBSTACLES: {}", self.scene_cfg.obstacles)
        self._post_init()

        
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
        
        if self.randomize_scenario:
            for scenario in self.scenarios:
                scene_cfg = SceneConfig(scenario=scenario)
                geom_ids = []
                for i in range(scene_cfg.num_obstacles):
                    geom_ids.append(self._mj_model.geom(f"obstacle_{i}").id)
                self.obstacle_geom_ids[scenario] = jp.array(geom_ids)
            self._obstacle_geom_ids = self.obstacle_geom_ids[self.current_scenario]
        else:
            self._obstacle_geom_ids = []
            for i in range(self.scene_cfg.num_obstacles):
                self._obstacle_geom_ids.append(self._mj_model.geom(f"obstacle_{i}").id)
            self._obstacle_geom_ids = jp.array(self._obstacle_geom_ids)
        
        force_range = self._mj_model.actuator_forcerange  # (nact, 2)
        force_limited = self._mj_model.actuator_forcelimited  # (nact,)
        hi = jp.array(force_range[:, 1])
        #   unlimited → treat as very large so the penalty goes to zero
        self._torque_limits = jp.where(force_limited, hi, jp.full_like(hi, 1e6))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Phase, freq=U(1.3, 1.5)
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (), minval=1.3, maxval=1.5)
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        phase = jp.array([0.0, jp.pi])
        
        # x=+U(-0.1, 0.1), y=+U(-0.1, 0.1), yaw=U(-0.5, 0.5).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1) # CONFIG
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-0.5, maxval=0.5)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # qpos[7:]=*U(0.9, 1.1)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(key, (12,), minval=0.9, maxval=1.1)
        )
        
        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)
        
        
        # =============== Scenario Selection (JAX-compatible) ============
        rng, scenario_rng = jax.random.split(rng)
        scenario_idx = jax.lax.cond(
            self.randomize_scenario,
            lambda rng: jax.random.randint(rng, (), 0, len(self.scenarios)),
            lambda rng: 0,
            scenario_rng
        )

        # --------- Initial feet poses
        left_foot_pose = data.site_xpos[self._feet_site_id[0]]
        right_foot_pose = data.site_xpos[self._feet_site_id[1]]
        left_foot_xy = left_foot_pose[:2]   # Shape: (2,) - [x, y]
        right_foot_xy = right_foot_pose[:2] # Shape: (2,) - [x, y]
        left_foot_pose = jp.concatenate([left_foot_xy, jp.array([0.0])])   # (3,)
        right_foot_pose = jp.concatenate([right_foot_xy, jp.array([0.0])]) # (3,)
        
        # Select scenario components - branch on self.randomize_scenario at Python level
        # since it's a constant known at trace time
        if self.randomize_scenario:
            # Use lax.switch to select from pre-created scenarios
            selected_goal = jax.lax.switch(
                scenario_idx,
                [lambda g=self.goals[s]: g for s in self.scenarios],
            )
            
            selected_obstacles = jax.lax.switch(
                scenario_idx,
                [lambda obs=jp.array(self.maps[s].obstacles): obs for s in self.scenarios],
            )
            
            # Pre-extract JAX arrays from maps
            selected_map_array = jax.lax.switch(
                scenario_idx,
                [lambda m=self.maps[s]: m.get_map() for s in self.scenarios],
            )
            
            selected_gradient = jax.lax.switch(
                scenario_idx,
                [lambda m=self.maps[s]: m.get_gradient() for s in self.scenarios],
            )
            
            selected_command = jax.lax.switch(
                scenario_idx,
                [lambda m=self.maps[s]: m.get_command(data.qpos[:2], 0.0) for s in self.scenarios],
            )
            
            fs_plan = jax.lax.switch(
                scenario_idx,
                [lambda p=self.planners[s]: p.plan(
                    left_foot_pose=left_foot_pose,
                    right_foot_pose=right_foot_pose,
                    step_frequency=2 * gait_freq,
                    start_time=0.0
                ) for s in self.scenarios],
            )
            
            zmp_traj = jax.lax.switch(
                scenario_idx,
                [lambda p=self.planners[s]: p.compute_zmp_trajectory(fs_plan) for s in self.scenarios],
            )
        else:
            # Fixed scenario - no switching needed
            selected_goal = self.goal
            selected_obstacles = jp.array(self.map.obstacles)
            selected_map_array = self.map.get_map()
            selected_gradient = self.map.get_gradient()
            selected_command = self.map.get_command(data.qpos[:2], 0.0)
            
                        
            fs_plan = self.planner.plan(
                left_foot_pose=left_foot_pose,
                right_foot_pose=right_foot_pose,
                step_frequency=2 * gait_freq,
                start_time=0.0
            )
            zmp_traj = self.planner.compute_zmp_trajectory(fs_plan)
            

        # ======== Footstep planning and zmp trajectory generation ========
        current_com_pos = data.subtree_com[self._torso_body_id][:2]
        current_com_vel = data.subtree_linvel[self._torso_body_id][:2]
        current_com_acc = jp.array([0.0, 0.0])
        com_traj = preview_control(self.lip, zmp_traj, current_com_pos, current_com_vel, current_com_acc)

        info = {
            # Map
            "abs_goal": selected_goal, # ENV
            "rel_goal": selected_goal - data.qpos[:2],
            "obstacles": jp.array(selected_obstacles),
            "global_step": jp.array(0, dtype=jp.int32),
            "map": selected_map_array,
            "gradient": selected_gradient,
            "previous_com": data.subtree_com[self._torso_body_id],
            "cumulative_distance_to_goal": 0.0,
            "scenario_idx": scenario_idx,
            # Footstep plan
            "plan_timestep": jp.array(0, dtype=jp.int32),
            "plan_time": 0.0,
            "swing_foot_ids": fs_plan.swing_foot_ids,
            "start_poses": fs_plan.start_poses,
            "end_poses": fs_plan.end_poses,
            "support_poses": fs_plan.support_poses,
            "start_times": fs_plan.start_times,
            "ds_start_times": fs_plan.ds_start_times,
            "end_times": fs_plan.end_times,
            "num_steps": fs_plan.num_steps,
            # COM Reference
            "ref_com_x": com_traj.x_positions,
            "ref_com_y": com_traj.y_positions,
            "ref_com_vel_x": com_traj.x_velocities,
            "ref_com_vel_y": com_traj.y_velocities,
            # Other
            "rng": rng,
            "step": 0,
            "command": selected_command,
            "last_command": selected_command,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "torques": jp.zeros(self.mjx_model.nu),
            "last_torques": jp.zeros(self.mjx_model.nu),
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
            "last_linvel": jp.zeros(3),
            "filtered_angvel": jp.zeros(3),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())
        metrics["goal_reached"] = jp.zeros((), dtype=jp.float32)
        metrics["fallen"] = jp.zeros((), dtype=jp.float32)
        metrics["timeout"] = jp.zeros((), dtype=jp.float32)
        metrics["success"] = jp.zeros((), dtype=jp.float32)
        metrics["max_steps_reached"] = jp.zeros((), dtype=jp.float32)
        metrics["cumulative_distance_to_goal"] = jp.zeros((), dtype=jp.float32)
        metrics["final_distance_to_goal"] = jp.zeros((), dtype=jp.float32)
        
        # gait quality metrics
        metrics["gait_avg_power"] = jp.zeros(())
        metrics["gait_total_power"] = jp.zeros(())
        metrics["gait_torque_smoothness"] = jp.zeros(())
        metrics["gait_base_vel_variance"] = jp.zeros(())
        metrics["gait_base_ang_vel_norm"] = jp.zeros(())
        metrics["gait_trunk_tilt"] = jp.zeros(())
        metrics["root_height"] = jp.zeros(())

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

        state.info["last_linvel"] = state.info["filtered_linvel"]
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
        
        # Get command from abstract map
        torso_R = data.site_xmat[self._site_id]
        robot_yaw = jp.arctan2(torso_R[1, 0], torso_R[0, 0])
        state.info["last_command"] = state.info["command"]
        if self.randomize_scenario:
            state.info["command"] = jax.lax.switch(
                state.info["scenario_idx"],
                [lambda m=self.maps[s]: m.get_command(data.qpos[:2], robot_yaw) for s in self.scenarios]
            )
        else:
            state.info["command"] = self.map.get_command(data.qpos[:2], robot_yaw)
            
        # Goal
        state.info["rel_goal"] = state.info["abs_goal"] - data.qpos[:2]
        state.info["cumulative_distance_to_goal"] += jp.linalg.norm(state.info["rel_goal"])
        
        # Get observation, reward and metrics
        obs = self._get_obs(data, state.info, contact)
        done, fallen, goal_reached, max_steps_reached = self._get_termination(data, state.info)
        
        # Track final distance to goal when episode ends
        final_distance = jp.linalg.norm(state.info["rel_goal"])
        
        gait_metrics = self._compute_gait_metrics(data, state.info)
        for k, v in gait_metrics.items():
            state.metrics[f"gait_{k}"] = v
        state.metrics["success"] = goal_reached.astype(jp.float32)
        state.metrics["fallen"] = fallen.astype(jp.float32)
        state.metrics["max_steps_reached"] = max_steps_reached.astype(jp.float32)
        state.metrics["timeout"] = (max_steps_reached & ~goal_reached & ~fallen).astype(jp.float32)
        state.metrics["goal_reached"] = goal_reached.astype(jp.float32)
        state.metrics["cumulative_distance_to_goal"] = state.info["cumulative_distance_to_goal"]
        state.metrics["final_distance_to_goal"] = final_distance
        state.metrics["root_height"] = data.qpos[2]
        rewards = self.rewards.get(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards["max_steps"] = jp.where(max_steps_reached & ~goal_reached, jp.array(-100.0), jp.array(0.0))
        
        curriculum_weights = self._get_curriculum_weights(state.info)
        for k, v in rewards.items():
            base_scale = self._config.reward_config.scales[k]
            if(k in curriculum_weights.keys()):
                rewards[k] *= curriculum_weights[k]
            else:
                rewards[k] *= base_scale

        reward = jp.clip(sum(rewards.values()) * self.dt, -1000.0, 10000.0)

        # Increment counters and update state
        state.info["push"] = push
        state.info["step"] += 1
        state.info["plan_time"] += self.dt
        state.info["plan_timestep"] += 1
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
        state.info["plan_timestep"] = jp.where(
            done,                             
            0,       
            state.info["plan_timestep"]   
        )
        state.info["plan_time"] = jp.where(
            done,
            0.0,
            state.info["plan_time"],
        )
        state.info["cumulative_distance_to_goal"] = jp.where(
            done,
            0.0,
            state.info["cumulative_distance_to_goal"],
        )
        
        state.info["last_torques"] = state.info["torques"]
        state.info["torques"] = data.actuator_force
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
        goal_termination = jp.linalg.norm(info["rel_goal"]) <= 0.4
        steps_termination = info["step"] >= self._config.episode_length
        return (
            fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | goal_termination | steps_termination, 
            fall_termination,
            goal_termination,
            steps_termination
        )

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
        
        ref_com_x = info["ref_com_x"][info["plan_timestep"]]
        ref_com_y = info["ref_com_y"][info["plan_timestep"]]
        current_com = data.subtree_com[self._torso_body_id][:2]
        ref_com_pos_relative = jp.array([ref_com_x, ref_com_y]) - current_com
        
        state = jp.hstack(
            [
                noisy_linvel, # 3
                noisy_gyro,  # 3
                noisy_gravity,  # 3
                info["command"],  # 3
                info["rel_goal"],  # 2
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
        
        left_pos = data.site_xpos[self._feet_site_id[0]][:2]
        right_pos = data.site_xpos[self._feet_site_id[1]][:2]

        privileged_state = jp.hstack(
            [
                state,
                ref_com_pos_relative,  # 2
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_pose,
                joint_vel,
                root_height,  # 1
                info["torques"],
                contact,  # 2
                feet_vel,  # 4*3
                left_pos,  # 2
                right_pos,  # 2
                info["feet_air_time"],  # 2
                info["cumulative_distance_to_goal"],  # 1
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
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
    
    
    def _extract_command_from_com_trajectory(self, info: dict[str, Any]) -> jax.Array:
        """
        Extract velocity command (vx, vy, omega) from the planned COM trajectory.
        This replaces the map-based command with planner-based command.
        """
        current_idx = info["plan_timestep"]
        next_idx = jp.minimum(current_idx + 1, info["ref_com_x"].shape[0] - 1)
        vx = (info["ref_com_vel_x"][next_idx] + info["ref_com_vel_x"][current_idx]) / 2
        vy = (info["ref_com_vel_y"][next_idx] + info["ref_com_vel_y"][current_idx]) / 2
        w = info["command"][2]  # Keep the same yaw rate command
        return jp.array([vx, vy, w])
        

    def _compute_gait_metrics(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        """Compute gait quality metrics from MJX state."""
        metrics = {}
        
        # 1. ENERGY METRICS
        # ------------------
        # Torques and joint velocities
        torques = data.actuator_force  # (n_act,)
        dof_vel = data.qvel[6:]        # (n_dof,) - skip free joint
        
        # Mechanical power (W) = |τ * ω|
        power = jp.abs(torques * dof_vel)
        metrics["avg_power"] = jp.mean(power)
        metrics["total_power"] = jp.sum(power)
        
        # Torque smoothness (lower = smoother)
        torque_change = jp.abs(torques - info["last_torques"]) / self.dt
        metrics["torque_smoothness"] = jp.mean(torque_change)
        info["last_torques"] = torques
        
        # 2. STABILITY METRICS
        # --------------------
        # Base linear velocity variance (lower = more stable)
        base_lin_vel = data.qvel[:3]  # (3,)
        metrics["base_vel_variance"] = jp.var(base_lin_vel)
        
        # Base angular velocity magnitude (lower = more stable)
        base_ang_vel = data.qvel[3:6]  # (3,)
        metrics["base_ang_vel_norm"] = jp.linalg.norm(base_ang_vel)
        
        # Trunk orientation stability
        gravity = self.get_gravity(data)  # Projected gravity vector
        # Trunk tilt = angle from upright (gravity should be [0,0,-1])
        trunk_tilt = jp.arccos(jp.clip(-gravity[2], -1.0, 1.0))
        metrics["trunk_tilt"] = trunk_tilt
        
        return metrics
    
            
    def _get_curriculum_weights(self, info: Dict[str, Any]):
        alpha = jp.clip(
            info["global_step"] / self._config.reward_config.curriculum["ramp_steps"], 0.0, 1.0
        )
        
        # Weights departing from normal config and ending in curriculum config
        # tracking_lin_vel_x = self._config.reward_config.scales["tracking_lin_vel_x"] + alpha * (
        #     self._config.reward_config.curriculum["tracking_lin_vel_x"] - self._config.reward_config.scales["tracking_lin_vel_x"]
        # )
        # tracking_lin_vel_y = self._config.reward_config.scales["tracking_lin_vel_y"] + alpha * (
        #     self._config.reward_config.curriculum["tracking_lin_vel_y"] - self._config.reward_config.scales["tracking_lin_vel_y"]
        # )
        # tracking_ang_vel = self._config.reward_config.scales["tracking_ang_vel"] + alpha * (
        #     self._config.reward_config.curriculum["tracking_ang_vel"] - self._config.reward_config.scales["tracking_ang_vel"]
        # )
        # velocity_direction_alignment = self._config.reward_config.scales["velocity_direction_alignment"] + alpha * (
        #     self._config.reward_config.curriculum["velocity_direction_alignment"] - self._config.reward_config.scales["velocity_direction_alignment"]
        # )
        # torso_velocity_alignment = self._config.reward_config.scales["torso_velocity_alignment"] + alpha * (
        #     self._config.reward_config.curriculum["torso_velocity_alignment"] - self._config.reward_config.scales["torso_velocity_alignment"]
        # )
        # planner_com_x = self._config.reward_config.scales["planner_com_x"] + alpha * (
        #     self._config.reward_config.curriculum["planner_com_x"] - self._config.reward_config.scales["planner_com_x"]
        # )
        # planner_com_y = self._config.reward_config.scales["planner_com_y"] + alpha * (
        #     self._config.reward_config.curriculum["planner_com_y"] - self._config.reward_config.scales["planner_com_y"]
        # )
        
        
        return {
            # "tracking_lin_vel_x": tracking_lin_vel_x,
            # "tracking_lin_vel_y": tracking_lin_vel_y,
            # "tracking_ang_vel": tracking_ang_vel,
            # "velocity_direction_alignment": velocity_direction_alignment,
            # "torso_velocity_alignment": torso_velocity_alignment,
            # "planner_com_x": planner_com_x,
            # "planner_com_y": planner_com_y,
        }

