from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple
import jax.numpy as jp
import jax


class Foot(IntEnum):
    LEFT = 0
    RIGHT = 1


@dataclass
class FootstepPlannerConfig:
    """
    Configuration for the footstep planner.
    """
    P: int = 500
    """Number of timesteps to plan ahead. TODO: should be the same as episode length? """
    dt: float = 0.02
    """Time step duration for the planner. TODO: should be the same as episode step length? """
    Tp: float = P * dt
    """Total planning horizon in seconds."""
    first_swing: Foot = Foot.RIGHT
    """Which foot is the first swing foot"""
    step_width: float = 0.1
    """Lateral foot separation from the pelvis. Depends on the robot dimensions"""
    swing_percentage: float = 0.6
    """Percentage duration of the swing phase for each step."""
    peak_height: float = 0.05
    """Desired height of each foot at peak phase."""
    warmup_ds_factor: float = 1
    """Number of warmup steps for the double support phase"""
    step_frequency: float = 1.0
    """ Steps per second (one step = swing phase + stance phase)"""
    theta_max: float = 0.25
    """Maximum foot rotation angle during the swing phase [rad]"""


@dataclass
class Footstep:
    """
    Represents a single footstep with position and orientation. One footstep
    comprises first a swing phase and then a stance phase. The footstep
    ends when the swing phase of the next footstep starts.
    """

    # Poses of the swing foot
    swing_foot: Foot
    start_pose: jp.ndarray  # [x, y, theta] when moving foot lifts off
    end_pose: jp.ndarray  # [x, y, theta] when moving foot touches down

    # Timing
    start_time: float  # time when foot lifts off
    ds_start_time: float  # time when foot hits ground
    end_time: float  # time when double support (and footsteps) ends


@dataclass
class FootstepPlan:
    """
    Represents a planned sequence of footstep positions as JAX arrays.
    """

    swing_foot_ids: jp.ndarray
    start_poses: jp.ndarray
    end_poses: jp.ndarray
    support_poses: jp.ndarray
    start_times: jp.ndarray
    ds_start_times: jp.ndarray
    end_times: jp.ndarray
    step_frequency: float


class FootstepPlanner:
    footstep_plan: FootstepPlan

    def __init__(self):
        self.config = FootstepPlannerConfig()

    def plan(
        self,
        command: jp.ndarray,  # [vx, vy, w]
        left_foot_pose: jp.ndarray,
        right_foot_pose: jp.ndarray,
        start_time: float = 0.0,
    ) -> FootstepPlan:
        """
        Generate a sequence of footstep positions based on the current foot positions.
        Generate a fixed footstep sequence following a constant velocity command.
            - Uses unicycle-like integration for pelvis motion per step.
            - Places swing foot at +/- step_width/2 lateral offset along heading.

        First step is virtual: SS duration = 0, DS duration = K * step_duration.
        It does not move any foot but advances the virtual pelvis to shift the CoM.
        """
        # Making sure parameters are in the right format
        step_frequency = self.config.step_frequency
        swing_percentage = self.config.swing_percentage
        step_width = self.config.step_width
        theta_max = self.config.theta_max
        P = self.config.P
        dt = self.config.dt
        vx, vy, w = command[0], command[1], command[2]
        warmup_ds_factor = self.config.warmup_ds_factor
        swing_foot = self.config.first_swing

        # Derived quantities
        step_duration = 1.0 / step_frequency
        num_steps = int((P * dt) / step_duration)
        nominal_ss_duration = swing_percentage * step_duration
        nominal_ds_duration = step_duration - nominal_ss_duration

        # Initialization
        pelvis_pos = 0.5 * (left_foot_pose[:2] + right_foot_pose[:2])
        pelvis_theta = float(
            self._wrap_angle(0.5 * (left_foot_pose[2] + right_foot_pose[2]))
        )
        t = start_time
        L = left_foot_pose.copy()
        R = right_foot_pose.copy()

        # Pre-allocate arrays for JAX loop
        swing_foot_ids = jp.zeros(num_steps, dtype=jp.int32)
        start_poses = jp.zeros((num_steps, 3))
        end_poses = jp.zeros((num_steps, 3))
        support_poses = jp.zeros((num_steps, 3))
        start_times = jp.zeros(num_steps)
        ds_start_times = jp.zeros(num_steps)
        end_times = jp.zeros(num_steps)

        init_val = (
            pelvis_pos,
            pelvis_theta,
            t,
            L,
            R,
            swing_foot,
            swing_foot_ids,
            start_poses,
            end_poses,
            support_poses,
            start_times,
            ds_start_times,
            end_times,
        )

        def step_iteration(j, val):
            (
                pelvis_pos,
                pelvis_theta,
                t,
                L,
                R,
                swing_foot,
                swing_foot_ids,
                start_poses,
                end_poses,
                support_poses,
                start_times,
                ds_start_times,
                end_times,
            ) = val

            # Determine step durations (virtual first step has no swing)
            ss_duration = jp.where(j == 0, 0.0, nominal_ss_duration)
            ds_duration = jp.where(
                j == 0, warmup_ds_factor * step_duration, nominal_ds_duration
            )
            
            # Step timing
            ss_start = t
            ds_start = t + ss_duration
            end_t = t + ss_duration + ds_duration

            # Determine start and end poses for the swing foot
            def first_step_poses(_):
                start_pose = jp.where(swing_foot == Foot.RIGHT, L, R)
                return pelvis_pos, pelvis_theta, start_pose, start_pose, start_pose, L, R

            def step_poses(_):
                # Foot.LEFT is 0, Foot.RIGHT is 1. We want lateral offset to be
                # positive for LEFT and negative for RIGHT.
                dtheta = jp.clip(w * step_duration, -theta_max, theta_max)
                theta_midpoint = self._wrap_angle(pelvis_theta + 0.5 * dtheta)
                pelvis_theta_new = self._wrap_angle(pelvis_theta + dtheta)
                dpos = jp.array([vx * step_duration, vy * step_duration])
                dpos_world = self._rot(theta_midpoint) @ dpos
                pelvis_pos_new = pelvis_pos + dpos_world
                lateral_sign = jp.where(swing_foot == Foot.LEFT, 1.0, -1.0)
                lateral_offset = self._rot(pelvis_theta_new) @ jp.array(
                    [0.0, lateral_sign * step_width]
                )
                end_pos = pelvis_pos_new + lateral_offset
                end_pose_new = jp.array([end_pos[0], end_pos[1], pelvis_theta_new])

                start_pose = jp.where(swing_foot == Foot.RIGHT, R, L)
                support_pose = jp.where(swing_foot == Foot.RIGHT, L, R)
                new_R = jp.where(swing_foot == Foot.RIGHT, end_pose_new, R)
                new_L = jp.where(swing_foot == Foot.LEFT, end_pose_new, L)
                return pelvis_pos_new, pelvis_theta_new, start_pose, end_pose_new, support_pose, new_L, new_R

            pelvis_pos_new, pelvis_theta_new, start_pose, end_pose, support_pose, next_L, next_R = jax.lax.cond(
                j == 0, first_step_poses, step_poses, operand=None
            )

            # Store results for the current step
            swing_foot_ids = swing_foot_ids.at[j].set(swing_foot)
            start_poses = start_poses.at[j].set(start_pose)
            end_poses = end_poses.at[j].set(end_pose)
            support_poses = support_poses.at[j].set(support_pose)
            start_times = start_times.at[j].set(ss_start)
            ds_start_times = ds_start_times.at[j].set(ds_start)
            end_times = end_times.at[j].set(end_t)

            # Update state for the next iteration
            pelvis_pos, pelvis_theta, t = pelvis_pos_new, pelvis_theta_new, end_t
            L, R = next_L, next_R

            # Switch swing foot (0->1, 1->0), but not after the virtual first step
            next_swing_foot = 1 - swing_foot
            swing_foot = jp.where(j > 0, next_swing_foot, swing_foot)

            return (
                pelvis_pos,
                pelvis_theta,
                t,
                L,
                R,
                swing_foot,
                swing_foot_ids,
                start_poses,
                end_poses,
                support_poses,
                start_times,
                ds_start_times,
                end_times,
            )

        (
            *_,
            swing_foot_ids,
            start_poses,
            end_poses,
            support_poses,
            start_times,
            ds_start_times,
            end_times,
        ) = jax.lax.fori_loop(0, num_steps, step_iteration, init_val)

        self.footstep_plan = FootstepPlan(
            swing_foot_ids=swing_foot_ids,
            start_poses=start_poses,
            end_poses=end_poses,
            support_poses=support_poses,
            start_times=start_times,
            ds_start_times=ds_start_times,
            end_times=end_times,
            step_frequency=step_frequency,
        )
        return self.footstep_plan

    @staticmethod
    def _rot(theta: float) -> jp.ndarray:
        c, s = jp.cos(theta), jp.sin(theta)
        return jp.array([[c, -s], [s, c]])

    def _wrap_angle(self, angle: float) -> float:
        """
        Wrap an angle in radians to the range [-pi, pi].
        """
        return jp.arctan2(jp.sin(angle), jp.cos(angle))
    
    def compute_zmp_midpoints(
        self,
        footstep_plan: FootstepPlan,
        left_foot_pose: jp.ndarray,
        right_foot_pose: jp.ndarray,
        current_time: float = 0.0,
        previous_zmp_midpoints: jp.ndarray = None
    ) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """
        Compute ZMP midpoints for the moving constraint.
        
        Args:
            footstep_plan: The planned footstep sequence
            left_foot_pose: Current left foot pose [x, y, theta]
            right_foot_pose: Current right foot pose [x, y, theta]
            current_time: Current simulation time
            previous_zmp_midpoints: Previous ZMP midpoints for continuity [x, y, theta]
            
        Returns:
            Tuple of (zmp_midpoints_x, zmp_midpoints_y, zmp_midpoints_theta)
        """
        time = jp.linspace(0.0, self.config.Tp, self.config.P)
        num_steps = footstep_plan.swing_foot_ids.shape[0]
        
        def process_footstep(i: int, zmp_midpoints: Tuple[jp.ndarray, jp.ndarray, jp.ndarray]) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
            zmp_x, zmp_y, zmp_theta = zmp_midpoints
            jax.debug.print("Processing footstep {i}", i=i)
            
            # Get footstep data
            start_pose = footstep_plan.support_poses[i]
            end_pose = footstep_plan.end_poses[i]
            ds_start_time = footstep_plan.ds_start_times[i]
            end_time = footstep_plan.end_times[i]
            
            # For the first footstep, use current ZMP position as start
            start_x = jp.where(i == 0, zmp_x[0], start_pose[0])
            start_y = jp.where(i == 0, zmp_y[0], start_pose[1])
            start_theta = jp.where(i == 0, zmp_theta[0], start_pose[2])
        
            # End position is the footstep end pose
            end_x = end_pose[0]
            end_y = end_pose[1]
            end_theta = end_pose[2]
            
            jax.debug.print("Footstep {i} start pose: ({start_x}, {start_y}, {start_theta})", i=i, start_x=start_x, start_y=start_y, start_theta=start_theta)
            jax.debug.print("Footstep {i} end pose: ({end_x}, {end_y}, {end_theta})", i=i, end_x=end_x, end_y=end_y, end_theta=end_theta)
            
            # Compute sigma function for smooth transition
            sigma = self._sigma_function(time, ds_start_time, end_time)
            
            # jax.debug.print("    start_time: {ds_start_time}, end_time: {end_time}", ds_start_time=ds_start_time, end_time=end_time)
            # jax.debug.print("    Sigma values for step {i}: {sigma}", i=i, sigma=sigma)
            
            # Update ZMP midpoints with smooth transition
            zmp_x = zmp_x + sigma * (end_x - start_x)
            zmp_y = zmp_y + sigma * (end_y - start_y)
            zmp_theta = zmp_theta + sigma * (end_theta - start_theta)
            
            jax.debug.print("ZMP x after step {i}: {zmp_x}", i=i, zmp_x=zmp_x)
            # jax.debug.print("ZMP y after step {i}: {zmp_y}", i=i, zmp_y=zmp_y)
            # jax.debug.print("ZMP theta after step {i}: {zmp_theta}", i=i, zmp_theta=zmp_theta)
            
            # jax.debug.print("Footstep end pose: ({end_x}, {end_y}, {end_theta})", end_x=end_x, end_y=end_y, end_theta=end_theta)
            return (zmp_x, zmp_y, zmp_theta)
        
        relevant_steps = jp.where(
            (footstep_plan.ds_start_times < current_time + self.config.Tp) &
            (footstep_plan.end_times > current_time),
            jp.arange(num_steps),
            -1
        )
        
        def process_relevant_step(i, zmp_midpoints):
            step_idx = relevant_steps[i]
            return jax.lax.cond(
                step_idx >= 0,
                lambda x: process_footstep(step_idx, x),
                lambda x: x,
                zmp_midpoints
            )
        
        midpoint = 0.5 * (left_foot_pose + right_foot_pose)
        zmp_midpoints_x = jp.full(self.config.P, midpoint[0])
        zmp_midpoints_y = jp.full(self.config.P, midpoint[1])
        zmp_midpoints_theta = jp.full(self.config.P, midpoint[2])
        
        zmp_midpoints_x, zmp_midpoints_y, zmp_midpoints_theta = jax.lax.fori_loop(
            0, num_steps, 
            process_relevant_step,
            (zmp_midpoints_x, zmp_midpoints_y, zmp_midpoints_theta)
        )
        
        return zmp_midpoints_x, zmp_midpoints_y, zmp_midpoints_theta

    def _sigma_function(self, time: jp.ndarray, t0: float, t1: float) -> jp.ndarray:
        """
        JAX implementation of the sigma function for smooth transitions.
        
        Args:
            time: Time vector
            t0: Start time of transition
            t1: End time of transition
            
        Returns:
            Sigma values for smooth transition (0 to 1)
        """
        duration = t1 - t0
        # Avoid division by zero
        duration = jp.maximum(duration, 1e-6)
        # Compute normalized time
        sigma = (time - t0) / duration
        # Clamp to [0, 1]
        sigma = jp.clip(sigma, 0.0, 1.0)
        return sigma
    
if __name__ == "__main__":
    planner = FootstepPlanner()
    command = jp.array([0.2, 0.0, 0.0])
    left_foot_pose = jp.array([0.0, 0.1, 0.0])
    right_foot_pose = jp.array([0.0, -0.1, 0.0])
    plan = planner.plan(command, left_foot_pose, right_foot_pose)
    # print("Planned footsteps:")
    # for i in range(len(plan.swing_foot_ids)):
    #     print(f"Step {i+1}: Swing foot: {'LEFT' if plan.swing_foot_ids[i]==Foot.LEFT else 'RIGHT'}")
    #     print(f"   Start pose: {plan.start_poses[i]}")
    #     print(f"   End pose: {plan.end_poses[i]}")
    #     print(f"   Start time: {plan.start_times[i]}, DS start time: {plan.ds_start_times[i]}, End time: {plan.end_times[i]}")
    zmp_x, zmp_y, zmp_theta = planner.compute_zmp_midpoints(
        plan, left_foot_pose, right_foot_pose
    )