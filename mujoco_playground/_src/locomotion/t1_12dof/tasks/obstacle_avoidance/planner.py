from dataclasses import dataclass, replace
import jax.numpy as jp
import jax
from mujoco_playground._src.locomotion.t1_12dof.tasks.obstacle_avoidance.config import FootstepPlannerConfig, Foot
from mujoco_playground._src.locomotion.t1_12dof.tasks.obstacle_avoidance.map import Map
from mujoco_playground._src.locomotion.t1_12dof.tasks.obstacle_avoidance.lqr import LipDynamics, preview_control, plot_control_results
from mujoco_playground._src.locomotion.t1_12dof.tasks.obstacle_avoidance.utils import ZMPTrajectory, FootstepPlan

@jax.tree_util.register_pytree_node_class
@dataclass
class PlannerState:
    pelvis_pos: jp.ndarray
    pelvis_theta: float
    L: jp.ndarray
    R: jp.ndarray
    previous_fs_end_time: float
    swing_foot: int
    step_index: int
    time_since_last_step: float
    swing_foot_ids: jp.ndarray
    start_poses: jp.ndarray
    end_poses: jp.ndarray
    support_poses: jp.ndarray
    start_times: jp.ndarray
    ds_start_times: jp.ndarray
    end_times: jp.ndarray
    num_steps: int = 0

    def tree_flatten(self):
        children = (
            self.pelvis_pos, self.pelvis_theta, self.L, self.R, self.previous_fs_end_time,
            self.swing_foot, self.step_index, self.time_since_last_step,
            self.swing_foot_ids, self.start_poses, self.end_poses, self.support_poses,
            self.start_times, self.ds_start_times, self.end_times,
            self.num_steps,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
    
class FootstepPlanner:
    footstep_plan: FootstepPlan

    def __init__(self, map: Map = None):
        self.config = FootstepPlannerConfig()
        self.map = map
        
    def plan(
        self,
        left_foot_pose: jp.ndarray,
        right_foot_pose: jp.ndarray,
        start_time: float = 0.0,
    ) -> FootstepPlan:
        """
        Generates a full footstep and ZMP trajectory by iterating over timesteps.
        At each timestep, it computes a velocity command from the Dijkstra map,
        integrates a virtual pelvis model, and places footsteps when necessary.
        """
        
        cfg = self.config
        eps = 1e-4
        
        # --- INITIALIZATION ---
        # Virtual pelvis state
        pelvis_pos = 0.5 * (left_foot_pose[:2] + right_foot_pose[:2])
        pelvis_theta = self._wrap_angle(0.5 * (left_foot_pose[2] + right_foot_pose[2]))
        
        # Foot state
        L = left_foot_pose.copy()
        R = right_foot_pose.copy()
        
        # Stepping state machine variables
        step_index = 0
        time_since_last_step = 0.0
        swing_foot = cfg.first_swing
        step_duration = 1.0 / cfg.step_frequency
        
        # --- PREALLOCATE ARRAYS ---
        swing_foot_ids = jp.zeros(cfg.max_steps, dtype=jp.int32)
        start_poses = jp.zeros((cfg.max_steps, 3))
        end_poses = jp.zeros((cfg.max_steps, 3))
        support_poses = jp.zeros((cfg.max_steps, 3))
        start_times = jp.full(cfg.max_steps, -1.0)
        ds_start_times = jp.full(cfg.max_steps, -1.0)
        end_times = jp.full(cfg.max_steps, -1.0)
        
        # --- THE MAIN LOOP OVER TIMESTEPS ---
        # The state tuple carried through the loop
        # Build single planner state (pytree)
        init_state = PlannerState(
            pelvis_pos=pelvis_pos,
            pelvis_theta=pelvis_theta,
            L=L,
            R=R,
            previous_fs_end_time=0.0,
            swing_foot=swing_foot,
            step_index=step_index,
            time_since_last_step=time_since_last_step,
            swing_foot_ids=swing_foot_ids,
            start_poses=start_poses,
            end_poses=end_poses,
            support_poses=support_poses,
            start_times=start_times,
            ds_start_times=ds_start_times,
            end_times=end_times,
            num_steps=0,
        )
        
        def timestep_iteration(i, state: PlannerState):
            
            t = start_time + (i+1) * cfg.dt
            
            # Time update for step placement
            time_since_last_step_new = state.time_since_last_step + cfg.dt
            
            def integrate_pelvis(state: PlannerState) -> PlannerState:
                # Integrate virtual pelvis state with velocity command
                # command = self.map.get_command(state.pelvis_pos, state.pelvis_theta)
                command = jp.array([0.5, 0.0, 0.0])  # Default command if no map
                
                # Scale command to reasonable walking speeds
                linear_scale = 0.5
                angular_scale = 0.5 
                vx = command[0] * linear_scale
                vy = command[1] * linear_scale
                w = command[2] * angular_scale
                
                dtheta = jp.clip(w * cfg.dt, -cfg.theta_max, cfg.theta_max)
                theta_midpoint = self._wrap_angle(state.pelvis_theta + 0.5 * dtheta)
                dpos_world = self._rot(theta_midpoint) @ jp.array([vx * cfg.dt, vy * cfg.dt])
                pelvis_pos_new = state.pelvis_pos + dpos_world
                pelvis_theta_new = self._wrap_angle(state.pelvis_theta + dtheta)
                return replace(state, pelvis_pos=pelvis_pos_new, pelvis_theta=pelvis_theta_new)
            

            def place_new_footstep(current_state: PlannerState):
                s = current_state # shorthand
                is_first_step = (s.step_index == 0)
                
                # Step timing
                ss_duration = jp.where(is_first_step, 0.0, cfg.swing_percentage * step_duration)
                ds_duration = jp.where(is_first_step, cfg.warmup_ds_factor * step_duration, (1.0 - cfg.swing_percentage) * step_duration)
                fs_start = t - time_since_last_step_new
                ss_start = fs_start
                ds_start = fs_start + ss_duration
                end_t = fs_start + ss_duration + ds_duration
                
                # Update swing foot
                next_swing_foot = jp.where(is_first_step, s.swing_foot, 1 - s.swing_foot)
                
                # Calculate new foot pose
                lat_sign = jp.where(next_swing_foot == Foot.LEFT, 1.0, -1.0)
                lat_offset = self._rot(s.pelvis_theta) @ jp.array([0.0, lat_sign * cfg.step_width])
                end_pose = jp.array([
                    s.pelvis_pos[0] + lat_offset[0], 
                    s.pelvis_pos[1] + lat_offset[1], 
                    s.pelvis_theta
                ])
                start_pose = jp.where(
                    is_first_step,
                    s.R,
                    jp.where(next_swing_foot == Foot.RIGHT, s.R, s.L)
                )
                end_pose = jp.where(
                    is_first_step, 
                    s.L, 
                    end_pose
                )
                support_pose = jp.where(
                    is_first_step,
                    s.R,
                    jp.where(next_swing_foot == Foot.RIGHT, s.L, s.R)
                )
                
                # Update new foot pose
                next_L = jp.where(
                    is_first_step,
                    s.L,
                    jp.where(next_swing_foot == Foot.LEFT, end_pose, s.L)
                )
                next_R = jp.where(
                    is_first_step,
                    s.R,
                    jp.where(next_swing_foot == Foot.RIGHT, end_pose, s.R)
                )

                # Update event-based arrays
                j = s.step_index
                swing_foot_ids_new = s.swing_foot_ids.at[j].set(next_swing_foot)
                start_poses_new = s.start_poses.at[j].set(start_pose)
                end_poses_new = s.end_poses.at[j].set(end_pose)
                support_poses_new = s.support_poses.at[j].set(support_pose)
                start_times_new = s.start_times.at[j].set(ss_start)
                ds_start_times_new = s.ds_start_times.at[j].set(ds_start)
                end_times_new = s.end_times.at[j].set(end_t)

                return replace(
                    current_state,
                    step_index=current_state.step_index + 1,
                    time_since_last_step=0.0,
                    L=next_L,
                    R=next_R,
                    swing_foot=next_swing_foot,
                    swing_foot_ids=swing_foot_ids_new,
                    start_poses=start_poses_new,
                    end_poses=end_poses_new,
                    support_poses=support_poses_new,
                    start_times=start_times_new,
                    ds_start_times=ds_start_times_new,
                    end_times=end_times_new,
                    num_steps=current_state.num_steps + 1,
                )
                
            def place_same_footstep(current_state: PlannerState):
                return replace(
                    current_state,
                    time_since_last_step=time_since_last_step_new,
                )
                
            should_integrate_pelvis = (state.step_index > 0)
            state = jax.lax.cond(
                should_integrate_pelvis,
                integrate_pelvis,
                lambda s: s,
                operand=state
            )
                
            should_place = (time_since_last_step_new >= step_duration - eps) & (state.step_index < cfg.max_steps)
            new_footstep_event: PlannerState = jax.lax.cond(
                should_place,
                place_new_footstep,
                place_same_footstep,
                operand=state
            )    
            
            return replace(
                state,
                step_index=new_footstep_event.step_index,
                time_since_last_step=new_footstep_event.time_since_last_step,
                pelvis_pos=new_footstep_event.pelvis_pos,
                pelvis_theta=new_footstep_event.pelvis_theta,
                L=new_footstep_event.L,
                R=new_footstep_event.R,
                swing_foot=new_footstep_event.swing_foot,
                swing_foot_ids=new_footstep_event.swing_foot_ids,
                start_poses=new_footstep_event.start_poses,
                end_poses=new_footstep_event.end_poses,
                support_poses=new_footstep_event.support_poses,
                start_times=new_footstep_event.start_times,
                ds_start_times=new_footstep_event.ds_start_times,
                end_times=new_footstep_event.end_times,
                num_steps=new_footstep_event.num_steps,
            )
            
        

        final_state: PlannerState = jax.lax.fori_loop(
            0, cfg.P, timestep_iteration, init_state
        )

        fs_plan = FootstepPlan(
            swing_foot_ids=final_state.swing_foot_ids,
            start_poses=final_state.start_poses,
            end_poses=final_state.end_poses,
            support_poses=final_state.support_poses,
            start_times=final_state.start_times,
            ds_start_times=final_state.ds_start_times,
            end_times=final_state.end_times,
            num_steps=final_state.num_steps,
        )
        
        self.footstep_plan = fs_plan
        
        return fs_plan

    @staticmethod
    def _rot(theta: float) -> jp.ndarray:
        c, s = jp.cos(theta), jp.sin(theta)
        return jp.array([[c, -s], [s, c]])

    def _wrap_angle(self, angle: float) -> float:
        """
        Wrap an angle in radians to the range [-pi, pi].
        """
        return jp.arctan2(jp.sin(angle), jp.cos(angle))
    
    def get_step_index(self, t: float) -> int:
        """
        Get the current step index at time t.
        Returns -1 if before the first step, or num_steps if after the last step.
        """
        if not hasattr(self, 'footstep_plan'):
            raise ValueError("Footstep plan not computed yet. Call plan() first.")
        
        # Find active step with boolean masking
        start_times = self.footstep_plan.start_times[:self.footstep_plan.num_steps]
        end_times = self.footstep_plan.end_times[:self.footstep_plan.num_steps]
        is_active = (start_times <= t) & (t < end_times)
        step_index = jp.where(
            jp.any(is_active),
            jp.argmax(is_active),  # First True index
            -1                     # Not found
        )
        return step_index
    
        
    def compute_zmp_trajectory(self, footstep_plan: FootstepPlan) -> ZMPTrajectory:
        cfg = self.config
        
        # Step 1: Compute the full ZMP trajectory (length P)
        time = jp.linspace(0.0, cfg.Tp, cfg.P)
        
        # Initialize with midpoint of initial feet
        R0 = footstep_plan.start_poses[0]
        L0 = footstep_plan.end_poses[0]
        
        zmp_x_full = jp.full(cfg.P, 0.5 * (L0[0] + R0[0]))
        zmp_y_full = jp.full(cfg.P, 0.5 * (L0[1] + R0[1]))
        zmp_theta_full = jp.full(cfg.P, 0.5 * (L0[2] + R0[2]))
        
        # Step 2: Accumulate contributions from each footstep
        def process_footstep(j, zmp_state):
            zmp_x, zmp_y, zmp_theta = zmp_state
            
            start_pose = footstep_plan.support_poses[j]
            end_pose = footstep_plan.end_poses[j]
            ds_start = footstep_plan.ds_start_times[j]
            end_time = footstep_plan.end_times[j]
            
            # For first step, use current ZMP as start
            start_x = jp.where(j == 0, zmp_x[0], start_pose[0])
            start_y = jp.where(j == 0, zmp_y[0], start_pose[1])
            start_theta = jp.where(j == 0, zmp_theta[0], start_pose[2])
            
            end_x = end_pose[0]
            end_y = end_pose[1]
            end_theta = end_pose[2]
            
            # Compute sigma for the entire time vector
            duration = jp.maximum(end_time - ds_start, 1e-6)
            sigma = jp.clip((time - ds_start) / duration, 0.0, 1.0)
            
            # Accumulate the contribution of this footstep
            zmp_x = zmp_x + sigma * (end_x - start_x)
            zmp_y = zmp_y + sigma * (end_y - start_y)
            zmp_theta = zmp_theta + sigma * (end_theta - start_theta)
            
            return (zmp_x, zmp_y, zmp_theta)
        
        zmp_x_full, zmp_y_full, zmp_theta_full = jax.lax.fori_loop(
            0, footstep_plan.num_steps,
            process_footstep,
            (zmp_x_full, zmp_y_full, zmp_theta_full)
        )
        
        # Step 3: Create sliding windows [P-W+1, W] using lax.dynamic_slice
        indices = jp.arange(cfg.P - cfg.W + 1)
        zmp_windows_x = jax.vmap(lambda i: jax.lax.dynamic_slice(zmp_x_full, (i,), (cfg.W,)))(indices)
        zmp_windows_y = jax.vmap(lambda i: jax.lax.dynamic_slice(zmp_y_full, (i,), (cfg.W,)))(indices)
        zmp_windows_theta = jax.vmap(lambda i: jax.lax.dynamic_slice(zmp_theta_full, (i,), (cfg.W,)))(indices)
        
        # Compute velocities from the full trajectory
        zmp_vel_x_full = jp.gradient(zmp_x_full, cfg.dt)
        zmp_vel_y_full = jp.gradient(zmp_y_full, cfg.dt)
        zmp_vel_theta_full = jp.gradient(zmp_theta_full, cfg.dt)
        
        # Window the velocities too
        zmp_vel_windows_x = jax.vmap(lambda i: jax.lax.dynamic_slice(zmp_vel_x_full, (i,), (cfg.W,)))(indices)
        zmp_vel_windows_y = jax.vmap(lambda i: jax.lax.dynamic_slice(zmp_vel_y_full, (i,), (cfg.W,)))(indices)
        zmp_vel_windows_theta = jax.vmap(lambda i: jax.lax.dynamic_slice(zmp_vel_theta_full, (i,), (cfg.W,)))(indices)
        
        zmp_traj = ZMPTrajectory(
            zmp_midpoints_x=zmp_x_full,
            zmp_midpoints_y=zmp_y_full,
            zmp_midpoints_theta=zmp_theta_full,
            zmp_windows_x=zmp_windows_x,
            zmp_windows_y=zmp_windows_y,
            zmp_windows_theta=zmp_windows_theta,
            zmp_vel_windows_x=zmp_vel_windows_x,
            zmp_vel_windows_y=zmp_vel_windows_y,
            zmp_vel_windows_theta=zmp_vel_windows_theta
        )
        
        self.zmp_traj = zmp_traj  # Store for later use
        return zmp_traj
    

def main():
    map = Map()
    planner = FootstepPlanner(map)
    
    initial_left_foot = jp.array([0.0, 0.1, 0.0])
    initial_right_foot = jp.array([0.0, -0.1, 0.0])
    fs_plan = planner.plan(
        initial_left_foot,
        initial_right_foot
    )
    
    print(fs_plan.start_times)
    
    for step_index in range(fs_plan.num_steps):
        print(f"Step {step_index}: Support foot: ", "L" if fs_plan.swing_foot_ids[step_index] == Foot.RIGHT else "R")
        print("  Start pose: ", fs_plan.start_poses[step_index])
        print("  End pose: ", fs_plan.end_poses[step_index])
        print("  Support pose: ", fs_plan.support_poses[step_index])
        print("  Start time: ", fs_plan.start_times[step_index])
        print("  DS start time: ", fs_plan.ds_start_times[step_index])
        print("  End time: ", fs_plan.end_times[step_index])
    
    zmp_traj = planner.compute_zmp_trajectory(fs_plan)
    lip = LipDynamics(N=100, dt=planner.config.dt, zc=0.68) 
    
    initial_com_pos = 0.5 * (initial_left_foot + initial_right_foot)[:2]
    initial_com_vel = jp.array([
        (zmp_traj.zmp_midpoints_x[1] - zmp_traj.zmp_midpoints_x[0]) / planner.config.dt,
        (zmp_traj.zmp_midpoints_y[1] - zmp_traj.zmp_midpoints_y[0]) / planner.config.dt
    ])
    initial_com_acc = jp.array([0.0, 0.0])
    
    com_traj = preview_control(
        lip,
        zmp_traj,
        initial_com_pos,
        initial_com_vel,
        initial_com_acc
    )
    
    plot_control_results(lip, zmp_traj, com_traj)
    map.plot_map(fs_plan, zmp_traj, com_traj)


if __name__ == "__main__":
    main()