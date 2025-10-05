from jax import numpy as jp
from dataclasses import dataclass

@dataclass
class COMTrajectory:
    """Container for COM trajectory data."""
    x_positions: jp.ndarray  # (N,) COM states in x direction
    y_positions: jp.ndarray  # (N,) COM states in y direction
    x_velocities: jp.ndarray  # (N,) COM velocities in x direction
    y_velocities: jp.ndarray  # (N,) COM velocities in y direction
    x_accelerations: jp.ndarray  # (N,) COM accelerations in x direction
    y_accelerations: jp.ndarray  # (N,) COM accelerations in y direction
    
@dataclass
class FootstepPlan:
    """
    Represents a planned sequence of footstep positions as JAX arrays.
    """
    # Footstep data (size = max_steps)
    swing_foot_ids: jp.ndarray
    start_poses: jp.ndarray
    end_poses: jp.ndarray
    support_poses: jp.ndarray
    start_times: jp.ndarray
    ds_start_times: jp.ndarray
    end_times: jp.ndarray
    num_steps: int = 0
    
    
@dataclass
class ZMPTrajectory:
    # ZMP Trajectory 
    zmp_midpoints_x: jp.ndarray
    zmp_midpoints_y: jp.ndarray
    zmp_midpoints_theta: jp.ndarray
    
    zmp_windows_x: jp.ndarray
    zmp_windows_y: jp.ndarray
    zmp_windows_theta: jp.ndarray
    zmp_vel_windows_x: jp.ndarray
    zmp_vel_windows_y: jp.ndarray
    zmp_vel_windows_theta: jp.ndarray
    
@dataclass
class FootstepState:
    """
    State information for the current footstep at a given time.
    """
    # Step identification
    step_index: int                 # Index of current footstep (-1 if before plan starts)
    is_valid: bool                  # True if time is within the plan
    
    # Timing information
    start_time: float               # When this footstep starts (single support)
    ds_start_time: float            # When double support begins
    end_time: float                 # When this footstep ends
    
    # Phase information (0.0 to 1.0)
    phase: float                    # Overall phase within this footstep
    ss_phase: float                 # Phase within single support (0-1, or -1 if in DS)
    ds_phase: float                 # Phase within double support (0-1, or -1 if in SS)

    # Foot information
    swing_foot_id: int              # Which foot is swinging (0=RIGHT, 1=LEFT)
    support_foot_id: int            # Which foot is support (0=RIGHT, 1=LEFT)

    # Poses [x, y, theta]
    swing_start_pose: jp.ndarray    # Starting pose of swing foot
    swing_end_pose: jp.ndarray      # Target pose of swing foot
    support_pose: jp.ndarray        # Pose of support foot


def query_footstep_plan(footstep_plan: FootstepPlan, time: float) -> FootstepState:
        """
        Query the footstep plan at a given time instant.
        
        This function finds which footstep is active at the query time and
        returns all relevant state information about that footstep.
        
        Args:
            footstep_plan: The precomputed footstep plan
            time: The query time instant
            
        Returns:
            FootstepState containing all information about the current footstep
            
        Example:
            >>> plan = planner.plan(left_pose, right_pose)
            >>> state = planner.query_plan(plan, time=0.5)
            >>> print(f"Step {state.step_index}, phase {state.phase:.2f}")
            >>> print(f"Support foot: {state.support_foot_id}")
        """
        # Find the active footstep index using vectorized comparison
        # A footstep is active if: start_time <= time < end_time
        
        start_times = footstep_plan.start_times[:footstep_plan.num_steps]
        end_times = footstep_plan.end_times[:footstep_plan.num_steps]
        
        # Create boolean mask for active step
        is_active = (start_times <= time) & (time < end_times)
        
        # Find index (-1 if no active step found)
        # argmax returns first True index, or 0 if all False
        # We check if any() to distinguish between "found at 0" and "not found"
        step_index = jp.where(
            jp.any(is_active),
            jp.argmax(is_active),  # First True index
            -1                      # Not found
        )
        
        is_valid = step_index >= 0
        
        # Helper function to safely index arrays (returns zeros if invalid)
        def safe_get(arr, idx, default_val):
            return jp.where(is_valid, arr[idx], default_val)
        
        # Extract timing information
        start_time = safe_get(start_times, step_index, 0.0)
        ds_start_time = safe_get(footstep_plan.ds_start_times, step_index, 0.0)
        end_time = safe_get(end_times, step_index, 0.0)
        
        # Compute phases
        duration = end_time - start_time
        phase = jp.where(
            is_valid & (duration > 1e-6),
            (time - start_time) / duration,
            0.0
        )
        phase = jp.clip(phase, 0.0, 1.0)
        
        # Determine if in single support or double support
        in_ss = is_valid & (time < ds_start_time)
        in_ds = is_valid & (time >= ds_start_time)
        
        # Single support phase (0 to 1 during SS, -1 during DS)
        ss_duration = ds_start_time - start_time
        ss_phase = jp.where(
            in_ss & (ss_duration > 1e-6),
            (time - start_time) / ss_duration,
            -1.0  # Not in single support
        )
        
        # Double support phase (0 to 1 during DS, -1 during SS)
        ds_duration = end_time - ds_start_time
        ds_phase = jp.where(
            in_ds & (ds_duration > 1e-6),
            (time - ds_start_time) / ds_duration,
            -1.0  # Not in double support
        )
        
        # Extract foot information
        swing_foot_id = safe_get(footstep_plan.swing_foot_ids, step_index, 0)
        support_foot_id = 1 - swing_foot_id  # Opposite foot
        
        # Extract poses
        swing_start_pose = safe_get(
            footstep_plan.start_poses, 
            step_index, 
            jp.zeros(3)
        )
        swing_end_pose = safe_get(
            footstep_plan.end_poses,
            step_index,
            jp.zeros(3)
        )
        support_pose = safe_get(
            footstep_plan.support_poses,
            step_index,
            jp.zeros(3)
        )
        
        return FootstepState(
            step_index=step_index,
            is_valid=is_valid,
            start_time=start_time,
            ds_start_time=ds_start_time,
            end_time=end_time,
            phase=phase,
            ss_phase=ss_phase,
            ds_phase=ds_phase,
            swing_foot_id=swing_foot_id,
            support_foot_id=support_foot_id,
            swing_start_pose=swing_start_pose,
            swing_end_pose=swing_end_pose,
            support_pose=support_pose
        )