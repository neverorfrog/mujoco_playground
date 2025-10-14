from typing import Literal
from jax import numpy as jp
from dataclasses import dataclass, field, replace
import jax
from jax import jit, lax
import numpy as np
from mujoco_playground._src.locomotion.t1_12dof.tasks.obstacle_avoidance_com.utils import COMTrajectory, ZMPTrajectory
    
@jit 
def dare(A: jp.ndarray, B: jp.ndarray, Q: jp.ndarray, R: jp.ndarray, max_iter=5_000) -> jp.ndarray:
    """
    Solve discrete-time algebraic Riccati equation using iteration.
    """
    R_mat = R * jp.eye(B.shape[1]) if B.ndim > 1 else jp.array([[R]])
    def body(_, P):
        """Riccati iteration: P_{k+1} = Q + A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A"""

        BPB = B.T @ P @ B
        BPA = B.T @ P @ A
        
        def siso():
            inv_term = 1.0 / (R + BPB)
            return Q + A.T @ P @ A - A.T @ P @ B @ (inv_term * BPA)
        
        def mimo():
            inv_term = jp.linalg.inv(R_mat + BPB)
            return Q + A.T @ P @ A - A.T @ P @ B @ inv_term @ BPA
        
        return lax.cond(B.shape[1] == 1, siso, mimo)
    
    P = jax.lax.fori_loop(0, max_iter, body, Q)
    return P

@jax.tree_util.register_pytree_node_class
@dataclass
class LipDynamics:
    N: int  # Preview horizon length
    dt: float  # Time step duration
    zc: float  # Nominal COM height
    Q_e: float = 1.0
    R: float = 1e-5
    formulation: Literal["jerk", "zmp"] = "jerk"
    
    # Only store what's needed for preview control
    A: jp.ndarray = field(init=False)  # State transition matrix (3, 3)
    B: jp.ndarray = field(init=False)  # Input matrix (3,) or (3, 1)
    C: jp.ndarray = field(init=False)  # Output matrix (1, 3)
    G_i: float = field(init=False)     # Gain on integrated error
    G_x: jp.ndarray = field(init=False)  # Gains on state (3,)
    G_p: jp.ndarray = field(init=False)  # Preview gains (N+1,)
    
    def tree_flatten(self):
        # Arrays that can change (children)
        children = (self.A, self.B, self.C, self.G_x, self.G_p, self.G_i)
        # Static configuration (aux_data)
        aux = (self.N, self.dt, self.zc, self.Q_e, self.R, self.formulation)
        return children, aux
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        N, dt, zc, Q_e, R, formulation = aux
        A, B, C, G_x, G_p, G_i = children
        
        # Create instance without calling __post_init__
        obj = object.__new__(cls)
        obj.N = N
        obj.dt = dt
        obj.zc = zc
        obj.Q_e = Q_e
        obj.R = R
        obj.formulation = formulation
        obj.A = A
        obj.B = B
        obj.C = C
        obj.G_i = G_i
        obj.G_x = G_x
        obj.G_p = G_p
        return obj
    
    def __post_init__(self):
        """
        Precompute the LIP state transition matrices and preview control gains.
        """
        g = 9.81  # Gravitational acceleration [m/s²]
        h = self.zc
        dt = self.dt
        omega = jp.sqrt(g / h)

        if self.formulation == "zmp":
            print("\n=== Using Formulation With ZMP in the state ===")
            print("State: [x_c, ẋ_c, p]")
            print("Control: u = ṗ (ZMP velocity)")
            eta_dt = omega * dt
            c = jp.cosh(eta_dt)
            s = jp.sinh(eta_dt)
            
            self.A = jp.array([
                [c,           s/omega,     1-c],
                [omega*s,     c,           -omega*s],
                [0.0,         0.0,         1.0]
            ])
            
            self.B = jp.array([
                dt - s/omega,
                1 - c,
                dt
            ])
            
            self.C = jp.array([[0.0, 0.0, 1.0]])
            
        elif self.formulation == "jerk":
            print("\n=== Using Jerk Formulation ===")
            print("State: [x_c, ẋ_c, ẍ_c]")
            print("Control: u = ẍ̇ (jerk)")
            
            self.A = jp.array([
                [1.0,  dt,   dt**2/2],
                [0.0,  1.0,  dt],
                [0.0,  0.0,  1.0]
            ])
            
            self.B = jp.array([
                [dt**3/6],
                [dt**2/2],
                [dt]
            ])
            
            self.C = jp.array([[1.0, 0.0, -h/g]])
        
        # --- Compute preview gains (everything below is local) ---
        CA = self.C @ self.A
        CB = self.C @ self.B.reshape(-1, 1) if self.B.ndim == 1 else self.C @ self.B
        
        A_tilde = jp.block([
            [jp.ones((1, 1)), CA],
            [jp.zeros((3, 1)), self.A]
        ])
        
        B_tilde = jp.vstack([CB, self.B.reshape(-1, 1) if self.B.ndim == 1 else self.B])
        C_tilde = jp.array([[1.0, 0.0, 0.0, 0.0]])
        Q_tilde = C_tilde.T * self.Q_e * C_tilde
        
        P_tilde = dare(A_tilde, B_tilde, Q_tilde, self.R)
        
        # Compute gains
        BPB = B_tilde.T @ P_tilde @ B_tilde
        BPA = B_tilde.T @ P_tilde @ A_tilde
        K_tilde = jp.linalg.inv(self.R + BPB) @ BPA
        
        self.G_i = K_tilde[0, 0]
        self.G_x = K_tilde[0, 1:].flatten()
        
        print(f"\nBase gains:")
        print(f"  G_I (integral): {self.G_i:.6f}")
        print(f"  G_x (state):    {self.G_x}")
        
        # Stability check
        A_cl = A_tilde - B_tilde @ K_tilde
        eigvals = jp.linalg.eigvals(A_cl)
        max_eig = jp.max(jp.abs(eigvals))
        print(f"\nClosed-loop stability:")
        print(f"  Max |λ|: {max_eig:.6f} {'✓' if max_eig < 1.0 else '✗'}")
        
        # Compute preview gains
        I_tilde = jp.array([[1.0], [0.0], [0.0], [0.0]])
        X_tilde = -A_cl.T @ P_tilde @ I_tilde
        
        preview_gains = [-self.G_i]

        for _ in range(self.N-1):
            G_i = ((B_tilde.T @ X_tilde) / (self.R + BPB))[0, 0]
            X_tilde = A_cl.T @ X_tilde
            preview_gains.append(G_i)
        
        self.G_p = jp.array(preview_gains)
        
        print(f"\nPreview gains:")
        print(f"  G_p shape: {self.G_p.shape}")
        print(f"  G_p[0:5]:  {self.G_p[:5]}")
        print(f"  G_p[-5:]:  {self.G_p[-5:]}")
        print(f"  Sum:       {jp.sum(self.G_p):.6f}")
        print(f"  Decay:     {self.G_p[0]:.4f} → {self.G_p[-1]:.4e}")
        
        
@jax.tree_util.register_pytree_node_class
@dataclass
class PreviewControlState:
    state: jp.ndarray  # (3,) - [x_c, ẋ_c, ẍ_c] or [x_c, ẋ_c, p]
    control: float  # Scalar control input (jerk or ZMP velocity)
    error_sum: float  # Scalar integral of ZMP error
    com_positions: jp.ndarray  # (N,) pre-allocated COM positions
    com_velocities: jp.ndarray  # (N,) pre-allocated COM velocities
    com_accelerations: jp.ndarray # (N,) pre-allocated COM accelerations
    
    def tree_flatten(self):
        children = (self.state, self.control, self.error_sum, self.com_positions, self.com_velocities, self.com_accelerations)
        aux = None
        return children, aux
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        state, control, error_sum, com_positions, com_velocities, com_accelerations = children
        return cls(state=state, control=control, error_sum=error_sum, com_positions=com_positions, com_velocities=com_velocities, com_accelerations=com_accelerations)


@jax.jit
def preview_control_axis(
    lip: LipDynamics,
    zmp_positions: jp.ndarray,
    com_pos: jp.ndarray,
    com_vel: jp.ndarray,
    com_acc: jp.ndarray,
    axis: int
) -> PreviewControlState:
    """
    Perform preview control to compute COM trajectory given ZMP reference.
    Args:
        lip: LipDynamics instance with precomputed matrices and gains.
        zmp_positions: (N_sim,) array of desired ZMP positions.
        com_pos: Initial COM position (scalar).
        com_vel: Initial COM velocity (scalar).
        com_acc: Initial COM acceleration (scalar).
    """
    N = len(zmp_positions)
    N_preview = lip.N
    dt = lip.dt
    
    zmp_positions = jp.concatenate([zmp_positions, jp.full(N_preview, zmp_positions[-1])])
    
    initial_state = jax.lax.cond(
        lip.formulation == "jerk",
        lambda: jp.array([com_pos, com_vel, com_acc]),  # State: [x_c, ẋ_c, ẍ_c]
        lambda: jp.array([com_pos, com_vel, zmp_positions[0]])  # State: [x_c, ẋ_c, p] TODO zmp should be estimated
    )
    
    pc = PreviewControlState(
        state=initial_state, 
        control=0.0, 
        error_sum=0.0,
        com_positions=jp.zeros(N),  # Pre-allocate fixed size
        com_velocities=jp.zeros(N), # Pre-allocate fixed size
        com_accelerations=jp.zeros(N)  # Pre-allocate fixed size
    )
    
    # Store initial values
    pc = replace(
        pc,
        com_positions=pc.com_positions.at[0].set(initial_state[0]),
        com_velocities=pc.com_velocities.at[0].set(initial_state[1]),
        com_accelerations=pc.com_accelerations.at[0].set(initial_state[2] if lip.formulation == "jerk" else 0.0)
    )
    
    def step_fn(i, pc: PreviewControlState) -> PreviewControlState:
        """Single time step of preview control."""
        # jax.debug.print("Step {}/{}", i+1, N-1)
        state = pc.state
        
        # Current zmp
        zmp = jax.lax.cond(
            lip.formulation == "jerk",
            lambda: (lip.C @ state)[0],  # ZMP = x_c - (z_c/g) * ẍ_c
            lambda: (pc.state[2])  # ZMP = p
        )
        
        # Tracking error
        error = zmp - zmp_positions[i]
        error_sum = pc.error_sum + error * dt
        
        # Preview window
        preview_refs = jax.lax.dynamic_slice(zmp_positions, (i+1,), (N_preview,))
        
        # Control input
        u = (
            -lip.G_i * error_sum
            -jp.dot(lip.G_x, state)
            -jp.dot(lip.G_p, preview_refs)
        )
        
        # Apply dynamics
        state_next = lip.A @ state + lip.B.flatten() * u
        
        return replace(
            pc,
            state=state_next,
            control=u,
            error_sum=error_sum,
            com_positions=pc.com_positions.at[i+1].set(state_next[0]),
            com_velocities=pc.com_velocities.at[i+1].set(state_next[1]),
            com_accelerations=pc.com_accelerations.at[i+1].set(state_next[2] if lip.formulation == "jerk" else 0.0)
        )
        
    result: PreviewControlState = jax.lax.fori_loop(
        0, N-1, lambda i, pc: step_fn(i, pc), pc
    )
    
    return result
    
def preview_control(
    lip: LipDynamics,
    zmp_traj: ZMPTrajectory,
    com_pos: jp.ndarray,
    com_vel: jp.ndarray,
    com_acc: jp.ndarray,
) -> COMTrajectory:
    """
    Wrapper to perform preview control for both x and y axes.
    Args:
        lip: LipDynamics instance with precomputed matrices and gains.
        zmp_positions: (N_sim, 2) array of desired ZMP positions in x and y.
        com_pos: Initial COM position (2,) array.
        com_vel: Initial COM velocity (2,) array.
        com_acc: Initial COM acceleration (2,) array.
    """
    
    result_x = preview_control_axis(
        lip, 
        zmp_traj.zmp_midpoints_x, 
        com_pos[0], com_vel[0], com_acc[0], axis=0
    )
    
    result_y = preview_control_axis(
        lip, 
        zmp_traj.zmp_midpoints_y, 
        com_pos[1], com_vel[1], com_acc[1], axis=1
    )
    
    return COMTrajectory(
        x_positions=result_x.com_positions,
        y_positions=result_y.com_positions,
        x_velocities=result_x.com_velocities,
        y_velocities=result_y.com_velocities,
        x_accelerations=result_x.com_accelerations,
        y_accelerations=result_y.com_accelerations
    )


def plot_control_results(
    lip: LipDynamics,
    zmp_traj: ZMPTrajectory,
    com_traj: COMTrajectory
):
    zmp_x_ref = zmp_traj.zmp_midpoints_x
    zmp_y_ref = zmp_traj.zmp_midpoints_y
    com_x_pos = com_traj.x_positions
    com_y_pos = com_traj.y_positions
    com_x_acc = com_traj.x_accelerations
    com_y_acc = com_traj.y_accelerations
    
     # 1. Preview horizon adequacy
    preview_time = lip.N * lip.dt
    print(f"Preview window: {preview_time:.2f}s")

    # At ramp rate 0.2 m/s, in 1 second you travel 0.2m
    # You need to see far enough ahead!

    # 2. Check if error is proportional to ramp rate
    ramp_rate = (zmp_x_ref[-1] - zmp_x_ref[0]) / (len(zmp_x_ref) * lip.dt)
    predicted_error = ramp_rate / lip.G_i
    print(f"Ramp rate: {ramp_rate:.3f} m/s")
    print(f"Predicted lag: {predicted_error:.3f} m")

    # 3. Check if preview gains are being used
    print(f"Preview contribution: {jp.sum(lip.G_p):.3f}")
    
    # Compute actual ZMP from COM trajectory
    ZMP_x_actual = []
    ZMP_y_actual = []
    for i in range(len(com_x_pos)):
        if lip.formulation == "jerk":
            # ZMP = x_c - (z_c/g) * ẍ_c
            zmp_x = com_x_pos[i] - (lip.zc/9.81) * com_x_acc[i]
            zmp_y = com_y_pos[i] - (lip.zc/9.81) * com_y_acc[i]
        else:
            # For ZMP formulation, ZMP is the third state
            zmp_x = com_x_pos[i] if i == len(com_x_pos) - 1 else com_x_pos[i]
            zmp_y = com_y_pos[i] if i == len(com_y_pos) - 1 else com_y_pos[i]

        ZMP_x_actual.append(zmp_x)
        ZMP_y_actual.append(zmp_y)
    
    # Convert to numpy for plotting
    zmp_x_ref_np = np.array(zmp_x_ref)
    zmp_y_ref_np = np.array(zmp_y_ref)
    zmp_x_actual_np = np.array(ZMP_x_actual)
    zmp_y_actual_np = np.array(ZMP_y_actual)
    com_x_np = np.array(com_x_pos)
    com_y_np = np.array(com_y_pos)

    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Preview Control Results ({lip.formulation.capitalize()} Formulation)")
    
    # X direction
    plt.subplot(3, 1, 1)
    plt.plot(zmp_x_ref_np, 'g--', linewidth=2, label='ZMP_x_ref')
    # plt.plot(zmp_x_actual_np, 'b', linewidth=1.5, label='ZMP_x')
    plt.plot(com_x_np, 'r--', linewidth=1.5, label='COM_x')
    plt.ylabel('X Position [m]')
    plt.legend()
    plt.grid(True)
    
    # Y direction
    plt.subplot(3, 1, 2)
    plt.plot(zmp_y_ref_np, 'g--', linewidth=2, label='ZMP_y_ref')
    # plt.plot(zmp_y_actual_np, 'b', linewidth=1.5, label='ZMP_y')
    plt.plot(com_y_np, 'r--', linewidth=1.5, label='COM_y')
    plt.ylabel('Y Position [m]')
    plt.legend()
    plt.grid(True)
    
    # 2D trajectory
    plt.subplot(3, 1, 3)
    plt.plot(zmp_x_ref_np, zmp_y_ref_np, 'g--', linewidth=2, label='ZMP_ref')
    # plt.plot(zmp_x_actual_np, zmp_y_actual_np, 'b', linewidth=1.5, label='ZMP')
    plt.plot(com_x_np, com_y_np, 'r--', linewidth=1.5, label='COM')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n=== Tracking Performance ===")
    print(f"X direction:")
    print(f"  Mean error:     {np.mean(zmp_x_ref_np - zmp_x_actual_np):.6f} m")
    print(f"  RMS error:      {np.sqrt(np.mean((zmp_x_ref_np - zmp_x_actual_np)**2)):.6f} m")
    print(f"  Max error:      {np.max(np.abs(zmp_x_ref_np - zmp_x_actual_np)):.6f} m")
    print(f"\nY direction:")
    print(f"  Mean error:     {np.mean(zmp_y_ref_np - zmp_y_actual_np):.6f} m")
    print(f"  RMS error:      {np.sqrt(np.mean((zmp_y_ref_np - zmp_y_actual_np)**2)):.6f} m")
    print(f"  Max error:      {np.max(np.abs(zmp_y_ref_np - zmp_y_actual_np)):.6f} m")

    plt.savefig(f"preview_control_{lip.formulation}.png")