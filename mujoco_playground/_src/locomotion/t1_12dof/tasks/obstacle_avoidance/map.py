import jax.numpy as jp
import jax
import pygame
import os
import math
from mujoco_playground._src.locomotion.t1_12dof.tasks.obstacle_avoidance.config import SceneConfig
from mujoco_playground._src.locomotion.t1_12dof.tasks.obstacle_avoidance.utils import ZMPTrajectory, FootstepPlan, COMTrajectory

class Map:
    """Abstract discrete map class for obstacle avoidance tasks."""
    
    _map: jp.ndarray
    
    def __init__(self, scene_config: SceneConfig = None):
        if scene_config is None:
            scene_config = SceneConfig()
        self.scene_config = scene_config
        self.bins = self.scene_config.bins
        self.bin_size = self.scene_config.bin_size
        self.goal = self.scene_config.goal_position
        self.obstacles = self.scene_config.obstacle_positions
        self.width = self.scene_config.width
        self.height = self.scene_config.height
        self.origin = self.scene_config.origin
        self.abs_gamma = self.scene_config.abs_gamma
        self.CARDINAL_COST = 1.0
        self.DIAGONAL_COST = 1.5
        self.FREE = -1.0
        self.GOAL = 0.0
        self.OBSTACLE = 1000.0  # Changed to large float instead of 2 * bins
        self.NEAR_OBSTACLE = 1.0  # Cost for cells adjacent to obstacles
        self.NEAR_OBSTACLE_INFLATE = 2.0  # Inflation cost for cells adjacent to obstacles
        self.directions = jp.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],  # Cardinal
            [1, 1], [1, -1], [-1, 1], [-1, -1] # Diagonal
        ])
        self.cardinal_directions = jp.array([
            [1, 0], [-1, 0], [0, 1], [0, -1]  # Cardinal
        ])
        self._map = self._compute_costs(self.goal, self.obstacles)
        self._gradient = self._compute_gradient(self._map)
        self._policy = self._compute_greedy_policy(self._map)
        self._xml = self.generate_scene_xml(self.goal, self.obstacles)
        
        
    def get_command(self, pos: jp.ndarray, robot_yaw: float) -> jp.ndarray:
        """Get the unit vector command from the map gradient at a given world position."""
        
        # 0) Choose command scaling based on distance to goal (HOTFIX)
        goal = self.goal[:2]
        goal_distance = jp.linalg.norm(pos - goal)
        conditions = [
            goal_distance < 0.25,
            (goal_distance < 0.5) & (goal_distance >= 0.25),
            (goal_distance < 0.75) & (goal_distance >= 0.5),
            True  # Default case
        ]
        scale_values = [
            jp.array([0.8, 0.8]),  
            jp.array([0.85, 0.85]),  
            jp.array([0.95, 0.95]),
            jp.array([1.0, 1.0])   
        ]
        linear_scale, angular_scale = jp.select(conditions, scale_values)

        # 1) Global desired velocity given by gradient at current position
        map_pos = self.world_to_map(pos)
        dir = self._policy[map_pos[0], map_pos[1]]
        
        # 2) Desired LOCAL linear velocity (by rotating the world gradient)
        c, s = jp.cos(robot_yaw), jp.sin(robot_yaw)
        rot_matrix_transpose = jp.array([[c, s], [-s, c]])
        desired_vel_local_xy = rot_matrix_transpose @ dir
        desired_vel_local_xy = linear_scale * desired_vel_local_xy / (jp.linalg.norm(desired_vel_local_xy) * 2.0 + 1e-6)
        
        # 3) Desired LOCAL angular velocity (proportional to angle difference)
        desired_yaw = jp.arctan2(dir[1], dir[0])
        yaw_error = desired_yaw - robot_yaw
        yaw_error = jp.arctan2(jp.sin(yaw_error), jp.cos(yaw_error))
        desired_vel_local_w = angular_scale * yaw_error
        
        return jp.array([desired_vel_local_xy[0], desired_vel_local_xy[1], desired_vel_local_w])
    
    def _compute_greedy_policy(self, map: jp.ndarray) -> jp.ndarray:
        policy = jp.full((self.bins, self.bins, 2), -1, dtype=jp.float32)
        # Freeze obstacle and goal cells to a neutral direction (0,0)
        policy = jp.where((map == self.OBSTACLE)[..., jp.newaxis], jp.array([0, 0], dtype=jp.float32), policy)
        policy = jp.where((map == self.GOAL)[..., jp.newaxis], jp.array([0, 0], dtype=jp.float32), policy)

        max_cost = jp.max(map, where=(map != self.OBSTACLE) & (map != self.FREE), initial=0)
        max_cost_cell = jp.array([-1, -1])
        
        def compute_best_direction(policy: jp.ndarray, flat_idx: int) -> tuple[jp.ndarray, None]:
            cell = jp.array([flat_idx // self.bins, flat_idx % self.bins])
            minimum_cost = max_cost
            minimum_cost_cell = max_cost_cell
            
            for dir in self.directions:
                neighbor = cell + dir
                
                def evaluate_neighbor(operand: tuple[float, jp.ndarray]) -> tuple[float, jp.ndarray]:
                    minimum_cost, minimum_cost_cell = operand
                    return jax.lax.cond(
                        map[neighbor[0], neighbor[1]] <= minimum_cost,
                        lambda: (map[neighbor[0], neighbor[1]], neighbor),
                        lambda: (minimum_cost, minimum_cost_cell)
                    )
                    
                def no_op(operand: tuple[float, jp.ndarray]) -> tuple[float, jp.ndarray]:
                    minimum_cost, minimum_cost_cell = operand
                    return minimum_cost, minimum_cost_cell
                    
                minimum_cost, minimum_cost_cell = jax.lax.cond(
                    self.is_in_bounds(neighbor),
                    evaluate_neighbor,
                    no_op,
                    (minimum_cost, minimum_cost_cell)
                )
        
            direction = minimum_cost_cell - cell
            policy = policy.at[cell[0], cell[1]].set(direction)
            
            return policy, None
            
        policy, _ = jax.lax.scan(
            compute_best_direction,
            policy,
            jp.arange(self.bins * self.bins)
        )
        
        return policy
        
    def world_to_map(self, pos: jp.ndarray) -> jp.ndarray:
        """
        Convert world coordinates to map indices
        world: x points north, y points west
        discrete map: x are rows (points north), y are columns (points west)
        """
        pos = pos[:2]
        
        # Offset to map origin (assumed to be at the bottom right of the map for convention)
        pos -= self.origin
        
        # Scale to map indices
        map_x = jp.floor(pos[0] / self.bin_size).astype(int)
        map_y = jp.floor(pos[1] / self.bin_size).astype(int)
        
        # Clip to map bounds
        map_x = jp.clip(map_x, 0, self.bins - 1)
        map_y = jp.clip(map_y, 0, self.bins - 1)
        
        return jp.array([map_x, map_y])
    
    def map_to_world(self, map_pos: jp.ndarray, center: bool = True) -> jp.ndarray:
        offset = 0.5 if center else 0.0  # fraction of bin_size
        world_x = self.origin[0] + (map_pos[0] + offset) * self.bin_size
        world_y = self.origin[1] + (map_pos[1] + offset) * self.bin_size
        return jp.array([world_x, world_y])
    
    def get_map(self) -> jp.ndarray:
        return self._map
    
    def get_gradient(self) -> jp.ndarray:
        return self._gradient
        
    def _compute_costs(self, goal: jp.ndarray, obstacles: jp.ndarray) -> jp.ndarray:
        map = jp.full((self.bins, self.bins), self.FREE, dtype=float)  # Changed to float
        
        # Obstacles are marked with high cost
        for obs in obstacles:
            obstacle = self.world_to_map(obs)
            map = map.at[obstacle[0], obstacle[1]].set(self.OBSTACLE)
            for dir in self.cardinal_directions:
                neighbor = obstacle + dir
                if self.is_in_bounds(neighbor) and map[neighbor[0], neighbor[1]] != self.OBSTACLE:
                    map = map.at[neighbor[0], neighbor[1]].set(self.NEAR_OBSTACLE)
                    
        # Goal is marked with low cost (0)
        goal = self.world_to_map(goal)
        map = map.at[goal[0], goal[1]].set(self.GOAL)

        # BFS to fill in costs for free space
        map = self._dijkstra(map)
        self._map = map
        
        return map
    
    def is_in_bounds(self, cell: jp.ndarray) -> bool:
        return jp.all(jp.logical_and(cell >= 0, cell < self.bins))
    
    def get_value(self, pos: jp.ndarray, map: jp.ndarray) -> float:  # Changed return type to float
        """Get the value of a world position with respect to the map."""
        map_pos = self.world_to_map(pos)
        cost = map[map_pos[0], map_pos[1]]
        value = jp.pow(self.abs_gamma, cost)
        return value
    
    def _dijkstra(self, map: jp.ndarray) -> jp.ndarray:
        # Costs setup

        move_costs = jp.array([self.CARDINAL_COST] * 4 + [self.DIAGONAL_COST] * 4)
        cost_map = jp.full((self.bins, self.bins), jp.inf, dtype=jp.float32)
        cost_map = jp.where(map == self.GOAL, jp.array(self.GOAL, dtype=cost_map.dtype), cost_map)
        
        # obstacle map
        is_obstacle = (map == self.OBSTACLE)
        is_near_obstacle = (map == self.NEAR_OBSTACLE)

        # queue
        visited_mask = jp.zeros_like(map, dtype=bool)
        max_iterations = self.bins * self.bins
        
        def step(carry, _):
            cost_map, visited_mask = carry
            
            # 1. Find the unvisited cell with the lowest cost
            temp_costs = jp.where(visited_mask, jp.inf, cost_map)
            flat_idx = jp.argmin(temp_costs)
            current_pos = jp.array([flat_idx // self.bins, flat_idx % self.bins])
            current_cost = cost_map[current_pos[0], current_pos[1]]
            
            def update(cost_map, visited_mask):
                new_visited_mask = visited_mask.at[current_pos[0], current_pos[1]].set(True)
                
                def update_neighbor(inner_carry, i):
                    cm = inner_carry
                    direction, move_cost = self.directions[i], move_costs[i]
                    neighbor = current_pos + direction
                    
                    def update_cost(c):
                        new_cost = current_cost + move_cost
                        new_cost = jp.where(is_near_obstacle[neighbor[0], neighbor[1]], new_cost + self.NEAR_OBSTACLE_INFLATE, new_cost)
                        old_cost = c[neighbor[0], neighbor[1]]
                        return c.at[neighbor[0], neighbor[1]].set(jp.minimum(old_cost, new_cost))
                    
                    is_updatable = jp.logical_and(
                        self.is_in_bounds(neighbor),
                        jp.logical_and(
                            new_visited_mask[neighbor[0], neighbor[1]] == False,
                            is_obstacle[neighbor[0], neighbor[1]] == False
                        )
                    )
                    return jax.lax.cond(is_updatable, update_cost, lambda c: c, cm), None
                    
                updated_cost_map, _ = jax.lax.scan(
                    update_neighbor,
                    cost_map,
                    jp.arange(self.directions.shape[0])
                )
                
                return updated_cost_map, new_visited_mask

            new_carry = jax.lax.cond(
                current_cost == jp.inf,
                lambda c, v: (c, v),  # No more reachable cells
                update,
                cost_map,
                visited_mask,
            )
            return new_carry, None
            
        (final_cost_map, _ ), _ = jax.lax.scan(
            step,
            (cost_map, visited_mask),
            jp.arange(max_iterations),
        )
        
        # Keep costs as floats, no rounding
        final_map = jp.where(
            jp.isinf(final_cost_map), 
            self.OBSTACLE, 
            final_cost_map
        )
        
        return final_map
    
    
    def _compute_gradient(self, map: jp.ndarray) -> jp.ndarray:
        # Simple gradient descent on Dijkstra costs
        grad_x, grad_y = jp.gradient(-map, self.bin_size)
        grad_final = jp.stack([grad_x, grad_y], axis=-1)
        
        # Normalize
        norm = jp.linalg.norm(grad_final, axis=-1, keepdims=True)
        grad_final = grad_final / (norm + 1e-6)
        
        # Zero at goal and obstacles
        goal_idx = self.world_to_map(self.goal)
        grad_final = grad_final.at[goal_idx[0], goal_idx[1]].set(jp.array([0.0, 0.0]))
        grad_final = jp.where((map == self.OBSTACLE)[..., jp.newaxis], 0.0, grad_final)
        
        return grad_final

    def plot_map(
        self, 
        footstep_plan: FootstepPlan = None,
        zmp_trajectory: ZMPTrajectory = None,
        com_trajectory: COMTrajectory = None,
        filename="map.png",
        show_velocity_field: bool = False
    ):
        """Plot the map with footsteps and ZMP trajectory overlaid.
        
        Args:
            footstep_plan: Optional footstep plan to overlay
            zmp_trajectory: Optional ZMP trajectory to overlay
            com_trajectory: Optional COM trajectory to overlay
            filename: Output filename for the plot
            show_velocity_field: If True, display velocity arrows instead of cost numbers
        """
        if not pygame.get_init():
            pygame.init()

        # Font setup
        font_emoji_path = None
        for path in [
            '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',
            '/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf',
        ]:
            if os.path.exists(path):
                font_emoji_path = path
                break
        font_emoji = pygame.font.Font(font_emoji_path, 5) if font_emoji_path else pygame.font.SysFont("Arial", 5)
        font_text = pygame.font.SysFont("dejavusans", 24, bold=True)
        font_small = pygame.font.SysFont("dejavusans", 20, bold=True)

        # Flip map for display
        disp_map = jp.flip(self._map, (0, 1))
        disp_policy = jp.flip(self._policy, (0, 1))
        max_cost = jp.max(disp_map, where=(disp_map != self.OBSTACLE) & (disp_map != self.FREE), initial=0)
        
        height, width = disp_map.shape
        cell_size = 110
        screen_width = width * cell_size
        screen_height = height * cell_size
        surface = pygame.Surface((screen_width, screen_height))
        surface.fill((255, 255, 255))

        # Draw base map
        for i in range(height):
            for j in range(width):
                val = disp_map[i, j]
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                
                bg_color = (200, 200, 200)
                if val == self.OBSTACLE:
                    bg_color = (50, 50, 50)
                elif val != self.FREE:
                    normalized_cost = float(val) / max_cost if max_cost > 0 else 0
                    r = int(255 * normalized_cost)
                    g = int(255 * (1 - normalized_cost))
                    b = 0
                    bg_color = (r, g, b)
                
                pygame.draw.rect(surface, bg_color, rect)
                pygame.draw.rect(surface, (128, 128, 128), rect, 1)

                # Draw velocity field arrows or cost numbers
                if show_velocity_field:
                    # Get the policy vector for this cell (flipped coordinates)
                    policy_vec = disp_policy[i, j]
                    
                    # Only draw arrows for free cells (not obstacles or goal)
                    if val != self.OBSTACLE and val != self.GOAL and val != self.FREE:
                        # Calculate arrow properties
                        vec_x = -float(policy_vec[1])  # column direction (y in world)
                        vec_y = -float(policy_vec[0])  # row direction (x in world, negated for display)
                        magnitude = (vec_x**2 + vec_y**2)**0.5
                        
                        # Arrow parameters
                        arrow_length = cell_size * 0.35
                        arrow_head_size = 12
                        
                        # Normalize and scale
                        vec_x = vec_x / magnitude * arrow_length
                        vec_y = vec_y / magnitude * arrow_length
                        
                        # Start and end points
                        center_x, center_y = rect.center
                        end_x = center_x + vec_x
                        end_y = center_y + vec_y
                        
                        # Draw arrow shaft
                        pygame.draw.line(surface, (0, 0, 0), (center_x, center_y), (end_x, end_y), 3)
                        
                        # Draw arrowhead
                        angle = float(jp.arctan2(vec_y, vec_x))
                        angle1 = angle + 3.14159 * 0.75
                        angle2 = angle - 3.14159 * 0.75
                        
                        head_point1 = (
                            float(end_x + arrow_head_size * math.cos(angle1)),
                            float(end_y + arrow_head_size * math.sin(angle1))
                        )
                        head_point2 = (
                            float(end_x + arrow_head_size * math.cos(angle2)),
                            float(end_y + arrow_head_size * math.sin(angle2))
                        )
                            
                        pygame.draw.polygon(surface, (0, 0, 0), [(float(end_x), float(end_y)), head_point1, head_point2], int(math.floor(magnitude)))
                    elif val == self.GOAL:
                        # Still show goal emoji
                        text_surface = font_emoji.render("🎯", True, (50, 50, 50))
                        text_rect = text_surface.get_rect(center=rect.center)
                        surface.blit(text_surface, text_rect)
                    elif val == self.OBSTACLE:
                        # Still show obstacle emoji
                        text_surface = font_emoji.render("🧱", True, (50, 50, 50))
                        text_rect = text_surface.get_rect(center=rect.center)
                        surface.blit(text_surface, text_rect)
                else:
                    # Original behavior: show cost numbers
                    is_emoji = False
                    text_str = ""
                    if val == self.GOAL:
                        text_str, is_emoji = "🎯", True
                    elif val == self.OBSTACLE:
                        text_str, is_emoji = "🧱", True
                    elif val != self.FREE and footstep_plan is None:
                        text_str = f"{val:.1f}"

                    text_color = (50, 50, 50)
                    if text_str:
                        font_to_use = font_emoji if is_emoji else font_text
                        text_surface = font_to_use.render(text_str, True, text_color)
                        text_rect = text_surface.get_rect(center=rect.center)
                        surface.blit(text_surface, text_rect)

        def world_to_pixel(pos):
            """Convert world [x, y] to pixel [px, py]"""
            map_pos = pos[:2] - self.origin
            map_x = map_pos[0] / self.bin_size
            map_y = map_pos[1] / self.bin_size
            
            px = int((self.bins - map_y) * cell_size - cell_size / 2)
            py = int((self.bins - map_x) * cell_size - cell_size / 2)
            
            return px, py
        
        # Draw ZMP trajectory if provided
        if zmp_trajectory is not None:
            zmp_x = zmp_trajectory.zmp_midpoints_x
            zmp_y = zmp_trajectory.zmp_midpoints_y
            
            points = []
            for i in range(len(zmp_x)):
                px, py = world_to_pixel(jp.array([zmp_x[i], zmp_y[i]]))
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(surface, (0, 150, 255), False, points, 4)
            
        # Draw footsteps if provided
        if footstep_plan is not None:
            num_steps = footstep_plan.num_steps
            
            for i in range(num_steps):
                start_pose = footstep_plan.start_poses[i]
                end_pose = footstep_plan.end_poses[i]
                swing_foot = footstep_plan.swing_foot_ids[i]
                
                # Color based on foot (left = blue, right = red)
                color = (100, 100, 255) if swing_foot == 0 else (255, 100, 100)
                
                # Draw start position (hollow)
                px_start, py_start = world_to_pixel(start_pose)
                pygame.draw.circle(surface, color, (px_start, py_start), 10, 4)
                
                # Draw end position (filled)
                px_end, py_end = world_to_pixel(end_pose)
                pygame.draw.circle(surface, color, (px_end, py_end), 10)
                
                
        if com_trajectory is not None:
            com_x = com_trajectory.x_positions
            com_y = com_trajectory.y_positions
            points = []
            for i in range(len(com_x)):
                px, py = world_to_pixel(jp.array([com_x[i], com_y[i]]))
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(surface, (0, 0, 0), False, points, 4)

        pygame.image.save(surface, filename)
        pygame.quit()
        print(f"Saved map with plan to {filename}")
        
    
    def __str__(self) -> str:
        # Emojis and colors for better visualization
        GOAL_EMOJI = " 🎯 "
        OBSTACLE_EMOJI = " 🧱 "
        START_EMOJI = " 🤖 "
        UNREACHABLE = " .  "
        RESET_COLOR = "\033[0m"

        # Flip both axes so (0,0) appears in the top-left for human-readable output
        disp_map = jp.flip(self._map, (0, 1))

        # Find max cost for heatmap normalization, ignoring special values
        max_cost = jp.max(self._map, where=(self._map != self.OBSTACLE) & (self._map != self.FREE), initial=0)

        lines = []
        for r_idx, row in enumerate(disp_map.tolist()):
            line = []
            for c_idx, val in enumerate(row):
                if val == self.GOAL:
                    line.append(GOAL_EMOJI)
                elif val == self.OBSTACLE:
                    line.append(OBSTACLE_EMOJI)
                elif val == self.FREE:
                    line.append(UNREACHABLE)
                else:
                    # Create a heatmap from green (low cost) to red (high cost)
                    normalized_cost = val / max_cost if max_cost > 0 else 0
                    r = int(255 * normalized_cost)
                    g = int(255 * (1 - normalized_cost))
                    b = 0
                    # Use ANSI escape codes for background color
                    bg_color = f"\033[48;2;{r};{g};{b}m"
                    # Use black text for better readability on colored backgrounds
                    line.append(f"{bg_color}{jp.int32(val):^4}{RESET_COLOR}")  # Changed to float formatting
            lines.append("".join(line))
        return "\n".join(lines)
    
    def print_gradient_map(self, thresh: float = 1e-3) -> None:
        """Print compact ASCII gradient arrows (flipped to human view)."""
        # Flip only the spatial axes — do NOT flip the vector-component axis.
        disp_map = jp.flip(self._map, (0, 1))
        disp_grad = jp.flip(self._policy, (0, 1))

        arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
        lines = []
        for i, row in enumerate(disp_grad.tolist()):
            line = []
            for j, vec in enumerate(row):
                cell_val = float(disp_map[i, j])
                if cell_val == self.OBSTACLE:
                    line.append("o")
                    continue
                if cell_val == self.GOAL:
                    line.append("🎯")
                    continue
                # vec is [gy, gx] (gradient returned as axis-0, axis-1).
                gx = float(vec[1])
                gy = -float(vec[0])  # negate because we flipped rows -> invert vertical axis
                mag = (gx * gx + gy * gy) ** 0.5
                if mag < thresh:
                    line.append("·")
                    continue
                ang = jp.arctan2(gy, gx)  # radians in [-pi, pi]
                # map angle to 0..7 octants
                sector = int(((ang + jp.pi) / (2 * jp.pi) * 8)) % 8
                line.append(arrows[sector])
            lines.append(" ".join(line))
        print("\n".join(lines))
        
        
    def generate_scene_xml(self, goal: jp.ndarray = jp.array([3.0, 2.0, 0.01]), obstacles: jp.ndarray = jp.array([])) -> str:
        """
        Generate a MuJoCo XML scene based on Map parameters.
        - goal: Goal position (x, y, z).
        - obstacles: Array of obstacle positions (Nx3).
        """
        # Calculate arena extents
        half_width = (self.bins * self.bin_size) / 2
        wall_thickness = 0.04
        wall_height = 0.5
        
        # XML template with placeholders
        xml_template = f"""<mujoco model="t1 lowdim feetonly obstacle scene">
      <!-- Base robot definition -->
      <include file="t1_12dof.xml"/>
    
      <!-- Camera/statistics adjusted to arena -->
      <statistic center="0.0 0.0 0.7" extent="{half_width * 2 + 1.0}" meansize="0.04"/>
    
      <visual>
        <headlight diffuse="0.55 0.55 0.55" ambient="0.45 0.45 0.45" specular="0.35 0.35 0.35"/>
        <rgba force="1 0 0 1"/>
        <global azimuth="-130" elevation="-15"/>
        <map force="0.01"/>
        <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
        <quality shadowsize="8192"/>
      </visual>
    
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
        <texture type="2d" name="groundplane" builtin="checker" rgb1="0.5 0.5 0.5" rgb2="0.28 0.30 0.34" width="600" height="600"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="{int(1.0 / self.bin_size)} {int(1.0 / self.bin_size)}" reflectance="0"/>
        <material name="wall_mat" rgba="0.25 0.25 0.28 1"/>
        <material name="obstacle_mat" rgba="0.8 0.3 0.1 1"/>
        <material name="goal_mat" rgba="0 0.8 0.2 1"/>
      </asset>
    
      <worldbody>
        <!-- Ground plane -->
        <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" priority="1" friction="0.8" contype="1" conaffinity="2" condim="3" pos="0.0 0.0 0"/>
    
        <!-- Vertical walls (static, adjusted to map extent) -->
        <geom name="wall_pos_x" type="box" size="{wall_thickness} {half_width} {wall_height}" pos="{half_width} 0 {wall_height}" material="wall_mat"
              contype="1" conaffinity="2" friction="0.8 0.1 0.01"/>
        <geom name="wall_neg_x" type="box" size="{wall_thickness} {half_width} {wall_height}" pos="{-half_width} 0 {wall_height}" material="wall_mat"
              contype="1" conaffinity="2" friction="0.8 0.1 0.01"/>
        <geom name="wall_pos_y" type="box" size="{half_width} {wall_thickness} {wall_height}" pos="0 {half_width} {wall_height}" material="wall_mat"
              contype="1" conaffinity="2" friction="0.8 0.1 0.01"/>
        <geom name="wall_neg_y" type="box" size="{half_width} {wall_thickness} {wall_height}" pos="0 {-half_width} {wall_height}" material="wall_mat"
              contype="1" conaffinity="2" friction="0.8 0.1 0.01"/>
    
        <!-- Dynamic obstacles based on map -->
    """
        
        # Add obstacles from the map, indexed from 0 onward
        obstacle_xml = ""
        obstacle_index = 0
        for obstacle in self.obstacles:
            obstacle_xml += f"""        <geom name="obstacle_{obstacle_index}" type="cylinder" size="{self.bin_size / 2.5} {self.bin_size / 2.5}" pos="{obstacle[0]} {obstacle[1]} 0.1" material="obstacle_mat"
                    contype="1" conaffinity="2" friction="0.9 0.1 0.01"/>\n"""
            obstacle_index += 1
 
        
        # Add goal and start sites
        sites_xml = f"""
        <!-- GOAL & START SITES -->
        <site name="start" pos="0 0 0.01" size="0.05" type="sphere" rgba="0.2 0.2 0.9 0.6"/>
        <site name="goal" pos="{goal[0]} {goal[1]} {goal[2]}" size="0.08" type="sphere" material="goal_mat" rgba="0 0.8 0.2 0.9"/>
        </worldbody>
        
        <keyframe>
            <key name="home"
            qpos="
            0 0 0.665
            1 0 0 0
            -0.2 0 0 0.4 -0.3 0
            -0.2 0 0 0.4 -0.3 0
            "
            ctrl="
            -0.2 0 0 0.4 -0.3 0
            -0.2 0 0 0.4 -0.3 0
            "/>
        </keyframe>
        
        </mujoco>"""
        
        # Combine and write to file
        full_xml = xml_template + obstacle_xml + sites_xml
        # with open(f"/home/neverorfrog/code/loco_rlmpc/mujoco_playground/mujoco_playground/_src/locomotion/t1_12dof/xmls/scene_obstacle_avoidance.xml", "w") as f:
            # f.write(full_xml)
        print(f"Generated scene_obstacle_avoidance.xml with {self.bins}x{self.bins} tiles, bin_size={self.bin_size}")
        return full_xml
  
if __name__ == "__main__":
    map = Map()
    disp_map = jp.flip(map._map, (0, 1))
    disp_grad = jp.flip(map._gradient, (0, 1, 2))
    map.plot_map(show_velocity_field=True)

