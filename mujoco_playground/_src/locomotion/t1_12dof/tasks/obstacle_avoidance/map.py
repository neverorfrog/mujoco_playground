import jax.numpy as jp
import jax
import pygame
import os

class Map:
    """Abstract discrete map class for obstacle avoidance tasks."""
    
    _map: jp.ndarray
    
    def __init__(self, bins: int, bin_size: float, goal: jp.ndarray, obstacles: jp.ndarray):
        assert bins > 0 and bin_size > 0, "bins and bin_size must be positive"
        self.bins = bins
        self.bin_size = bin_size
        self.goal = goal
        self.obstacles = obstacles
        self.width = self.bins * self.bin_size
        self.height = self.bins * self.bin_size
        self.origin = jp.array([ -self.width / 2, -self.height / 2 ])  # Bottom-right corner in world coordinates
        self.abs_gamma = 0.9
        self.FREE = -1.0  # Changed to float
        self.GOAL = 0.0   # Changed to float
        self.OBSTACLE = 1000.0  # Changed to large float instead of 2 * bins
        self._map = self._compute_costs(goal, obstacles)
        self._xml = self.generate_scene_xml(goal, obstacles)
        
        
        
    def world_to_map(self, pos: jp.ndarray) -> jp.ndarray:
        """
        Convert world coordinates to map indices
        world: x points north, y points west, z points up
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
    
    def get(self) -> jp.ndarray:
        return self._map
        
    def _compute_costs(self, goal: jp.ndarray, obstacles: jp.ndarray) -> jp.ndarray:
        map = jp.full((self.bins, self.bins), self.FREE, dtype=float)  # Changed to float
        
        # Obstacles and origin are marked with high cost
        for obs in obstacles:
            jax.debug.print("Obstacle at: {}", obs)
            obstacle = self.world_to_map(obs)
            map = map.at[obstacle[0], obstacle[1]].set(self.OBSTACLE)

        # Goal is marked with low cost (0)
        goal = self.world_to_map(goal)
        map = map.at[goal[0], goal[1]].set(self.GOAL)

        # BFS to fill in costs for free space
        map = self.dijkstra(map)
        self._map = map
        
        return map
    
    def is_in_bounds(self, cell: jp.ndarray) -> bool:
        return jp.all(jp.logical_and(cell >= 0, cell < self.bins))
    
    def get_cost(self, pos: jp.ndarray) -> float:  # Changed return type to float
        """Get the cost of a world position."""
        map_pos = self.world_to_map(pos)
        cost = self._map[map_pos[0], map_pos[1]]
        cost = 1.0 * (1.0 * jp.pow(self.abs_gamma, cost) - 1.0)
        return cost
    
    def get_cost(self, pos: jp.ndarray, map: jp.ndarray) -> float:  # Changed return type to float
        """Get the cost of a world position."""
        map_pos = self.world_to_map(pos)
        cost = map[map_pos[0], map_pos[1]]
        cost = jp.pow(self.abs_gamma, cost)
        return cost
    
    def dijkstra(self, map: jp.ndarray) -> jp.ndarray:
        # Costs setup
        CARDINAL_COST = 1.0
        DIAGONAL_COST = jp.sqrt(2)
        directions = jp.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],  # Cardinal
            [1, 1], [1, -1], [-1, 1], [-1, -1] # Diagonal
        ])
        move_costs = jp.array([CARDINAL_COST] * 4 + [DIAGONAL_COST] * 4)
        cost_map = jp.full((self.bins, self.bins), jp.inf, dtype=jp.float32)
        cost_map = jp.where(map == self.GOAL, jp.array(self.GOAL, dtype=cost_map.dtype), cost_map)
        
        # obstacle map
        is_obstacle = (map == self.OBSTACLE)
        
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
                    direction, move_cost = directions[i], move_costs[i]
                    neighbor = current_pos + direction
                    
                    def update_cost(c):
                        new_cost = current_cost + move_cost
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
                    jp.arange(directions.shape[0])
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
                if r_idx == self.bins // 2 and c_idx == self.bins // 2:
                    line.append(START_EMOJI)
                elif val == self.GOAL:
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
    
    
    def plot_map(self, filename="map.png"):
        """Plot the abstract map and save as PNG with colors, emojis, and high-contrast numbers."""
        # Initialize Pygame if not already
        if not pygame.get_init():
            pygame.init()

        # --- KEY CHANGE: Initialize two separate fonts ---
        # Font for Emojis
        font_emoji_path = None
        for path in [
            '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',
            '/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf',
        ]:
            if os.path.exists(path):
                font_emoji_path = path
                break
        font_emoji = pygame.font.Font(font_emoji_path, 5) if font_emoji_path else pygame.font.SysFont("Arial", 5)

        # Font for Text/Numbers (use a standard system font)
        font_text = pygame.font.SysFont("dejavusans", 24, bold=True)  # Smaller font for floats

        # Flip the map for display consistency (0,0 at top-left)
        disp_map = jp.flip(self._map, (0, 1))
        
        # Find max cost for heatmap normalization, ignoring special values
        max_cost = jp.max(disp_map, where=(disp_map != self.OBSTACLE) & (disp_map != self.FREE), initial=0)
        
        height, width = disp_map.shape
        cell_size = 110
        screen_width = width * cell_size
        screen_height = height * cell_size
        surface = pygame.Surface((screen_width, screen_height))
        surface.fill((255, 255, 255))

        center_i, center_j = height // 2, width // 2

        for i in range(height):
            for j in range(width):
                val = disp_map[i, j]
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                
                # Determine background color
                bg_color = (200, 200, 200) # Default to gray for unreachable cells
                if val == self.OBSTACLE:
                    bg_color = (50, 50, 50)  # Dark Gray for obstacle
                elif val != self.FREE:
                    normalized_cost = float(val) / max_cost if max_cost > 0 else 0
                    r = int(255 * normalized_cost)
                    g = int(255 * (1 - normalized_cost))
                    b = 0
                    bg_color = (r, g, b)
                
                pygame.draw.rect(surface, bg_color, rect)
                pygame.draw.rect(surface, (128, 128, 128), rect, 1) # Grid lines

                # --- KEY CHANGE: Select font and text based on cell content ---
                is_emoji = False
                text_str = ""
                if i == center_i and j == center_j:
                    text_str, is_emoji = "🤖", True
                elif val == self.GOAL:
                    text_str, is_emoji = "🎯", True
                elif val == self.OBSTACLE:
                    text_str, is_emoji = "🧱", True
                elif val != self.FREE:
                    text_str = f"{val:.1f}"  # Float formatting

                # Determine text color based on background brightness
                # r, g, b = bg_color
                # luminance = (0.299 * r + 0.587 * g + 0.114 * b)
                # text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)
                text_color = (50, 50, 50)

                # Render and draw the text using the appropriate font
                if text_str:
                    font_to_use = font_emoji if is_emoji else font_text
                    text_surface = font_to_use.render(text_str, True, text_color)
                    text_rect = text_surface.get_rect(center=rect.center)
                    surface.blit(text_surface, text_rect)
        
        pygame.image.save(surface, filename)
        pygame.quit()
        
        
    
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
        <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" priority="1" friction="0.8" contype="1" conaffinity="2" condim="3" pos="{self.bin_size/2.0} {self.bin_size/2.0} 0"/>
    
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
        for i in range(self.bins):
            for j in range(self.bins):
                if self._map[i, j] == self.OBSTACLE:
                    # Convert map indices back to world coords (center of bin)
                    world_x = (i - self.bins // 2) * self.bin_size + self.bin_size / 2
                    world_y = (j - self.bins // 2) * self.bin_size + self.bin_size / 2
                    obstacle_xml += f"""        <geom name="obstacle_{obstacle_index}" type="cylinder" size="{self.bin_size / 2.2} {self.bin_size / 2.2}" pos="{world_x} {world_y} 0.125" material="obstacle_mat"
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
        with open(f"/home/neverorfrog/code/loco_rlmpc/mujoco_playground/mujoco_playground/_src/locomotion/t1_12dof/xmls/scene_obstacle_avoidance.xml", "w") as f:
            f.write(full_xml)
        print(f"Generated scene_obstacle_avoidance.xml with {self.bins}x{self.bins} tiles, bin_size={self.bin_size}")
        return full_xml
  
if __name__ == "__main__":
    goal = jp.array([5.0, 0.0, 0.0])
    obstacles = jp.array([
        [2.5, 0.5, 0.0],
        [2.5, 0.0, 0.0],
        [2.5, -0.5, 0.0],
    ])
    map = Map(bins = 35, bin_size = 0.49, goal=goal, obstacles=obstacles)
    print(map)
    
    # U-shaped obstacle
    # obstacles = jp.array([
    #     [2.0, 1.0, 0.0],
    #     [2.0, 0.75, 0.0],
    #     [2.0, 0.5, 0.0],
    #     [2.0, 0.25, 0.0],
    #     [2.0, 0.0, 0.0],
    #     [2.0, -0.25, 0.0],
    #     [2.0, -0.5, 0.0],
    #     [2.0, -0.75, 0.0],
    #     [2.0, -1.0, 0.0],
        
    #     [1.75, 1.0, 0.0],
    #     [1.5, 1.0, 0.0],
    #     [1.25, 1.0, 0.0],
    #     [1.0, 1.0, 0.0],
    #     [0.75, 1.0, 0.0],
    #     [0.5, 1.0, 0.0],
    #     [0.25, 1.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [-0.25, 1.0, 0.0],
    #     [-0.5, 1.0, 0.0],
        
    #     [1.75, -1.0, 0.0],
    #     [1.5, -1.0, 0.0],
    #     [1.25, -1.0, 0.0],
    #     [1.0, -1.0, 0.0],
    #     [0.75, -1.0, 0.0],
    #     [0.5, -1.0, 0.0],
    #     [0.25, -1.0, 0.0],
    #     [0.0, -1.0, 0.0],
    #     [-0.25, -1.0, 0.0],
    #     [-0.5, -1.0, 0.0],
    # ])
    map.plot_map("test_map.png")
    
