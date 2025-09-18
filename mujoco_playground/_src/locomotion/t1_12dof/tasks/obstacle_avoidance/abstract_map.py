import jax.numpy as jp
import jax

class AbstractMap:
    """Abstract discrete map class for obstacle avoidance tasks."""
    
    _map: jp.ndarray
    
    def __init__(self, bins: int = 15, bin_size: float = 0.5):
        assert bins > 0 and bin_size > 0, "bins and bin_size must be positive"
        self.bins = bins
        self.bin_size = bin_size
        self.width = self.bins * self.bin_size
        self.height = self.bins * self.bin_size
        self.origin = jp.array([ -self.width / 2, -self.height / 2 ])  # Bottom-right corner in world coordinates
        self.abs_gamma = 0.9
        self.FREE = -1
        self.GOAL = 0
        self.OBSTACLE = 2 * bins
        self._map = self.reset(jp.array([0.0, 0.0]), jp.array([])) # Default map with no obstacles and goal at origin
        
        
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
        
    def reset(self, goal: jp.ndarray, obstacles: jp.ndarray) -> jp.ndarray:
        map = jp.full((self.bins, self.bins), self.FREE, dtype=int) 
        
        # Obstacles and origin are marked with high cost (2 * bins)
        for obs in obstacles:
            obs_x, obs_y = self.world_to_map(obs)
            map = map.at[obs_x, obs_y].set(self.OBSTACLE)

        # Goal is marked with low cost (0)
        goal = self.world_to_map(goal)
        map = map.at[goal[0], goal[1]].set(self.GOAL)

        # BFS to fill in costs for free space
        map = self.dijkstra(map)
        self._map = map
        
        return map
    
    def is_in_bounds(self, cell: jp.ndarray) -> bool:
        return jp.all(jp.logical_and(cell >= 0, cell < self.bins))
    
    def get_cost(self, pos: jp.ndarray) -> int:
        """Get the cost of a world position."""
        map_pos = self.world_to_map(pos)
        cost = self._map[map_pos[0], map_pos[1]]
        cost = 1.0 * (1.0 * jp.pow(self.abs_gamma, cost) - 1.0)
        return cost
    
    def get_cost(self, pos: jp.ndarray, map: jp.ndarray) -> int:
        """Get the cost of a world position."""
        map_pos = self.world_to_map(pos)
        cost = map[map_pos[0], map_pos[1]]
        cost = jp.pow(self.abs_gamma, cost)
        return cost
    
    def bfs(self, map: jp.ndarray) -> jp.ndarray:
        """Fill in all reachable free cells with a default cost using BFS."""
        max_iterations = self.bins * self.bins
        
        def step(current_map, i):
            # Create a boolean mask for cells that are part of the current frontier (cost == i)
            is_frontier = (current_map == i)
            
            # "Dilate" the frontier mask by one cell in each of the 4 cardinal directions.
            # This identifies all direct neighbors of the current frontier.
            # We use jp.roll for a vectorized shift of the entire map.
            dilated = jp.logical_or.reduce(jp.array([
                is_frontier,
                jp.roll(is_frontier, shift=1, axis=0),
                jp.roll(is_frontier, shift=-1, axis=0),
                jp.roll(is_frontier, shift=1, axis=1),
                jp.roll(is_frontier, shift=-1, axis=1),
            ]))
            
            should_update = jp.logical_and(
                dilated,
                current_map == self.FREE
            )
            
            new_map = jp.where(
                should_update,
                i + 1,
                current_map
            )
            
            return new_map, None
        
        final_map, _ = jax.lax.scan(
            step,
            map,
            jp.arange(max_iterations),
        )
        
        return final_map
    
    
    def dijkstra(self, map: jp.ndarray) -> jp.ndarray:
        # Costs setup
        CARDINAL_COST = 1.0
        DIAGONAL_COST = jp.sqrt(2.0)
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
        
        final_map = jp.where(
            jp.isinf(final_cost_map), 
            self.OBSTACLE, 
            jp.round(final_cost_map).astype(int)
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
                    line.append(f"{bg_color}{int(val):^4}{RESET_COLOR}")
            lines.append("".join(line))
        return "\n".join(lines)


if __name__ == "__main__":
    map = AbstractMap(bins = 25, bin_size = 0.25)
    obstacles = jp.array([])
    goal = jp.array([7.0, 0.0, 0.0])
    map.reset(goal, obstacles)
    print(map)
    
