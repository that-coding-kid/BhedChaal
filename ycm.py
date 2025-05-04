import pygame
import json
import random
import math
import numpy as np
import os
# Import of CrowdSpawner moved to after all necessary classes are defined

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
# Create two separate surfaces instead of one
main_screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Agent-Based Simulation")
# Second window (for heat map)
heat_map_screen = pygame.Surface((width, height))

# Colors
white = (255, 255, 255)
black = (0, 0, 0)  # Added black for heatmap background
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
gray = (100, 100, 100)
yellow = (255, 255, 0)
orange = (255, 165, 0) #for awareness

# Heat map colors - using a more continuous gradient with more colors
heat_colors = [
    (0, 0, 128),     # Dark Blue
    (0, 0, 255),     # Blue
    (0, 127, 255),   # Light Blue
    (0, 255, 255),   # Cyan
    (0, 255, 127),   # Teal
    (0, 255, 0),     # Green
    (127, 255, 0),   # Lime Green
    (255, 255, 0),   # Yellow
    (255, 127, 0),   # Orange
    (255, 0, 0),     # Red
    (255, 0, 127)    # Pink/Magenta
]

# Scale factor for coordinates
SCALE_FACTOR = 10  # Scale up the coordinates from the JSON file

# Cache for polygon containment tests
polygon_test_cache = {}
max_cache_size = 10000

class Particle:
    def __init__(self, x, y, is_density_group=False):
        # Position and velocity in 2D
        self.position = np.array([x, y], dtype=float)  # ri(t)
        self.velocity = np.array([0.0, 0.0], dtype=float)  # vi(t)
        self.acceleration = np.array([0.0, 0.0], dtype=float)  # ai(t)
        
        # Physical properties
        self.radius = 10  # Ri (increased for visibility)
        self.mass = 1.0  # mi
        self.color = (0, 100, 255)  # More distinct blue
        
        # Movement parameters
        self.desired_speed = 1500.0  # v0i (significantly increased)
        self.desired_direction = np.array([1.0, 0.0], dtype=float)  # Default right direction
        self.relaxation_time = 0.05  # τi (reduced for faster response)
        
        # Panic-related attributes
        self.panic_level = 0.0  # Pi(t) - panic level between 0 and 1
        self.panic_contagion_rate = 0.15  # α - increased rate at which panic spreads
        self.panic_decay_rate = 0.05  # β - rate at which panic naturally decreases
        self.panic_speed_factor = 5.5  # γ - increased factor for panic speed
        self.panic_force_factor = 2.0  # increased factor for panic forces
        self.panic_reaction_factor = 0.7  # factor by which reaction time decreases with panic
        self.panic_awareness_radius = 150  # increased radius for panic spread
        
        # Crowd surge attributes
        self.surge_force = 0.0  # Additional force during crowd surge
        self.surge_direction = np.array([0.0, 0.0], dtype=float)
        self.surge_decay = 0.98  # Slower decay for more persistent surge
        self.surge_wave_radius = 0  # Current radius of the surge wave
        self.surge_wave_speed = 100  # Speed of the surge wave propagation
        self.surge_origin = np.array([0.0, 0.0], dtype=float)  # Initialize surge origin
        
        # Psychological state
        self.aggressiveness = 0.5  # 0 to 1
        self.cooperativeness = 0.3  # Reduced from 0.5
        
        # Social relationships
        self.group_affiliation = None
        self.familiarity = {}  # agent_id: familiarity_level
        
        # Personal traits
        self.reaction_time = 100  # ms
        self.sensitivity_to_crowding = 0.8  # 0 to 1
        
        # Goals and targets
        self.target_position = None
        self.path = []
        self.id = None
        
        # Force model parameters
        self.A = 2000.0  # Repulsion strength
        self.B = 0.15    # Repulsion range
        self.C = 500.0   # Reduced attraction strength
        self.D = 0.1     # Attraction range
        self.k1 = 1.2e5  # Wall repulsion strength
        self.k2 = 2.4e5  # Wall repulsion range
        
        # State tracking
        self.panicked = False
        self.awareness = 50
        self.friction = 0.85  # Reduced friction for faster movement
        
        # Additional properties for density groups
        self.is_density_group = is_density_group
        if is_density_group:
            self.radius = 15  # Larger radius for density groups
            self.color = (0, 150, 255)  # Different color for density groups
            self.desired_speed = 12.0  # Slightly slower than individuals
            self.cooperativeness = 0.4  # Reduced group cohesion
            self.panic_contagion_rate = 0.2  # Groups are more susceptible to panic
            self.panic_awareness_radius = 180  # Larger awareness radius for groups

    def calculate_flow_direction(self, x, y):
        """Calculate the natural flow direction based on position"""
        # Define flow direction based on x position (left to right)
        if x < width * 0.3:
            return np.array([1.0, 0.0])  # Move right
        elif x > width * 0.7:
            return np.array([1.0, 0.0])  # Move right
        else:
            # In the middle, add some vertical movement
            return np.array([1.0, random.uniform(-0.2, 0.2)])

    def update_panic_level(self, particles):
        """Update panic level based on proximity to other panicked agents"""
        if self.panic_level > 0:
            # Natural decay of panic
            self.panic_level = max(0, self.panic_level - self.panic_decay_rate)
            
            # Check for panic propagation from nearby agents
            for other in particles:
                if other != self:
                    distance = np.linalg.norm(self.position - other.position)
                    if distance < self.panic_awareness_radius:
                        # Calculate weight based on distance and familiarity
                        distance_weight = 1 - (distance / self.panic_awareness_radius)
                        familiarity_weight = self.familiarity.get(other.id, 0.5)
                        weight = distance_weight * (1 + familiarity_weight)
                        
                        # Update panic level using the differential equation
                        panic_increase = (self.panic_contagion_rate * 
                                       weight * 
                                       other.panic_level * 
                                       (1 - self.panic_level))
                        self.panic_level = min(1.0, self.panic_level + panic_increase)
                        
                        # Update panic state
                        self.panicked = self.panic_level > 0.7

    def calculate_desired_force(self):
        """Calculate the desired force (driving force) with panic influence"""
        # Adjust desired speed based on panic level
        current_desired_speed = self.desired_speed * (1 + self.panic_speed_factor * self.panic_level)
        
        # Make direction more erratic when panicked
        if self.panicked:
            # Add some random variation to the desired direction
            random_angle = np.random.normal(0, 0.2 * self.panic_level)
            rotation_matrix = np.array([
                [np.cos(random_angle), -np.sin(random_angle)],
                [np.sin(random_angle), np.cos(random_angle)]
            ])
            self.desired_direction = rotation_matrix @ self.desired_direction
        
        desired_velocity = current_desired_speed * self.desired_direction
        force = (desired_velocity - self.velocity) / (self.relaxation_time * 
                (1 - self.panic_reaction_factor * self.panic_level))
        return force

    def calculate_repulsive_force(self, other):
        """Calculate repulsive force between two agents with panic influence"""
        diff = self.position - other.position
        distance = np.linalg.norm(diff)
        
        if distance == 0:
            return np.array([0.0, 0.0])
            
        # Calculate the effective distance
        effective_distance = distance - (self.radius + other.radius)
        
        if effective_distance < 0:
            # Direct collision - stronger repulsion when panicked
            direction = diff / distance
            panic_factor = 1 + self.panic_force_factor * (self.panic_level + other.panic_level) / 2
            return 1e5 * direction * panic_factor
        
        # If both are panicked, reduce repulsion to allow clustering
        if self.panicked and other.panicked:
            return np.array([0.0, 0.0])
            
        # Normal repulsion based on social force model
        direction = diff / distance
        # Increase repulsion strength based on panic levels
        panic_factor = 1 + self.panic_force_factor * (self.panic_level + other.panic_level) / 2
        force_magnitude = self.A * np.exp(-self.B * effective_distance) * panic_factor
        return force_magnitude * direction

    def calculate_attractive_force(self, other):
        """Calculate attractive force between two agents (for group cohesion)"""
        if other.id not in self.familiarity or self.familiarity[other.id] < 0.3:
            return np.array([0.0, 0.0])
            
        diff = other.position - self.position
        distance = np.linalg.norm(diff)
        
        if distance == 0:
            return np.array([0.0, 0.0])
            
        direction = diff / distance
        force_magnitude = self.C * np.exp(-self.D * distance)
        return force_magnitude * direction

    def calculate_polygon_force(self, polygon):
        """Calculate repulsive force from polygon boundary"""
        # Find closest point on polygon to agent
        min_distance = float('inf')
        closest_point = None
        closest_edge = None
        
        for i in range(len(polygon)):
            p1 = np.array(polygon[i])
            p2 = np.array(polygon[(i + 1) % len(polygon)])
            
            # Vector from p1 to p2
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)
            # Skip degenerate edges
            if edge_length < 0.0001:
                continue
                
            edge_direction = edge / edge_length
            
            # Vector from p1 to agent
            to_agent = self.position - p1
            
            # Project to_agent onto edge
            projection = np.dot(to_agent, edge_direction)
            projection = np.clip(projection, 0, edge_length)
            
            # Calculate closest point on edge
            closest = p1 + projection * edge_direction
            
            # Calculate distance to closest point
            distance = np.linalg.norm(self.position - closest)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = closest
                closest_edge = (p1, p2)
        
        if min_distance < self.k2:
            # Calculate force direction (away from edge)
            direction = self.position - closest_point
            direction_norm = np.linalg.norm(direction)
            
            # Only apply force if not exactly at the closest point
            if direction_norm > 0.0001:
                direction = direction / direction_norm
                
                # Calculate force magnitude
                force_magnitude = self.k1 * np.exp(-min_distance/self.k2)
                return force_magnitude * direction
        
        return np.array([0.0, 0.0])

    def update(self, walking_areas, particles):
        # Calculate net force
        net_force = np.array([0.0, 0.0])
        
        # Different behavior based on panic state
        if not self.panicked:
            # Normal movement behavior - not panicked
            # Add some randomness to movement direction
            random_angle = random.uniform(-0.2, 0.2)  # Small random angle
            rotation_matrix = np.array([
                [np.cos(random_angle), -np.sin(random_angle)],
                [np.sin(random_angle), np.cos(random_angle)]
            ])
            current_direction = rotation_matrix @ self.desired_direction
            
            # Strong directional force with increased strength
            directional_force = current_direction * self.desired_speed * 2000.0
            net_force += directional_force
            
            # OPTIMIZED: Only consider nearby agents using a distance threshold
            # This creates a simple spatial partitioning effect without full grid implementation
            nearby_particles = []
            interaction_radius = 100  # Only interact with particles within this radius
            
            for other in particles:
                if other == self:
                    continue
                    
                # Quick distance check using squared distance (avoids square root calculation)
                dx = self.position[0] - other.position[0]
                dy = self.position[1] - other.position[1]
                dist_squared = dx*dx + dy*dy
                
                if dist_squared < interaction_radius*interaction_radius:
                    nearby_particles.append(other)
            
            # Now only calculate repulsive forces for nearby particles
            for other in nearby_particles:
                net_force += self.calculate_repulsive_force(other) * 0.3
        else:
            # MODIFIED: Panic behavior - flee from panic center not simulation center
            # Calculate direction from panic center (surge_origin)
            from_panic_center = self.position - self.surge_origin
            distance_from_panic = np.linalg.norm(from_panic_center)
            
            if distance_from_panic > 0.0001:  # Only apply if not too close to panic center
                # Normalize direction
                flee_direction = from_panic_center / distance_from_panic
                
                # Repulsion force away from panic center increases with panic
                flee_force = flee_direction * self.desired_speed * 3000.0 * self.panic_level
                net_force += flee_force
            
            # Add surge wave effect
            if self.surge_wave_radius > 0:
                # Calculate distance from panic center
                distance_from_center = np.linalg.norm(self.position - self.surge_origin)
                wave_effect = max(0, 1 - abs(distance_from_center - self.surge_wave_radius) / 50)
                
                # Add surge force based on wave effect
                if wave_effect > 0:
                    surge_direction = self.position - self.surge_origin
                    surge_norm = np.linalg.norm(surge_direction)
                    if surge_norm > 0.0001:  # Avoid division by zero
                        surge_direction = surge_direction / surge_norm
                        net_force += surge_direction * wave_effect * 3000.0
            
            # Update surge wave radius
            self.surge_wave_radius += self.surge_wave_speed * 0.15  # Increased time step
            
            # OPTIMIZED: Only repel from nearby particles during panic
            nearby_particles = []
            interaction_radius = 150  # Panic has a larger interaction radius
            
            for other in particles:
                if other == self:
                    continue
                    
                # Quick distance check using squared distance
                dx = self.position[0] - other.position[0]
                dy = self.position[1] - other.position[1]
                dist_squared = dx*dx + dy*dy
                
                if dist_squared < interaction_radius*interaction_radius:
                    # Calculate repulsion force
                    distance = np.sqrt(dist_squared)  # Now we need the actual distance
                    
                    # FIXED: Avoid division by zero which causes NaN
                    if distance < 0.0001:  # Very small non-zero threshold
                        direction = np.array([1.0, 0.0])  # Default direction if too close
                    else:
                        direction = np.array([dx, dy]) / distance
                    
                    # Strength increases as distance decreases
                    strength = 2000.0 * (1 - distance/150) * self.panic_level
                    net_force += direction * strength
        
        # OPTIMIZED: Only check if we're near any boundary instead of all polygons
        # This reduces polygon force calculations which are expensive
        # Add polygon forces but only for nearby boundaries - use a simple AABB check first
        position_x, position_y = self.position[0], self.position[1]
        boundary_check_distance = 50  # Only check boundaries within this distance
        
        for area in walking_areas:
            min_x, min_y, max_x, max_y = get_polygon_bounds(area)
            
            # Quick AABB check first - only do detailed calculation if we're near the boundary
            if (position_x > min_x - boundary_check_distance and 
                position_x < max_x + boundary_check_distance and
                position_y > min_y - boundary_check_distance and 
                position_y < max_y + boundary_check_distance):
                
                # Only calculate polygon force if near the boundary
                inside = is_point_in_polygon((position_x, position_y), area)
                
                if inside:
                    # We're inside this area - check if we're near a boundary
                    min_distance = float('inf')
                    
                    for i in range(len(area)):
                        p1 = np.array(area[i])
                        p2 = np.array(area[(i + 1) % len(area)])
                        
                        # Use optimized point-to-line distance calculation
                        v = p2 - p1
                        w = self.position - p1
                        
                        c1 = np.dot(w, v)
                        if c1 <= 0:
                            # Before p1 - use distance to p1
                            distance = np.linalg.norm(self.position - p1)
                        else:
                            c2 = np.dot(v, v)
                            if c2 <= c1:
                                # After p2 - use distance to p2
                                distance = np.linalg.norm(self.position - p2)
                            else:
                                # Between p1 and p2 - use perpendicular distance
                                b = c1 / c2
                                pb = p1 + b * v
                                distance = np.linalg.norm(self.position - pb)
                        
                        if distance < min_distance:
                            min_distance = distance
                    
                    # Only apply force if we're near the boundary
                    if min_distance < boundary_check_distance:
                        net_force += self.calculate_polygon_force(area)
                else:
                    # We're outside but potentially near - calculate force directly
                    net_force += self.calculate_polygon_force(area)
        
        # Avoid top and bottom boundaries
        boundary_threshold = 50
        if self.position[1] < boundary_threshold:
            # Repel downward
            net_force += np.array([0.0, self.k1])
        elif self.position[1] > height - boundary_threshold:
            # Repel upward
            net_force += np.array([0.0, -self.k1])
        
        # Update acceleration (F = ma, assuming m = 1)
        self.acceleration = net_force
        
        # IMPROVED: Physics time step for smoother, faster animation
        physics_dt = 0.05  # Reduced from 0.1 to speed up simulation
        
        # Update velocity using Verlet integration with adjusted time step
        self.velocity += self.acceleration * physics_dt
        self.velocity *= self.friction
        
        # Limit maximum velocity to prevent unrealistic speeds but allow proper movement speed
        max_speed = 55.0 if self.panicked else 35.0
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > max_speed:
            self.velocity = self.velocity / current_speed * max_speed
            
        # Ensure minimum speed to prevent stuck agents
        min_speed = 6.0 if self.panicked else 4.0
        if 0 < current_speed < min_speed:
            self.velocity = self.velocity / current_speed * min_speed
        
        # Update position with adjusted time step
        self.position += self.velocity * physics_dt
        
        # OPTIMIZED: Simplified boundary handling with direct check
        # We only need to check if we're in any walking area once per update
        in_any_area = any(is_point_in_polygon((self.position[0], self.position[1]), area) for area in walking_areas)
        
        # If outside all walking areas, use simplified handling
        if not in_any_area and walking_areas:
            # Find closest point on any polygon boundary
            closest_point = None
            min_distance = float('inf')
            closest_normal = None
            
            for area in walking_areas:
                for i in range(len(area)):
                    p1 = np.array(area[i])
                    p2 = np.array(area[(i + 1) % len(area)])
                    
                    # Use the optimized point-to-line distance calculation from above
                    v = p2 - p1
                    w = self.position - p1
                    
                    c1 = np.dot(w, v)
                    if c1 <= 0:
                        # Before p1 - use distance to p1
                        distance = np.linalg.norm(self.position - p1)
                        closest = p1
                    else:
                        c2 = np.dot(v, v)
                        if c2 <= c1:
                            # After p2 - use distance to p2
                            distance = np.linalg.norm(self.position - p2)
                            closest = p2
                        else:
                            # Between p1 and p2 - use perpendicular distance
                            b = c1 / c2
                            closest = p1 + b * v
                            distance = np.linalg.norm(self.position - closest)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = closest
                        
                        # Calculate normal - perpendicular to the edge
                        edge_norm = np.linalg.norm(v)
                        # FIXED: Avoid division by zero
                        if edge_norm < 0.0001:
                            edge_direction = np.array([1.0, 0.0])  # Default direction for degenerate edges
                        else:
                            edge_direction = v / edge_norm
                            
                        normal = np.array([-edge_direction[1], edge_direction[0]])
                        
                        # Test if this normal points into the area
                        test_point = closest + normal * 5
                        if not is_point_in_polygon((test_point[0], test_point[1]), area):
                            normal = -normal
                            
                        closest_normal = normal
            
            # If we found a closest point, move there and bounce
            if closest_point is not None and closest_normal is not None:
                # Move slightly inside
                self.position = closest_point + closest_normal * 2
                
                # Bounce with simple reflection
                dot_product = np.dot(self.velocity, closest_normal)
                if dot_product < 0:  # Only bounce if moving toward boundary
                    self.velocity = self.velocity - 2 * dot_product * closest_normal
                    
                    # Add randomness when panicked
                    if self.panicked:
                        self.velocity += np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
        
        # Update path with less frequent saving
        if len(self.path) < 5 or np.linalg.norm(self.position - self.path[-1]) > 5:
            self.path.append(self.position.copy())
            if len(self.path) > 10:
                self.path.pop(0)
        
        # Update color based on panic state
        self.color = red if self.panicked else (0, 100, 255)

    def draw(self, surface):
        # FIXED: Ensure we never try to draw at NaN positions
        # Check if position contains NaN values
        if np.isnan(self.position[0]) or np.isnan(self.position[1]):
            # Reset to a valid position
            self.position = np.array([400.0, 300.0])  # Center of the screen
            self.velocity = np.array([0.0, 0.0])
            print("Warning: Particle had NaN position. Reset to center.")
            
        # Draw the agent
        pygame.draw.circle(surface, self.color, 
                         (int(self.position[0]), int(self.position[1])), 
                         self.radius)
        
        # Only draw awareness radius for panicked agents - REMOVED yellow circle
        if self.panicked:
            # Removed the drawing of the yellow circle here
            pass

class HeatMap:
    """Class to visualize density of agents in the simulation with fluid-like appearance"""
    def __init__(self, width, height, cell_size=20):  # Increased cell size for performance
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_width = width // cell_size + 1
        self.grid_height = height // cell_size + 1
        self.density_grid = np.zeros((self.grid_width, self.grid_height))
        self.max_density = 3.0  # Initial max density (will be adjusted dynamically)
        self.alpha = 255  # Full opacity for better fluid appearance
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.smoothing_factor = 2.0  # Reduced for faster calculation but still smooth
        self.influence_radius = 3  # Reduced radius for performance
        
        # Pre-compute the Gaussian kernel for smoothing
        self.kernel = self.gaussian_kernel(7, self.smoothing_factor)
        
        # Performance optimization - cache the color lookup
        self.color_cache = {}
        self.max_color_cache_size = 100
        
    def gaussian_kernel(self, size=7, sigma=2.0):
        """Create a 2D Gaussian kernel for smoothing"""
        # Create a coordinate grid
        x = np.linspace(-(size-1)/2, (size-1)/2, size)
        y = np.linspace(-(size-1)/2, (size-1)/2, size)
        x, y = np.meshgrid(x, y)
        
        # Calculate the 2D gaussian function
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize the kernel
        return kernel / np.sum(kernel)
        
    def apply_smoothing(self, grid):
        """Apply Gaussian smoothing to the density grid using the pre-computed kernel"""
        kernel_size = self.kernel.shape[0]
        
        # Create a padded version of the grid to handle edges
        padded = np.pad(grid, kernel_size//2, mode='constant')
        
        # Apply convolution using vectorized operations where possible
        smoothed = np.zeros_like(grid)
        
        # Use vectorized operations for the inner part of the grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # Extract the window around this cell
                window = padded[i:i+kernel_size, j:j+kernel_size]
                # Apply the kernel
                smoothed[i, j] = np.sum(window * self.kernel)
                
        return smoothed
    
    def update(self, particles):
        """Update the density grid based on current particle positions"""
        # Reset the grid
        self.density_grid = np.zeros((self.grid_width, self.grid_height))
        
        # OPTIMIZATION: Build a cell occupancy grid directly
        # Count particles in each cell instead of calculating Gaussian distribution for each particle
        for particle in particles:
            # Get the cell coordinates for this particle
            grid_x = int(particle.position[0] // self.cell_size)
            grid_y = int(particle.position[1] // self.cell_size)
            
            # Skip if outside the grid
            if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
                continue
                
            # Add 1.0 to the density at this cell
            self.density_grid[grid_x, grid_y] += 1.0
            
            # Add a bit to neighboring cells for smoother density
            radius = self.influence_radius
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    # Skip the center cell (that's handled above)
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        # Distance-based falloff
                        dist_sq = dx*dx + dy*dy
                        if dist_sq <= radius*radius:
                            # Use a simpler falloff function: 1/distance^2
                            falloff = 1.0 / (1.0 + dist_sq)
                            self.density_grid[nx, ny] += falloff * 0.5
        
        # Apply a single pass of Gaussian smoothing - that's enough for most cases
        self.density_grid = self.apply_smoothing(self.density_grid)
                
        # Update max density with smoothing
        current_max = np.max(self.density_grid)
        if current_max > self.max_density:
            self.max_density = current_max * 0.8 + self.max_density * 0.2  # Smooth the max increase
        elif current_max < self.max_density * 0.5:
            # Gradually decrease max if density drops significantly
            self.max_density = max(2.0, self.max_density * 0.95)
    
    def get_interpolated_color(self, density):
        """Get interpolated color based on density with caching for performance"""
        # Check the cache first
        # Round to 2 decimal places for better cache hits
        rounded_density = round(density * 100) / 100
        cache_key = rounded_density
        
        if cache_key in self.color_cache:
            return self.color_cache[cache_key]
        
        # Calculate the color
        color_ratio = min(0.999, density / self.max_density)
        
        # Map color_ratio to an index in heat_colors
        color_idx = color_ratio * (len(heat_colors) - 1)
        
        # Get the two colors to interpolate between
        idx_low = int(color_idx)
        idx_high = min(len(heat_colors) - 1, idx_low + 1)
        
        # Calculate the interpolation factor
        frac = color_idx - idx_low
        
        # Get the two colors
        color_low = heat_colors[idx_low]
        color_high = heat_colors[idx_high]
        
        # Interpolate between the colors
        r = int(color_low[0] * (1 - frac) + color_high[0] * frac)
        g = int(color_low[1] * (1 - frac) + color_high[1] * frac)
        b = int(color_low[2] * (1 - frac) + color_high[2] * frac)
        
        # Calculate alpha
        alpha_factor = min(1.0, (density / (self.max_density * 0.7)))
        alpha = int(255 * alpha_factor)
        
        # Cache the result
        color = (r, g, b, alpha)
        self.color_cache[cache_key] = color
        
        # Limit cache size
        if len(self.color_cache) > self.max_color_cache_size:
            # Remove a random key to keep the size in check
            self.color_cache.pop(next(iter(self.color_cache)))
            
        return color
    
    def draw(self, surface):
        """Draw the heat map with optimized rendering"""
        # Clear the surface
        self.surface.fill(black)
        
        # OPTIMIZATION: Draw fewer, larger circles for better performance
        # Draw a circle at each cell with density > 0
        sample_stride = 1  # Sample stride for performance
        
        for i in range(0, self.grid_width, sample_stride):
            for j in range(0, self.grid_height, sample_stride):
                density = self.density_grid[i, j]
                
                # Skip very low density areas
                if density < 0.1:
                    continue
                
                # Calculate position
                x = i * self.cell_size
                y = j * self.cell_size
                
                # Get color based on density
                color = self.get_interpolated_color(density)
                
                # Only draw if somewhat visible
                if color[3] > 10:
                    # Draw a circle with radius proportional to density
                    radius = min(self.cell_size//2, int(self.cell_size * density / self.max_density) + 2)
                    pygame.draw.circle(self.surface, color, (x, y), radius)
        
        # Draw the heat map surface on the main surface
        surface.blit(self.surface, (0, 0))
        
        # Draw legend
        self._draw_legend(surface)
    
    def _draw_legend(self, surface):
        """Draw a legend for the heat map"""
        legend_width = 20
        legend_height = 150
        x_pos = 10
        y_pos = 50  # Moved down slightly to avoid overlapping with title
        
        # Draw gradient using smoother transitions
        steps = 100  # Even more steps for smoother gradient
        for i in range(steps):
            # Get color using the same interpolation as main visualization
            color_ratio = 1.0 - (i / steps)  # Reverse order (hot at top)
            color_idx = color_ratio * (len(heat_colors) - 1)
            
            # Interpolate between colors
            idx_low = int(color_idx)
            idx_high = min(len(heat_colors) - 1, idx_low + 1)
            frac = color_idx - idx_low
            
            # Get colors
            color_low = heat_colors[idx_low]
            color_high = heat_colors[idx_high]
            
            # Interpolate
            r = int(color_low[0] * (1 - frac) + color_high[0] * frac)
            g = int(color_low[1] * (1 - frac) + color_high[1] * frac)
            b = int(color_low[2] * (1 - frac) + color_high[2] * frac)
            
            # Draw line segment
            y_pos_i = y_pos + (i * legend_height // steps)
            height_segment = legend_height // steps + 1  # +1 to avoid gaps
            pygame.draw.rect(surface, (r, g, b), 
                            (x_pos, y_pos_i, legend_width, height_segment))
        
        # Draw border
        pygame.draw.rect(surface, white, 
                        (x_pos, y_pos, legend_width, legend_height), 1)
        
        # Draw labels with white text on black background
        font = pygame.font.SysFont(None, 20)
        
        # High density label
        high_label = font.render(f"{self.max_density:.1f}", True, white)
        surface.blit(high_label, (x_pos + legend_width + 5, y_pos))
        
        # Medium density label
        med_label = font.render(f"{self.max_density/2:.1f}", True, white)
        surface.blit(med_label, (x_pos + legend_width + 5, y_pos + legend_height//2))
        
        # Low density label
        low_label = font.render("0.0", True, white)
        surface.blit(low_label, (x_pos + legend_width + 5, y_pos + legend_height))

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm with caching"""
    # Convert inputs to hashable types for caching
    point_tuple = (point[0], point[1])
    # Use the id of the polygon since the polygon itself isn't hashable
    polygon_id = id(polygon)
    
    # Create a cache key from the point and polygon id
    cache_key = (point_tuple, polygon_id)
    
    # Check if we already computed this test
    if cache_key in polygon_test_cache:
        return polygon_test_cache[cache_key]
    
    # If not in cache, calculate it using a faster implementation
    x, y = point
    
    # First, do a quick bounding box test
    min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
    if x < min_x or x > max_x or y < min_y or y > max_y:
        polygon_test_cache[cache_key] = False
        
        # Limit cache size
        if len(polygon_test_cache) > max_cache_size:
            # Just remove a random key if we reach the limit
            polygon_test_cache.pop(next(iter(polygon_test_cache)))
            
        return False
    
    # If inside the bounding box, do the full test
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    
    for i in range(n):
        p2x, p2y = polygon[i % n]
        # Crossing test - ray casting algorithm
        if ((p1y <= y and p2y > y) or (p1y > y and p2y <= y)) and (p1x <= x or p2x <= x):
            if p1x + (y - p1y) / (p2y - p1y) * (p2x - p1x) < x:
                inside = not inside
        p1x, p1y = p2x, p2y
    
    # Cache the result
    polygon_test_cache[cache_key] = inside
    
    # Limit cache size
    if len(polygon_test_cache) > max_cache_size:
        # Just remove a random key if we reach the limit
        polygon_test_cache.pop(next(iter(polygon_test_cache)))
        
    return inside

def get_polygon_center(polygon):
    """Calculate the center point of a polygon"""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

def get_polygon_bounds(polygon):
    """Get the bounding box of a polygon"""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def generate_points_in_polygon(polygon, num_points):
    """Generate random points within a polygon"""
    points = []
    min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
    
    while len(points) < num_points:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        if is_point_in_polygon((x, y), polygon):
            points.append((x, y))
    
    return points

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    particles_data = []
    walking_areas_data = []
    roads_data = []
    
    # Calculate scaling factors to fit the areas to our screen dimensions
    original_width = 1280  # Approximate width from the coordinates
    original_height = 720  # Approximate height from the coordinates
    
    # Calculate scaling factors
    width_scale = width / original_width
    height_scale = height / original_height
    
    if 'walking_areas' in data:
        for area in data['walking_areas']:
            # Scale the polygon points
            scaled_area = []
            for point in area:
                scaled_x = point[0] * width_scale
                scaled_y = point[1] * height_scale
                scaled_area.append([scaled_x, scaled_y])
            walking_areas_data.append(scaled_area)
    
    if 'roads' in data:
        for road in data['roads']:
            # Scale the polygon points
            scaled_road = []
            for point in road:
                scaled_x = point[0] * width_scale
                scaled_y = point[1] * height_scale
                scaled_road.append([scaled_x, scaled_y])
            roads_data.append(scaled_road)
    
    # Add extra people at the bottom of the screen with random distribution
    bottom_y_min = height - 150  # Minimum y position
    bottom_y_max = height - 50   # Maximum y position
    num_bottom_people = 40  # Number of people to add at bottom
    
    for _ in range(num_bottom_people):
        # Random position within bottom area
        x = random.uniform(50, width - 50)
        y = random.uniform(bottom_y_min, bottom_y_max)
        particles_data.append([x, y, False])
        
        # Add equivalent person on opposite side
        opposite_x = width - x
        particles_data.append([opposite_x, y, False])
    
    # Load object detection data
    objects_dir = 'video_data/objects'
    frame_files = sorted([f for f in os.listdir(objects_dir) if f.endswith('.json')])
    
    if frame_files:
        # Use the first frame to initialize individual agents
        first_frame = os.path.join(objects_dir, frame_files[0])
        with open(first_frame, 'r') as f:
            frame_data = json.load(f)
            
        if 'objects' in frame_data:
            for obj in frame_data['objects']:
                if obj['type'] == 'tracked_person':
                    # Use top_view coordinates and scale them
                    x = obj['top_view'][0] * width_scale
                    y = obj['top_view'][1] * height_scale
                    # Check if the point is in any walking area
                    in_walking_area = any(is_point_in_polygon((x, y), area) for area in walking_areas_data)
                    if in_walking_area:
                        particles_data.append([x, y, False])  # False for individual agents
                        
                        # Add equivalent number of people on the opposite side
                        opposite_x = width - x
                        if any(is_point_in_polygon((opposite_x, y), area) for area in walking_areas_data):
                            particles_data.append([opposite_x, y, False])
    
    # Load density data
    density_dir = 'video_data/density'
    density_files = sorted([f for f in os.listdir(density_dir) if f.endswith('.json')])
    
    if density_files:
        # Use the first frame to initialize density groups
        first_density = os.path.join(density_dir, density_files[0])
        with open(first_density, 'r') as f:
            density_data = json.load(f)
            
        if 'density_points' in density_data:
            # Group density points that are close to each other
            density_groups = []
            for point in density_data['density_points']:
                x = point['top_view'][0] * width_scale
                y = point['top_view'][1] * height_scale
                density_value = point['density_value']
                
                # Check if the point is in any walking area
                in_walking_area = any(is_point_in_polygon((x, y), area) for area in walking_areas_data)
                if in_walking_area:
                    # Determine number of agents based on density value
                    num_agents = 0
                    if density_value > 0.85:
                        num_agents = 15  # Increased number of agents
                    elif density_value > 0.5:
                        num_agents = 8   # Increased number of agents
                    else:
                        num_agents = 5   # Increased number of agents
                    
                    # Add agents around the density point
                    for _ in range(num_agents):
                        # Add some random offset to the position
                        offset_x = random.uniform(-20, 20)
                        offset_y = random.uniform(-20, 20)
                        particles_data.append([x + offset_x, y + offset_y, False])
                        
                        # Add equivalent number of people on the opposite side
                        opposite_x = width - (x + offset_x)
                        if any(is_point_in_polygon((opposite_x, y + offset_y), area) for area in walking_areas_data):
                            particles_data.append([opposite_x, y + offset_y, False])
    
    # If no particles were created, add some default ones within walking areas
    if not particles_data:
        for area in walking_areas_data:
            # Generate points within the polygon
            points = generate_points_in_polygon(area, 35)  # Increased number of points per area
            for x, y in points:
                particles_data.append([x, y, False])
                # Add equivalent number of people on the opposite side
                opposite_x = width - x
                if any(is_point_in_polygon((opposite_x, y), area) for area in walking_areas_data):
                    particles_data.append([opposite_x, y, False])
    
    return particles_data, walking_areas_data, roads_data

def draw_environment(surface, walking_areas, roads):
    # Draw walking areas
    for area in walking_areas:
        pygame.draw.polygon(surface, green, area)
    
    # Draw roads
    for road in roads:
        pygame.draw.polygon(surface, gray, road)

def inject_panic(particles, center_x, center_y, radius, injection_type='spatial'):
    """Inject panic into a subset of agents based on the specified method"""
    panic_center = np.array([center_x, center_y])
    
    if injection_type == 'spatial':
        # Spatial injection: affect agents within a radius
        for particle in particles:
            distance = np.linalg.norm(particle.position - panic_center)
            if distance < radius:
                # Panic level decreases with distance from center
                panic_factor = 1 - (distance / radius)
                particle.panic_level = min(1.0, particle.panic_level + 0.8 * panic_factor)
                particle.panicked = particle.panic_level > 0.7
                # Initialize surge wave parameters
                particle.surge_origin = panic_center
                particle.surge_wave_radius = 0
                
    elif injection_type == 'random':
        # Random injection: affect random percentage of agents
        num_agents = int(len(particles) * 0.2)  # Affect 20% of agents
        selected_agents = random.sample(particles, num_agents)
        
        # Choose a random position within the walking areas for panic origin
        if walking_areas:
            # Get a random area
            area = random.choice(walking_areas)
            # Get bounds
            x_coords = [p[0] for p in area]
            y_coords = [p[1] for p in area]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Try to find a valid position
            for _ in range(50):  # Try up to 50 times
                rand_x = random.uniform(min_x, max_x)
                rand_y = random.uniform(min_y, max_y)
                if is_point_in_polygon((rand_x, rand_y), area):
                    panic_center = np.array([rand_x, rand_y])
                    break
        
        for particle in selected_agents:
            particle.panic_level = 1.0
            particle.panicked = True
            particle.surge_origin = panic_center
            particle.surge_wave_radius = 0
            
    elif injection_type == 'targeted':
        # Targeted injection: affect specific agents
        # Choose a random position within the walking areas for panic origin
        if walking_areas:
            # Get a random area
            area = random.choice(walking_areas)
            # Get bounds
            x_coords = [p[0] for p in area]
            y_coords = [p[1] for p in area]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Try to find a valid position
            for _ in range(50):  # Try up to 50 times
                rand_x = random.uniform(min_x, max_x)
                rand_y = random.uniform(min_y, max_y)
                if is_point_in_polygon((rand_x, rand_y), area):
                    panic_center = np.array([rand_x, rand_y])
                    break
        
        for particle in particles:
            if particle.position[0] > width * 0.8:
                particle.panic_level = 1.0
                particle.panicked = True
                particle.surge_origin = panic_center
                particle.surge_wave_radius = 0
    
    # Print panic center location
    print(f"Panic originated at: ({panic_center[0]:.1f}, {panic_center[1]:.1f})")

def propagate_panic(particles):
    """Propagate panic through the crowd based on proximity and familiarity"""
    for particle in particles:
        if particle.panicked:
            for other in particles:
                if other != particle:
                    distance = np.linalg.norm(particle.position - other.position)
                    if distance < particle.awareness:
                        # Panic propagation is stronger between familiar agents
                        familiarity = particle.familiarity.get(other.id, 0.0)
                        panic_transfer = 0.1 * (1 + familiarity)
                        other.panic_level = min(1.0, other.panic_level + panic_transfer)
                        if other.panic_level > 0.7:
                            other.panicked = True

def update_group_behavior(particles):
    """Update group behavior and cohesion"""
    for particle in particles:
        if particle.group_affiliation is not None:
            # Find other group members
            group_members = [p for p in particles if p.group_affiliation == particle.group_affiliation]
            if len(group_members) > 1:
                # Calculate group center
                group_center = np.mean([p.position for p in group_members], axis=0)
                # Update desired direction towards group center
                direction = group_center - particle.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    particle.desired_direction = direction / distance

# Now that all necessary classes and functions are defined, import CrowdSpawner
from crowd_spawner import CrowdSpawner

def main():
    # Update the json_file path to use the areas data
    json_file = 'video_data/areas/areas_e48cdf5c9f4567cc315f28760929e2ac.json'
    particle_starts, walking_areas, roads = load_data(json_file)
    
    # Create particles with scaled coordinates
    particles = [Particle(x, y, is_density_group) for x, y, is_density_group in particle_starts]
    
    # IMPROVED: Add many more particles within walking areas for a denser crowd
    for area in walking_areas:
        # Generate points within the polygon - significantly increased count
        points = generate_points_in_polygon(area, 40)  # Increased from 25 to 40 points per area
        for x, y in points:
            particles.append(Particle(x, y, False))
    
    # ADDED: Additional random particles at the bottom of the screen
    bottom_y = height - 50  # Near the bottom edge
    for _ in range(30):  # Add 30 additional particles
        x = random.uniform(50, width - 50)
        y = random.uniform(bottom_y - 30, bottom_y)
        # Check if the point is within any walking area
        for area in walking_areas:
            if is_point_in_polygon((x, y), area):
                particles.append(Particle(x, y, False))
                break
    
    agent_count = len(particles)
    print(f"Created {agent_count} agents")
    print(f"Created {len(walking_areas)} walking areas")
    print(f"Created {len(roads)} roads")

    # Initialize agent properties
    for i, p in enumerate(particles):
        p.id = i
        p.familiarity = {other.id: random.random() * 0.5 for other in particles if other != p}
        # Assign random group affiliations
        if random.random() < 0.3:  # 30% chance of being in a group
            p.group_affiliation = random.randint(0, 3)  # 4 possible groups
        
        # Set random initial direction
        angle = random.uniform(0, 2 * math.pi)
        p.desired_direction = np.array([math.cos(angle), math.sin(angle)])

    # Initialize crowd spawner
    crowd_spawner = CrowdSpawner(width, height, walking_areas, Particle, is_point_in_polygon)

    # Initialize heat map
    heat_map = HeatMap(width, height, cell_size=20)
    
    # Create the second window
    pygame.display.set_caption("Agent Simulation | Heat Map")
    heat_window = pygame.display.set_mode((width * 2, height))

    running = True
    clock = pygame.time.Clock()
    panic_injected = False
    target_timer = 0
    simulation_time = 0
    panic_timer = 0  # Timer for panic injection

    # Print instructions
    print("Controls:")
    print("  SPACE: Inject panic at random location")
    print("  R: Inject random panic")
    print("  T: Inject targeted panic")
    
    # Set the target frame rate
    target_fps = 60
    # Initialize accumulated time
    accumulated_time = 0
    # Fixed simulation time step
    fixed_time_step = 1.0 / 60.0  # Reduced from 120Hz to 60Hz to improve performance
    
    # OPTIMIZATION: Pre-allocate a list for nearby particles to avoid creating it each frame
    # This variable will be reused to avoid garbage collection overhead
    shared_nearby_particles = []
    
    # OPTIMIZATION: Keep track of whether we need to redraw the static environment
    # This allows us to redraw only when necessary to improve performance
    redraw_environment = True
    
    # OPTIMIZATION: Process fewer particles per frame to maintain speed
    max_updates_per_frame = 3
    
    previous_frame_time = pygame.time.get_ticks() / 1000.0

    while running:
        # Calculate real time step
        current_time = pygame.time.get_ticks() / 1000.0
        frame_time = current_time - previous_frame_time
        previous_frame_time = current_time
        
        # Limit maximum frame time to prevent huge jumps
        if frame_time > 0.25:
            frame_time = 0.25
            
        # Add to accumulated time
        accumulated_time += frame_time
        
        # Allow panic injection to be used multiple times
        if pygame.key.get_pressed()[pygame.K_SPACE]:
            panic_timer += frame_time
            if panic_timer > 1.0:  # Can inject panic every second
                panic_injected = False
                panic_timer = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # All key events potentially change the simulation state
                redraw_environment = True
                
                if event.key == pygame.K_SPACE and not panic_injected:
                    # Generate a random panic center within a walking area
                    area = random.choice(walking_areas)
                    min_x, min_y, max_x, max_y = get_polygon_bounds(area)
                    
                    # Try to find a valid position within the walking area
                    max_attempts = 20
                    for _ in range(max_attempts):
                        rand_x = random.uniform(min_x, max_x)
                        rand_y = random.uniform(min_y, max_y)
                        if is_point_in_polygon((rand_x, rand_y), area):
                            # Inject panic at this random point
                            inject_panic(particles, rand_x, rand_y, 100, 'spatial')
                            panic_injected = True
                            print(f"Panic injected at position ({rand_x:.1f}, {rand_y:.1f})!")
                            break
                    
                    # Fallback if no point was found
                    if not panic_injected:
                        center_x, center_y = get_polygon_center(area)
                        inject_panic(particles, center_x, center_y, 100, 'spatial')
                        panic_injected = True
                        print(f"Fallback: Panic injected at area center ({center_x:.1f}, {center_y:.1f})!")
                elif event.key == pygame.K_r:
                    # Inject panic using random method
                    inject_panic(particles, 0, 0, 0, 'random')
                    print("Random panic injected!")
                elif event.key == pygame.K_t:
                    # Inject panic using targeted method
                    inject_panic(particles, 0, 0, 0, 'targeted')
                    print("Targeted panic injected!")

        # Clear screens - main screen white, heat map black for better contrast
        main_screen.fill(white)
        heat_map_screen.fill(black)
        
        # OPTIMIZATION: Only redraw environment when necessary
        if redraw_environment:
            # Draw environment on both screens
            draw_environment(main_screen, walking_areas, roads)
            draw_environment(heat_map_screen, walking_areas, roads)
            redraw_environment = False

        # Update crowd spawner - only redraw environment if new particles were added
        new_particles = crowd_spawner.update(particles)
        if new_particles:
            redraw_environment = True
            # Initialize properties for new particles
            for i, p in enumerate(new_particles):
                p.id = f"spawned_{len(particles) - len(new_particles) + i}"
                p.familiarity = {other.id: random.random() * 0.5 for other in particles if other != p}

        # OPTIMIZATION: Limit the number of physics updates per frame to prevent freezing
        # on slower computers or when there are many particles
        updates_this_frame = 0
        
        # Fixed timestep physics update
        # Process as many fixed physics updates as needed to catch up with real time
        # but don't do too many to ensure the application remains responsive
        while accumulated_time >= fixed_time_step and updates_this_frame < max_updates_per_frame:
            updates_this_frame += 1
            
            # OPTIMIZATION: Break up the particle updates to avoid having one extremely
            # long frame when there are too many particles
            # Process particles in batches
            batch_size = 50  # Process 50 particles at a time
            
            for start_idx in range(0, len(particles), batch_size):
                end_idx = min(start_idx + batch_size, len(particles))
                
                # Process this batch
                for particle in particles[start_idx:end_idx]:
                    # Update panic level
                    particle.update_panic_level(particles)
                    
                    # Update agent state
                    particle.update(walking_areas, particles)
            
            # Group behavior updates are relatively inexpensive
            update_group_behavior(particles)

            # Propagate panic - this is also not that expensive
            propagate_panic(particles)
            
            # Update targets periodically
            target_timer += fixed_time_step * 1000  # Convert to milliseconds
            if target_timer > 2000:  # Every 2 seconds
                # OPTIMIZATION: Only update targets for a fraction of particles each time
                # to spread the work across frames
                update_count = min(20, len(particles) // 5)  # Update at most 20 or 20% of particles
                particles_to_update = random.sample(particles, update_count)
                
                for particle in particles_to_update:
                    if particle.target_position is None:
                        # Set new target
                        if random.random() < 0.3:  # 30% chance to follow another agent
                            other = random.choice(particles)
                            if other != particle:
                                particle.target_position = other.position.copy()
                        else:
                            # Random target within screen bounds
                            particle.target_position = np.array([
                                random.randrange(50, width-50),
                                random.randrange(50, height-50)
                            ])
                
                # If we've updated all particles, reset the timer
                if update_count == len(particles):
                    target_timer = 0
            
            # Decrease accumulated time
            accumulated_time -= fixed_time_step
        
        # Draw particles
        for particle in particles:
            particle.draw(main_screen)

        # OPTIMIZATION: Don't update the heat map every frame, but every few frames
        # Update less frequently for better performance
        if int(simulation_time * 10) % 2 == 0:  # Update every 5 frames approximately
            heat_map.update(particles)
        
        # Always draw the heat map (even if it wasn't updated this frame)
        heat_map.draw(heat_map_screen)

        # Draw density stats on both screens
        font = pygame.font.SysFont(None, 20)
        text_color = (50, 50, 50)
        
        # Count panicked agents
        panic_count = sum(1 for p in particles if p.panicked)
        panic_text = font.render(f"Panicked: {panic_count}/{len(particles)}", True, text_color)
        main_screen.blit(panic_text, (width - 200, 10))
        heat_map_screen.blit(panic_text, (width - 200, 10))
        
        # Calculate average density
        max_density_text = font.render(f"Max density: {int(heat_map.max_density)}", True, text_color)
        main_screen.blit(max_density_text, (width - 200, 30))
        heat_map_screen.blit(max_density_text, (width - 200, 30))
        
        # Add titles to distinguish the views
        title_font = pygame.font.SysFont(None, 24)
        main_title = title_font.render("Regular View", True, (0, 0, 0))
        heat_title = title_font.render("Heat Map View", True, (0, 0, 0))
        main_screen.blit(main_title, (10, 10))
        heat_map_screen.blit(heat_title, (10, 10))
        
        # Add performance metrics
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.1f}", True, text_color)
        main_screen.blit(fps_text, (width - 200, 50))
        heat_map_screen.blit(fps_text, (width - 200, 50))
        
        particles_text = font.render(f"Particles: {len(particles)}", True, text_color)
        main_screen.blit(particles_text, (width - 200, 70))
        heat_map_screen.blit(particles_text, (width - 200, 70))
        
        frame_time_text = font.render(f"Frame time: {frame_time*1000:.1f}ms", True, text_color)
        main_screen.blit(frame_time_text, (width - 200, 90))
        heat_map_screen.blit(frame_time_text, (width - 200, 90))
        
        updates_text = font.render(f"Updates/frame: {updates_this_frame}/{max_updates_per_frame}", True, text_color)
        main_screen.blit(updates_text, (width - 200, 110))
        heat_map_screen.blit(updates_text, (width - 200, 110))

        # Combine both surfaces onto the window
        heat_window.blit(main_screen, (0, 0))
        heat_window.blit(heat_map_screen, (width, 0))

        # Update display
        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS for smooth animation
        simulation_time += 1/60  # Update simulation time

    pygame.quit()

if __name__ == "__main__":
    main()
