import pygame
import numpy as np
import json
import os
import math
import random
import time
from pathlib import Path
from collections import defaultdict
import cv2

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (100, 100, 100)

# Heat map colors
HEAT_COLORS = [
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

class Agent:
    """
    Agent class for social force model simulation of crowd dynamics
    """
    def __init__(self, x, y, agent_id=None, is_density_point=False):
        # Position and velocity
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        
        # Physical properties
        self.is_density_point = is_density_point
        
        if is_density_point:
            # Density points are larger, distinctly colored
            self.radius = 5
            self.mass = 0.8
            self.color = (50, 150, 255)  # Yellow
            self.outline_color = (0, 0, 0)  # Black outline
            self.outline_width = 1
        else:
            # Regular agents (from object detection)
            self.radius = 5
            self.mass = 1.0
            self.color = (50, 150, 255)  # Blue
            self.outline_color = (0, 0, 0)  # Black outline
            self.outline_width = 1
            
            # Check if this is a secondary (upsampled) agent
            if agent_id and "_2" in str(agent_id):
                # Make upsampled agents slightly smaller and differently colored
                self.radius = 5
                self.color = (100, 180, 255)  # Lighter blue
        
        # Movement parameters
        self.desired_speed = 1.2
        self.desired_direction = np.array([1.0, 0.0], dtype=float)
        self.relaxation_time = 0.5
        
        # Panic-related attributes
        self.panic_level = 0.0  # [0.0-1.0]
        self.panic_contagion_rate = 0.1
        self.panic_decay_rate = 0.03
        self.panic_speed_factor = 1.5
        self.panic_force_factor = 1.5
        self.panic_reaction_factor = 0.4
        self.panic_awareness_radius = 25
        
        # Surge wave attributes (for stampede simulation)
        self.surge_wave_radius = 0
        self.surge_wave_speed = 50
        self.surge_origin = np.array([0.0, 0.0], dtype=float)
        
        # Social properties
        self.group_affiliation = None
        self.familiarity = {}  # agent_id: familiarity_level
        
        # State tracking
        self.panicked = False
        self.id = agent_id
        
        # Social force model parameters
        self.A = 2000.0  # Repulsion strength
        self.B = 0.08    # Repulsion range
        self.k1 = 1.2e5  # Wall repulsion strength
        self.k2 = 2.4e5  # Wall repulsion range

    def calculate_desired_force(self):
        """Calculate the desired force with panic influence"""
        # Adjust desired speed based on panic level
        current_desired_speed = self.desired_speed * (1 + self.panic_speed_factor * self.panic_level)
        
        # Make direction more erratic when panicked
        if self.panicked:
            # Add random variation to the desired direction
            random_angle = np.random.normal(0, 0.3 * self.panic_level)
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
        """Calculate repulsive force between agents"""
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

    def calculate_wall_force(self, polygon):
        """Calculate repulsive force from walls (polygon boundaries)"""
        # Find closest point on polygon to agent
        min_distance = float('inf')
        closest_point = None
        
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
            
            # Clamp projection to edge length
            projection = max(0, min(projection, edge_length))
            
            # Find closest point on edge
            closest = p1 + projection * edge_direction
            
            # Calculate distance to this edge
            distance = np.linalg.norm(self.position - closest)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = closest
        
        if closest_point is None:
            return np.array([0.0, 0.0])
            
        # Calculate effective distance
        effective_distance = min_distance - self.radius
        
        if effective_distance < 0:
            # Inside wall, strong repulsion
            direction = (self.position - closest_point) / min_distance
            return self.k2 * abs(effective_distance) * direction
        
        # Normal wall repulsion
        direction = (self.position - closest_point) / min_distance
        force_magnitude = self.k1 * np.exp(-2.0 * effective_distance)
        return force_magnitude * direction

    def update_panic_level(self, agents):
        """Update panic level based on proximity to other panicked agents"""
        # Base decay rate - reduced for more persistent panic
        decay_rate = self.panic_decay_rate * 0.5
        
        # Natural decay of panic (slower)
        if self.panic_level > 0:
            self.panic_level = max(0, self.panic_level - decay_rate)
        
        # Panic propagation calculation
        max_panic_increase = 0
        
        # Find panicked agents in proximity
        for other in agents:
            if other != self:
                distance = np.linalg.norm(self.position - other.position)
                
                # Increased awareness radius for better propagation
                awareness_radius = self.panic_awareness_radius * 1.5
                
                if distance < awareness_radius and other.panic_level > 0.2:
                    # Calculate weight based on distance - inverse square for stronger nearby effect
                    distance_weight = (1 - (distance / awareness_radius)) ** 2
                    
                    # Increase panic spread rate
                    contagion_rate = self.panic_contagion_rate * 2.0
                    
                    # Calculate potential panic increase from this agent
                    panic_increase = (contagion_rate * 
                                   distance_weight * 
                                   other.panic_level * 
                                   (1.0 - self.panic_level * 0.5))  # Allow panic to continue increasing
                    
                    # Keep track of maximum increase from any single agent
                    max_panic_increase = max(max_panic_increase, panic_increase)
        
        # Apply the maximum panic increase
        if max_panic_increase > 0:
            self.panic_level = min(1.0, self.panic_level + max_panic_increase)
            
            # Update panicked state - lower threshold for panicked state
            self.panicked = self.panic_level > 0.4  # Lower threshold to trigger panic behavior

    def update(self, agents, walking_areas):
        """Update agent position and velocity based on social forces"""
        # Calculate net force
        net_force = np.array([0.0, 0.0])
        
        if self.panicked:
            # Panicked behavior - Flee from panic source
            # Calculate direction from panic center
            from_panic_center = self.position - self.surge_origin
            distance_from_panic = np.linalg.norm(from_panic_center)
            
            if distance_from_panic > 0.0001:
                # Normalize direction and flee from panic source with a stronger force
                flee_direction = from_panic_center / distance_from_panic
                # Increase flee force significantly to make agents spread out more
                flee_force = flee_direction * self.desired_speed * 5.0 * self.panic_level
                net_force += flee_force
                
                # Update desired direction to match flee direction for more consistent movement
                self.desired_direction = flee_direction
                
            # Add random movement to simulate chaotic behavior
            random_angle = np.random.normal(0, 0.5 * self.panic_level)
            random_direction = np.array([
                np.cos(random_angle),
                np.sin(random_angle)
            ])
            random_force = random_direction * self.desired_speed * 2.0 * self.panic_level
            net_force += random_force
            
            # Add surge wave effect
            if self.surge_wave_radius > 0:
                distance_from_center = np.linalg.norm(self.position - self.surge_origin)
                wave_effect = max(0, 1 - abs(distance_from_center - self.surge_wave_radius) / 20)
                
                if wave_effect > 0:
                    surge_direction = self.position - self.surge_origin
                    surge_norm = np.linalg.norm(surge_direction)
                    if surge_norm > 0.0001:
                        surge_direction = surge_direction / surge_norm
                        net_force += surge_direction * wave_effect * 2.0
                        
            # Update surge wave radius
            self.surge_wave_radius += self.surge_wave_speed * 0.1
        else:
            # Normal behavior
            # Add desired force
            net_force += self.calculate_desired_force()
        
        # Repulsive forces from other agents
        for other in agents:
            if other != self:
                # Enhanced repulsion during panic to avoid clustering
                if self.panicked:
                    # Calculate stronger repulsion with other panicked agents to avoid clustering
                    repulsion = self.calculate_enhanced_panic_repulsion(other)
                else:
                    repulsion = self.calculate_repulsive_force(other)
                net_force += repulsion
        
        # Wall forces from walking area boundaries
        for area in walking_areas:
            wall_force = self.calculate_wall_force(area)
            # Enhance wall force during panic to prevent getting stuck at walls
            if self.panicked:
                wall_force *= (1.0 + 2.0 * self.panic_level)
            net_force += wall_force
            
        # Calculate acceleration (F = ma)
        self.acceleration = net_force / self.mass
        
        # Update velocity (with clamping to prevent extreme values)
        # Higher max speed for panicked agents
        max_speed = self.desired_speed * (4.0 if self.panicked else 1.5)
        self.velocity += self.acceleration * 0.1  # time step
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed
            
        # Less friction for panicked agents to maintain higher speeds
        friction = 0.9 if self.panicked else 0.95
        self.velocity *= friction
        
        # Update position
        self.position += self.velocity * 0.1  # time step
        
        # Ensure agent stays within the screen bounds
        self.position[0] = max(0, min(WIDTH, self.position[0]))
        self.position[1] = max(0, min(HEIGHT, self.position[1]))

    def draw(self, surface):
        """Draw the agent on the surface"""
        # Calculate color based on panic level
        if self.panic_level > 0:
            # Gradual color change to red based on panic level
            red = min(255, int(self.color[0] + (255 - self.color[0]) * self.panic_level))
            green = max(0, int(self.color[1] * (1 - self.panic_level * 0.8)))
            blue = max(0, int(self.color[2] * (1 - self.panic_level)))
            color = (red, green, blue)
        else:
            color = self.color
            
        # Draw agent outline first (slightly larger circle)
        outline_radius = self.radius + self.outline_width
        pygame.draw.circle(surface, self.outline_color, 
                         (int(self.position[0]), int(self.position[1])), 
                         outline_radius)
        
        # Draw agent circle
        pygame.draw.circle(surface, color, 
                         (int(self.position[0]), int(self.position[1])), 
                         self.radius)
        
        # Draw velocity vector if moving
        if np.linalg.norm(self.velocity) > 0.1:
            end_pos = self.position + self.velocity * 0.8  # Slightly longer vector
            pygame.draw.line(surface, BLACK, 
                           (int(self.position[0]), int(self.position[1])),
                           (int(end_pos[0]), int(end_pos[1])), 2)
            
        # Draw panic awareness radius when panicked
        if self.panicked:
            pygame.draw.circle(surface, (255, 60, 0), 
                             (int(self.position[0]), int(self.position[1])), 
                             int(self.panic_awareness_radius), 1)

    def calculate_enhanced_panic_repulsion(self, other):
        """Calculate enhanced repulsive force between agents during panic"""
        diff = self.position - other.position
        distance = np.linalg.norm(diff)
        
        if distance == 0:
            # Prevent division by zero
            return np.array([0.0, 0.0])
        
        # Calculate the effective distance
        effective_distance = distance - (self.radius + other.radius)
        
        if effective_distance < 0:
            # Direct collision - stronger repulsion when panicked
            direction = diff / distance
            panic_factor = 3.0 * (self.panic_level + other.panic_level) / 2
            return 2e5 * direction * (1.0 + panic_factor)
        
        # If both agents are panicked, increase repulsion to avoid clustering
        if self.panicked and other.panicked:
            panic_factor = 3.0 * (self.panic_level + other.panic_level) / 2
            direction = diff / distance
            force_magnitude = self.A * 2.0 * np.exp(-self.B * effective_distance * 0.5) * (1.0 + panic_factor)
            return force_magnitude * direction
        
        # Normal repulsion based on social force model
        direction = diff / distance
        # Increase repulsion strength based on panic levels
        panic_factor = 1.0 + self.panic_force_factor * (self.panic_level + other.panic_level) / 2
        force_magnitude = self.A * np.exp(-self.B * effective_distance) * panic_factor
        return force_magnitude * direction

class HeatMap:
    """Heat map visualization for crowd density"""
    def __init__(self, width, height, cell_size=20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_width = width // cell_size + 1
        self.grid_height = height // cell_size + 1
        self.grid = np.zeros((self.grid_height, self.grid_width))
        self.kernel = self.gaussian_kernel()
        self.alpha = 0.5  # Transparency level for the heat map
        
    def gaussian_kernel(self, size=5, sigma=1.0):
        """Create a gaussian kernel for smoothing"""
        x, y = np.mgrid[-size:size+1, -size:size+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()
        
    def update(self, agents):
        """Update heat map based on agent positions"""
        # Reset grid
        self.grid = np.zeros((self.grid_height, self.grid_width))
        
        # Add agents to grid
        for agent in agents:
            # Convert position to grid coordinates
            grid_x = min(int(agent.position[0] / self.cell_size), self.grid_width - 1)
            grid_y = min(int(agent.position[1] / self.cell_size), self.grid_height - 1)
            
            # Add to grid
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                # Higher value for density points and panicked agents
                value = 1.0
                if agent.is_density_point:
                    value = 1.5
                if agent.panic_level > 0:
                    value *= (1 + agent.panic_level)
                
                self.grid[grid_y, grid_x] += value
        
        # Apply smoothing
        from scipy.ndimage import convolve
        self.grid = convolve(self.grid, self.kernel, mode='constant')
        
    def draw(self, surface):
        """Draw heat map on surface with transparency"""
        # Create a transparent surface for the heat map
        heat_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Find max value for normalization
        max_val = np.max(self.grid)
        if max_val == 0:
            max_val = 1.0  # Avoid division by zero
            
        # Draw heat map cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Normalize value
                value = self.grid[y, x] / max_val
                
                if value > 0.05:  # Only draw cells with significant density
                    # Get color based on value
                    color = self.get_interpolated_color(value)
                    
                    # Add alpha channel for transparency (0-255)
                    alpha_value = int(min(255, 120 + 135 * value))  # Higher values are more opaque
                    color_with_alpha = (*color, alpha_value)
                    
                    # Draw rectangle
                    rect = pygame.Rect(
                        x * self.cell_size, 
                        y * self.cell_size,
                        self.cell_size, 
                        self.cell_size
                    )
                    pygame.draw.rect(heat_surface, color_with_alpha, rect)
        
        # Blit the heat map surface onto the main surface
        surface.blit(heat_surface, (0, 0))
        
        # Draw legend
        self._draw_legend(surface)
                    
    def get_interpolated_color(self, value):
        """Get interpolated color based on value (0.0-1.0)"""
        # Map value to color index
        index = value * (len(HEAT_COLORS) - 1)
        
        # Get lower and upper color indices
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(HEAT_COLORS) - 1)
        
        # Get interpolation factor
        factor = index - lower_idx
        
        # Get lower and upper colors
        lower_color = HEAT_COLORS[lower_idx]
        upper_color = HEAT_COLORS[upper_idx]
        
        # Interpolate RGB values
        r = int(lower_color[0] * (1 - factor) + upper_color[0] * factor)
        g = int(lower_color[1] * (1 - factor) + upper_color[1] * factor)
        b = int(lower_color[2] * (1 - factor) + upper_color[2] * factor)
        
        return (r, g, b)
    
    def _draw_legend(self, surface):
        """Draw heat map legend"""
        legend_width = 20
        legend_height = 200
        legend_x = self.width - legend_width - 10
        legend_y = 10
        
        # Draw legend background
        legend_bg = pygame.Surface((legend_width + 10, legend_height + 30), pygame.SRCALPHA)
        pygame.draw.rect(legend_bg, (255, 255, 255, 200), (0, 0, legend_width + 10, legend_height + 30))
        pygame.draw.rect(legend_bg, (0, 0, 0, 200), (0, 0, legend_width + 10, legend_height + 30), 1)
        surface.blit(legend_bg, (legend_x - 5, legend_y - 5))
        
        # Draw color gradient with transparency
        for i in range(legend_height):
            value = 1 - (i / legend_height)
            color = self.get_interpolated_color(value)
            # Add alpha to match the main visualization
            alpha_value = int(min(255, 120 + 135 * value))
            color_with_alpha = (*color, alpha_value)
            
            # Create a small surface for this line of the gradient
            line_surface = pygame.Surface((legend_width, 1), pygame.SRCALPHA)
            pygame.draw.line(line_surface, color_with_alpha, 
                           (0, 0), 
                           (legend_width, 0))
            surface.blit(line_surface, (legend_x, legend_y + i))
            
        # Draw labels
        font = pygame.font.SysFont(None, 24)
        
        # High density
        high_text = font.render("High", True, BLACK)
        surface.blit(high_text, (legend_x + legend_width + 5, legend_y))
        
        # Low density
        low_text = font.render("Low", True, BLACK)
        surface.blit(low_text, (legend_x + legend_width + 5, legend_y + legend_height - 15))


class PanicSimulation:
    """Main class for panic/stampede simulation"""
    def __init__(self):
        # Initialize displays
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Panic/Stampede Simulation")
        
        # Initialize data
        self.agents = []
        self.walking_areas = []
        self.roads = []
        self.perspective_points = []
        self.video_id = None
        
        # Initialize simulation state
        self.running = True
        self.paused = False
        self.show_heatmap = True
        self.clock = pygame.time.Clock()
        
        # Initialize heat map
        self.heat_map = HeatMap(WIDTH, HEIGHT)
        
        # Panic simulation parameters
        self.panic_source = None
        self.panic_active = False
        self.simulation_time = 0
        
        # Animation recording parameters
        self.recording = False
        self.video_writer = None
        self.fps = 30
        self.frame_counter = 0
        self.record_frequency = 2  # Record every 2 frames to get 30fps (60fps simulation / 2)
        
    def load_data_from_video(self, video_path):
        """Load data from a processed video's saved data"""
        # Extract video ID from path
        import hashlib
        if os.path.exists(video_path):
            stats = os.stat(video_path)
            video_id = f"{os.path.basename(video_path)}_{stats.st_size}_{int(stats.st_mtime)}"
            self.video_id = hashlib.md5(video_id.encode()).hexdigest()
            print(f"Loading data for video ID: {self.video_id}")
            
            # Load walking areas and perspective points
            if self.load_walking_areas(self.video_id) and self.load_perspective_points(self.video_id):
                # Load objects and density from the most recent frame saved
                frame_numbers = self.get_available_frame_numbers(self.video_id)
                if frame_numbers:
                    latest_frame = max(frame_numbers)
                    print(f"Loading data from frame {latest_frame}")
                    return self.load_objects_and_density(self.video_id, latest_frame)
            return False
        else:
            print(f"Error: Video file '{video_path}' not found")
            return False
        
    def get_available_frame_numbers(self, video_id):
        """Get a list of available frame numbers for the video ID"""
        frame_numbers = []
        objects_dir = os.path.join("video_data", "objects")
        if os.path.exists(objects_dir):
            prefix = f"objects_{video_id}_frame_"
            for filename in os.listdir(objects_dir):
                if filename.startswith(prefix) and filename.endswith(".json"):
                    try:
                        # Extract frame number from filename
                        frame_str = filename[len(prefix):-5]  # Remove prefix and .json
                        frame_number = int(frame_str)
                        frame_numbers.append(frame_number)
                    except ValueError:
                        continue
        return frame_numbers
        
    def load_walking_areas(self, video_id):
        """Load walking areas from saved data"""
        areas_file = os.path.join("video_data", "areas", f"areas_{video_id}.json")
        if os.path.exists(areas_file):
            try:
                with open(areas_file, 'r') as f:
                    areas_data = json.load(f)
                
                # Convert lists to numpy arrays
                self.walking_areas = [np.array(area) for area in areas_data.get("walking_areas", [])]
                self.roads = [np.array(road) for road in areas_data.get("roads", [])]
                
                print(f"Loaded {len(self.walking_areas)} walking areas and {len(self.roads)} roads")
                return True
            except Exception as e:
                print(f"Error loading areas: {e}")
                return False
        else:
            print(f"No areas file found for video ID: {video_id}")
            return False
        
    def load_perspective_points(self, video_id):
        """Load perspective transformation points from saved data"""
        perspective_file = os.path.join("video_data", "perspective", f"perspective_{video_id}.json")
        if os.path.exists(perspective_file):
            try:
                with open(perspective_file, 'r') as f:
                    perspective_data = json.load(f)
                
                # Convert lists to numpy arrays
                self.perspective_points = [np.array(point) for point in perspective_data.get("perspective_points", [])]
                
                print(f"Loaded {len(self.perspective_points)} perspective points")
                
                # Apply perspective transformation to walking areas and roads
                if self.perspective_points:
                    self.transform_areas()
                    return True
                return False
            except Exception as e:
                print(f"Error loading perspective points: {e}")
                return False
        else:
            print(f"No perspective file found for video ID: {video_id}")
            return False
        
    def transform_areas(self):
        """Apply perspective transformation to areas using the loaded perspective points"""
        if len(self.perspective_points) != 4:
            print("Error: Need exactly 4 perspective points for transformation")
            return
            
        # Destination points for top-view (800x600 canvas with margins)
        margin = 50
        dst_points = np.array([
            [margin, margin],  # Top-left
            [WIDTH - margin, margin],  # Top-right
            [WIDTH - margin, HEIGHT - margin],  # Bottom-right
            [margin, HEIGHT - margin]  # Bottom-left
        ], dtype=np.float32)
        
        # Compute perspective transformation matrix
        src_points = np.array(self.perspective_points, dtype=np.float32)
        transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Transform walking areas
        transformed_walking_areas = []
        for area in self.walking_areas:
            transformed_area = []
            for point in area:
                # Convert to homogeneous coordinates
                homogeneous_point = np.array([point[0], point[1], 1.0])
                # Apply transformation
                transformed = transformation_matrix.dot(homogeneous_point)
                # Convert back from homogeneous coordinates
                transformed /= transformed[2]
                transformed_area.append([int(transformed[0]), int(transformed[1])])
            transformed_walking_areas.append(np.array(transformed_area))
        
        # Transform roads
        transformed_roads = []
        for road in self.roads:
            transformed_road = []
            for point in road:
                # Convert to homogeneous coordinates
                homogeneous_point = np.array([point[0], point[1], 1.0])
                # Apply transformation
                transformed = transformation_matrix.dot(homogeneous_point)
                # Convert back from homogeneous coordinates
                transformed /= transformed[2]
                transformed_road.append([int(transformed[0]), int(transformed[1])])
            transformed_roads.append(np.array(transformed_road))
        
        # Replace original areas with transformed ones
        self.walking_areas = transformed_walking_areas
        self.roads = transformed_roads
        
        print("Applied perspective transformation to areas")
        
    def load_objects_and_density(self, video_id, frame_number):
        """Load detected objects and density points from a specific frame"""
        # Load objects
        objects_file = os.path.join("video_data", "objects", f"objects_{video_id}_frame_{frame_number:06d}.json")
        density_file = os.path.join("video_data", "density", f"density_{video_id}_frame_{frame_number:06d}.json")
        
        objects_loaded = False
        density_loaded = False
        
        # Clear existing agents
        self.agents = []
        
        # Load objects
        if os.path.exists(objects_file):
            try:
                with open(objects_file, 'r') as f:
                    objects_data = json.load(f)
                
                # Create agents from detected objects - UPSAMPLING by creating 2 agents per object
                for obj in objects_data.get("objects", []):
                    # Get position in top view
                    top_view = obj.get("top_view", [0, 0])
                    
                    # Create primary agent at exact position
                    agent = Agent(
                        x=top_view[0],
                        y=top_view[1],
                        agent_id=obj.get("id"),
                        is_density_point=False
                    )
                    
                    # Set velocity if available
                    if "vector" in obj:
                        vector = obj.get("vector", [0, 0])
                        agent.velocity = np.array(vector, dtype=float)
                        
                        # Set desired direction based on velocity
                        vel_norm = np.linalg.norm(agent.velocity)
                        if vel_norm > 0.1:
                            agent.desired_direction = agent.velocity / vel_norm
                    
                    self.agents.append(agent)
                    
                    # Create second agent slightly offset (upsampling)
                    offset_x = random.uniform(-15, 15)
                    offset_y = random.uniform(-15, 15)
                    second_agent = Agent(
                        x=top_view[0] + offset_x,
                        y=top_view[1] + offset_y,
                        agent_id=f"{obj.get('id')}_2" if obj.get('id') else None,
                        is_density_point=False
                    )
                    
                    # Copy velocity and direction from first agent
                    if "vector" in obj:
                        second_agent.velocity = agent.velocity.copy()
                        second_agent.desired_direction = agent.desired_direction.copy()
                    
                    self.agents.append(second_agent)
                
                objects_loaded = True
                print(f"Loaded and upsampled {len(objects_data.get('objects', []))} objects to {len(objects_data.get('objects', [])) * 2} agents")
            except Exception as e:
                print(f"Error loading objects: {e}")
        
        # Load density points
        if os.path.exists(density_file):
            try:
                with open(density_file, 'r') as f:
                    density_data = json.load(f)
                
                # DOWNSAMPLING - Create agents from density points with filtering
                density_points = density_data.get("density_points", [])
                # Downsample by taking every other point to reduce density
                downsampled_points = density_points[::2]  # Take every second point
                
                # Create agents from the downsampled points
                for point in downsampled_points:
                    # Check if it's a valid density point
                    if "top_view" in point and point.get("type", "") == "density_point":
                        top_view = point.get("top_view", [0, 0])
                        
                        # Create agent
                        agent = Agent(
                            x=top_view[0],
                            y=top_view[1],
                            agent_id=f"density_{len(self.agents)}",
                            is_density_point=True
                        )
                        
                        # Set density value if available
                        if "density_value" in point:
                            # Higher density = higher panic susceptibility
                            density_value = point.get("density_value", 0.0)
                            agent.panic_contagion_rate = 0.15 + 0.1 * density_value
                        
                        self.agents.append(agent)
                
                density_loaded = True
                print(f"Loaded and downsampled {len(density_points)} density points to {len(downsampled_points)} agents")
            except Exception as e:
                print(f"Error loading density points: {e}")
        
        # Initialize agent properties
        for i, agent in enumerate(self.agents):
            if agent.id is None:
                agent.id = f"agent_{i}"
                
            # Randomize initial movement direction slightly
            angle = random.uniform(-0.2, 0.2)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            agent.desired_direction = rotation_matrix @ agent.desired_direction
            
            # Initialize familiarity with a few random agents
            agent.familiarity = {}
            num_familiar = min(5, len(self.agents) // 10)
            for _ in range(num_familiar):
                other_idx = random.randint(0, len(self.agents) - 1)
                if other_idx != i:
                    other = self.agents[other_idx]
                    agent.familiarity[other.id] = random.random() * 0.7
        
        return objects_loaded or density_loaded
        
    def inject_panic(self, x, y, radius=50):
        """Inject panic at the specified location with a given radius"""
        if not self.agents:
            print("No agents to panic!")
            return
            
        self.panic_source = (x, y)
        self.panic_active = True
        
        # Count initially panicked agents
        panic_count = 0
        
        # Find agents within radius of panic source
        for agent in self.agents:
            distance = np.linalg.norm(agent.position - np.array([x, y]))
            if distance <= radius:
                # Set panic level based on distance to source - stronger initial panic
                panic_level = min(1.0, 1.5 - (distance / radius) * 0.8)
                agent.panic_level = max(agent.panic_level, panic_level)
                agent.panicked = agent.panic_level > 0.4  # Lower threshold
                
                # Set surge origin
                agent.surge_origin = np.array([x, y])
                agent.surge_wave_radius = 0
                
                # Count panicked agents
                panic_count += 1
        
        # Force some minimum number of agents to panic if only a few were in radius
        if panic_count < 5:
            # Sort agents by distance to panic source
            sorted_agents = sorted(
                self.agents, 
                key=lambda a: np.linalg.norm(a.position - np.array([x, y]))
            )
            
            # Force at least 5 agents to panic
            for agent in sorted_agents[:min(5, len(sorted_agents))]:
                if agent.panic_level < 0.5:
                    distance = np.linalg.norm(agent.position - np.array([x, y]))
                    agent.panic_level = min(1.0, 1.0 - (distance / (radius * 2)) * 0.5)
                    agent.panicked = True
                    agent.surge_origin = np.array([x, y])
                    agent.surge_wave_radius = 0
        
        print(f"Panic injected at ({x}, {y}) with radius {radius}, affecting {panic_count} agents")
        
    def propagate_panic(self):
        """Propagate panic through the crowd"""
        if not self.panic_active:
            return
            
        # Update panic levels for all agents
        for agent in self.agents:
            agent.update_panic_level(self.agents)
            
    def start_recording(self, duration=60, output_dir="panic_sim_results"):
        """Start recording the simulation as a 30fps animation
        
        Args:
            duration: Duration of the recording in seconds (default: 60)
            output_dir: Directory to save the animation
        """
        if self.recording:
            print("Already recording!")
            return
            
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video_path = os.path.join(output_dir, f"panic_sim_animation_{timestamp}.mp4")
        
        self.video_writer = cv2.VideoWriter(
            video_path, 
            fourcc, 
            self.fps, 
            (WIDTH, HEIGHT)
        )
        
        if not self.video_writer.isOpened():
            print("Failed to create video writer")
            return
            
        self.recording = True
        self.frame_counter = 0
        self.recording_duration = duration * self.fps  # Convert seconds to frames at 30fps
        
        print(f"Started recording animation to {video_path}")
        print(f"Recording will stop after {duration} seconds")
        
    def stop_recording(self):
        """Stop recording the animation"""
        if not self.recording:
            return
            
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            print("Recording stopped")
            
        self.recording = False
        self.video_writer = None
        
    def record_frame(self):
        """Record the current frame for the animation"""
        if not self.recording or self.video_writer is None:
            return
            
        # Only record every record_frequency frames to achieve 30fps
        if self.frame_counter % self.record_frequency == 0:
            # Convert Pygame surface to OpenCV image
            frame_data = pygame.surfarray.array3d(self.screen)
            frame_data = frame_data.transpose([1, 0, 2])  # Transpose to get correct order
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            
            # Write frame
            self.video_writer.write(frame_data)
            
            # Check if we've reached the recording duration
            if self.recording_duration > 0 and self.frame_counter >= self.recording_duration * self.record_frequency:
                self.stop_recording()
                print("Recording completed")
                
        self.frame_counter += 1
        
    def run_simulation(self):
        """Run the main simulation loop"""
        if not self.agents:
            print("No agents loaded. Please load data first.")
            return
            
        # Main simulation loop
        self.running = True
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Skip updates if paused
            if not self.paused:
                # Update simulation time
                self.simulation_time += 1
                
                # Propagate panic
                if self.panic_active:
                    self.propagate_panic()
                
                # Update agent positions
                for agent in self.agents:
                    agent.update(self.agents, self.walking_areas)
                
                # Update heat map
                if self.show_heatmap:
                    self.heat_map.update(self.agents)
            
            # Render simulation
            self.render()
            
            # Record frame if recording
            self.record_frame()
            
            # Cap the frame rate
            self.clock.tick(60)
            
        # Make sure to stop recording before quitting
        if self.recording:
            self.stop_recording()
            
        pygame.quit()
        
    def render(self):
        """Render the current simulation state"""
        # Clear screen
        self.screen.fill(WHITE)
        
        # Draw environment (roads and walking areas) first
        for road in self.roads:
            pygame.draw.polygon(self.screen, (200, 200, 255), road)
            pygame.draw.polygon(self.screen, BLUE, road, 2)
            
        for area in self.walking_areas:
            pygame.draw.polygon(self.screen, (200, 255, 200), area)
            pygame.draw.polygon(self.screen, GREEN, area, 2)
        
        # Draw heat map above the walkable areas if enabled
        if self.show_heatmap:
            self.heat_map.draw(self.screen)
        
        # Draw agents
        for agent in self.agents:
            agent.draw(self.screen)
        
        # Draw panic source if active
        if self.panic_active and self.panic_source:
            pygame.draw.circle(self.screen, RED, self.panic_source, 10)
            pygame.draw.circle(self.screen, ORANGE, self.panic_source, 50, 2)
        
        # Draw simulation info
        font = pygame.font.SysFont(None, 24)
        
        # Draw simulation time
        time_text = font.render(f"Time: {self.simulation_time // 60}s", True, BLACK)
        self.screen.blit(time_text, (10, 10))
        
        # Draw agent count
        agent_text = font.render(f"Agents: {len(self.agents)}", True, BLACK)
        self.screen.blit(agent_text, (10, 40))
        
        # Draw panic count
        panic_count = sum(1 for agent in self.agents if agent.panicked)
        panic_text = font.render(f"Panicked: {panic_count}", True, RED)
        self.screen.blit(panic_text, (10, 70))
        
        # Draw pause indicator if paused
        if self.paused:
            pause_text = font.render("PAUSED", True, RED)
            self.screen.blit(pause_text, (WIDTH - 100, 10))
        
        # Update display
        pygame.display.flip()
        
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_h:
                    self.show_heatmap = not self.show_heatmap
                elif event.key == pygame.K_r:
                    # Reset panic
                    self.panic_active = False
                    for agent in self.agents:
                        agent.panic_level = 0.0
                        agent.panicked = False
                elif event.key == pygame.K_s:
                    # Save results
                    self.save_results()
                elif event.key == pygame.K_v:
                    # Toggle video recording
                    if not self.recording:
                        self.start_recording()
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Inject panic at mouse position
                    x, y = event.pos
                    self.inject_panic(x, y, radius=50)
                    
                    # Auto-start recording if panic is injected and not already recording
                    if not self.recording:
                        self.start_recording()
        
    def save_results(self, output_dir="panic_sim_results"):
        """Save simulation results to the specified directory"""
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save screenshot
        screenshot_path = os.path.join(output_dir, f"panic_sim_{timestamp}.png")
        pygame.image.save(self.screen, screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        
        # Save agent data
        data = {
            "simulation_time": self.simulation_time,
            "agent_count": len(self.agents),
            "panic_source": self.panic_source,
            "panic_active": self.panic_active,
            "agents": []
        }
        
        for agent in self.agents:
            agent_data = {
                "id": agent.id,
                "position": agent.position.tolist(),
                "velocity": agent.velocity.tolist(),
                "panic_level": agent.panic_level,
                "panicked": agent.panicked,
                "is_density_point": agent.is_density_point
            }
            data["agents"].append(agent_data)
        
        # Save to JSON
        data_path = os.path.join(output_dir, f"panic_sim_data_{timestamp}.json")
        with open(data_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Simulation data saved to {data_path}")
        
        # Ask to record animation if not already recording
        if not self.recording:
            print("Would you like to record an animation? Press 'v' to start recording")

# Main function to run the simulation
def main():
    """Main function for running the panic simulation"""
    import sys
    
    # Initialize Pygame
    pygame.init()
    
    # Create simulation
    simulation = PanicSimulation()
    
    if len(sys.argv) > 1:
        # Load data from specified video
        video_path = sys.argv[1]
        if simulation.load_data_from_video(video_path):
            # Run simulation
            simulation.run_simulation()
        else:
            print(f"Failed to load data from {video_path}")
    else:
        # Check the available videos and load the first one
        video_files = []
        video_dir = "."
        
        # Look for MP4 files
        for file in os.listdir(video_dir):
            if file.endswith(".mp4"):
                video_files.append(os.path.join(video_dir, file))
        
        if video_files:
            video_path = video_files[0]
            print(f"Using video: {video_path}")
            if simulation.load_data_from_video(video_path):
                # Run simulation
                simulation.run_simulation()
            else:
                print(f"Failed to load data from {video_path}")
        else:
            print("No video files found. Please specify a video path.")

if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) is required for this simulation.")
        print("Install it with: pip install opencv-python")
        sys.exit(1)
        
    main() 