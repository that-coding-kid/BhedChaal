import mesa
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import json
import os
from mesa_viz_tornado.ModularVisualization import ModularServer, VisualizationElement
from mesa_viz_tornado.modules import CanvasGrid, ChartModule
from mesa_viz_tornado.UserParam import Slider
from mesa.space import ContinuousSpace
from mesa.agent import AgentSet
from mesa.datacollection import DataCollector


##############################################################################
# Utility Functions
##############################################################################

def get_polygon_bounds(polygon):
    """Get the bounding box of a polygon"""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def get_polygon_center(polygon):
    """Calculate the center point of a polygon"""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calculate_polygon_force(position, polygon, k1=1.2e5, k2=2.4e5):
    """Calculate repulsive force from polygon boundary"""
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
        to_agent = np.array(position) - p1
        
        # Project to_agent onto edge
        projection = np.dot(to_agent, edge_direction)
        projection = np.clip(projection, 0, edge_length)
        
        # Calculate closest point on edge
        closest = p1 + projection * edge_direction
        
        # Calculate distance to closest point
        dist = np.linalg.norm(np.array(position) - closest)
        
        if dist < min_distance:
            min_distance = dist
            closest_point = closest
    
    if min_distance < k2 and closest_point is not None:
        # Calculate force direction (away from edge)
        direction = np.array(position) - closest_point
        direction_norm = np.linalg.norm(direction)
        
        # Only apply force if not exactly at the closest point
        if direction_norm > 0.0001:
            direction = direction / direction_norm
            
            # Calculate force magnitude
            force_magnitude = k1 * np.exp(-min_distance/k2)
            return force_magnitude * direction
    
    return np.array([0.0, 0.0])

def generate_points_in_polygon(polygon, num_points):
    """Generate random points within a polygon"""
    print(f"Attempting to generate {num_points} within polygon: {polygon}")
    points = []
    min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
    print(f"Polygon bounds: {min_x}, {min_y}, {max_x}, {max_y}")
    
    attempts = 0
    max_attempts = num_points * 100  # Avoid infinite loop
    
    while len(points) < num_points and attempts < max_attempts:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        if is_point_in_polygon((x, y), polygon):
            points.append((x, y))
        attempts += 1
    
    print(f"Generated {len(points)} points after {attempts} attempts")
    return points


##############################################################################
# Mesa Agent Class 
##############################################################################

class CrowdAgent(mesa.Agent):
    """An agent in the crowd simulation."""
    
    def __init__(self, unique_id, model, pos, walking_areas, 
                 is_density_group=False, init_direction=None, 
                 group_affiliation=None):
        # Updated initialization for Mesa 3.0 compatibility
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos
        self._position = pos  # Internal position for Mesa 3.0 compatibility
        self.walking_areas = walking_areas
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.is_density_group = is_density_group
        
        # Physical properties
        self.radius = 5  # Increased size of agent
        self.mass = 1.0
        
        # Movement parameters  
        self.desired_speed = 25.0
        if init_direction is None:
            angle = random.uniform(0, 2 * math.pi)
            self.desired_direction = np.array([math.cos(angle), math.sin(angle)])
        else:
            self.desired_direction = np.array(init_direction)
        
        self.relaxation_time = 0.05
        
        # Panic-related attributes
        self.panic_level = 0.0  # 0 to 1
        self.panic_contagion_rate = 0.15
        self.panic_decay_rate = 0.05
        self.panic_speed_factor = 5.5
        self.panic_force_factor = 2.0
        self.panic_reaction_factor = 0.7
        self.panic_awareness_radius = 15
        
        # Surge attributes
        self.surge_force = 0.0
        self.surge_direction = np.array([0.0, 0.0])
        self.surge_decay = 0.98
        self.surge_wave_radius = 0
        self.surge_wave_speed = 10
        self.surge_origin = np.array([0.0, 0.0])
        
        # State tracking
        self.panicked = False
        self.awareness = 10
        self.friction = 0.85
        
        # Social properties
        self.group_affiliation = group_affiliation
        self.familiarity = {}
        
        # Force model parameters
        self.A = 2000.0  # Repulsion strength
        self.B = 0.15    # Repulsion range
        self.C = 500.0   # Attraction strength
        self.D = 0.1     # Attraction range
        self.k1 = 1.2e5  # Wall repulsion strength
        self.k2 = 2.4e5  # Wall repulsion range
        
    @property
    def position(self):
        """Property for Mesa 3.0 visualization compatibility"""
        return self.pos
    
    @position.setter
    def position(self, pos):
        """Property setter for Mesa 3.0 visualization compatibility"""
        self.pos = pos
    
    def update_panic_level(self):
        """Update panic level based on proximity to other panicked agents"""
        if self.panic_level > 0:
            # Natural decay of panic
            self.panic_level = max(0, self.panic_level - self.panic_decay_rate)
            
            # Check for panic propagation from nearby agents
            for agent in self.model.crowd_agents:
                if agent.unique_id == self.unique_id:
                    continue
                    
                dist = distance(self.pos, agent.pos)
                if dist < self.panic_awareness_radius:
                    # Calculate weight based on distance and familiarity
                    distance_weight = 1 - (dist / self.panic_awareness_radius)
                    familiarity_weight = self.familiarity.get(agent.unique_id, 0.5)
                    weight = distance_weight * (1 + familiarity_weight)
                    
                    # Update panic level 
                    panic_increase = (self.panic_contagion_rate * 
                                   weight * 
                                   agent.panic_level * 
                                   (1 - self.panic_level))
                    self.panic_level = min(1.0, self.panic_level + panic_increase)
                    
                    # Update panic state
                    self.panicked = self.panic_level > 0.7
    
    def calculate_social_forces(self):
        """Calculate forces from other agents"""
        net_force = np.array([0.0, 0.0])
        
        if not self.panicked:
            # Normal movement behavior
            random_angle = random.uniform(-0.2, 0.2)  
            rotation_matrix = np.array([
                [np.cos(random_angle), -np.sin(random_angle)],
                [np.sin(random_angle), np.cos(random_angle)]
            ])
            current_direction = rotation_matrix @ self.desired_direction
            
            # Strong directional force with increased strength
            directional_force = current_direction * self.desired_speed * 20.0
            net_force += directional_force
            
            # Repulsive forces from other agents
            interaction_radius = 20  # Only interact with agents within this radius
            
            for agent in self.model.crowd_agents:
                if agent.unique_id == self.unique_id:
                    continue
                
                dist = distance(self.pos, agent.pos)
                if dist < interaction_radius:
                    # Calculate repulsion
                    dx = self.pos[0] - agent.pos[0]
                    dy = self.pos[1] - agent.pos[1]
                    
                    # Avoid division by zero
                    if dist < 0.0001:
                        direction = np.array([1.0, 0.0])
                    else:
                        direction = np.array([dx, dy]) / dist
                        
                    effective_distance = dist - (self.radius + agent.radius)
                    
                    if effective_distance < 0:
                        # Direct collision - stronger repulsion when panicked
                        panic_factor = 1 + self.panic_force_factor * (self.panic_level + agent.panic_level) / 2
                        net_force += 1e3 * direction * panic_factor
                    elif not (self.panicked and agent.panicked):
                        # Normal repulsion (skip if both are panicked to allow clustering)
                        force_magnitude = self.A * np.exp(-self.B * max(0.01, effective_distance))
                        net_force += force_magnitude * direction * 0.3  # Reduced weight
                        
        else:
            # Panic behavior - flee from panic center
            from_panic_center = np.array(self.pos) - self.surge_origin
            distance_from_panic = np.linalg.norm(from_panic_center)
            
            if distance_from_panic > 0.0001:
                flee_direction = from_panic_center / distance_from_panic
                flee_force = flee_direction * self.desired_speed * 30.0 * self.panic_level
                net_force += flee_force
            
            # Add surge wave effect
            if self.surge_wave_radius > 0:
                distance_from_center = np.linalg.norm(np.array(self.pos) - self.surge_origin)
                wave_effect = max(0, 1 - abs(distance_from_center - self.surge_wave_radius) / 10)
                
                if wave_effect > 0:
                    surge_direction = np.array(self.pos) - self.surge_origin
                    surge_norm = np.linalg.norm(surge_direction)
                    if surge_norm > 0.0001:
                        surge_direction = surge_direction / surge_norm
                        net_force += surge_direction * wave_effect * 30.0
            
            # Update surge wave radius
            self.surge_wave_radius += self.surge_wave_speed * 0.15
            
            # Repulsive forces from all other agents (flee from everyone)
            interaction_radius = 25
            
            for agent in self.model.crowd_agents:
                if agent.unique_id == self.unique_id:
                    continue
                
                dist = distance(self.pos, agent.pos)
                if dist < interaction_radius:
                    # Calculate repulsion
                    dx = self.pos[0] - agent.pos[0]
                    dy = self.pos[1] - agent.pos[1]
                    
                    # Avoid division by zero
                    if dist < 0.0001:
                        direction = np.array([1.0, 0.0])
                    else:
                        direction = np.array([dx, dy]) / dist
                    
                    # Strength increases as distance decreases
                    strength = 20.0 * (1 - dist/interaction_radius) * self.panic_level
                    net_force += direction * strength
                    
        return net_force
                
    def calculate_boundary_forces(self):
        """Calculate forces from boundaries"""
        net_force = np.array([0.0, 0.0])
        position = np.array(self.pos)
        
        # Calculate forces from walking areas
        for area in self.walking_areas:
            force = calculate_polygon_force(position, area)
            net_force += force
        
        # Avoid top and bottom boundaries
        boundary_threshold = 20
        if position[1] < boundary_threshold:
            # Repel downward
            net_force += np.array([0.0, self.k1 * 0.1])
        elif position[1] > self.model.height - boundary_threshold:
            # Repel upward
            net_force += np.array([0.0, -self.k1 * 0.1])
            
        return net_force
    
    def handle_out_of_bounds(self):
        """Handle case when agent is outside all walking areas"""
        position = np.array(self.pos)
        
        # Check if agent is in any walking area
        in_any_area = any(is_point_in_polygon(position, area) for area in self.walking_areas)
        
        if not in_any_area and self.walking_areas:
            # Find closest point on any walking area
            closest_point = None
            min_distance = float('inf')
            closest_normal = None
            
            for area in self.walking_areas:
                for i in range(len(area)):
                    p1 = np.array(area[i])
                    p2 = np.array(area[(i + 1) % len(area)])
                    
                    v = p2 - p1
                    w = position - p1
                    
                    c1 = np.dot(w, v)
                    if c1 <= 0:
                        # Before p1 - use distance to p1
                        dist = np.linalg.norm(position - p1)
                        closest = p1.copy()
                    else:
                        c2 = np.dot(v, v)
                        if c2 <= c1:
                            # After p2 - use distance to p2
                            dist = np.linalg.norm(position - p2)
                            closest = p2.copy()
                        else:
                            # Between p1 and p2 - use perpendicular distance
                            b = c1 / c2
                            closest = p1 + b * v
                            dist = np.linalg.norm(position - closest)
                    
                    if dist < min_distance:
                        min_distance = dist
                        closest_point = closest
                        
                        # Calculate normal - perpendicular to the edge
                        edge_norm = np.linalg.norm(v)
                        if edge_norm < 0.0001:
                            edge_direction = np.array([1.0, 0.0])
                        else:
                            edge_direction = v / edge_norm
                        
                        normal = np.array([-edge_direction[1], edge_direction[0]])
                        
                        # Test if this normal points into the area
                        test_point = closest + normal * 2
                        if not is_point_in_polygon(test_point, area):
                            normal = -normal
                            
                        closest_normal = normal
            
            # If we found a closest point, move there and bounce
            if closest_point is not None and closest_normal is not None:
                # Move slightly inside
                self.pos = tuple(closest_point + closest_normal * 2)
                
                # Bounce
                dot_product = np.dot(self.velocity, closest_normal)
                if dot_product < 0:  # Only bounce if moving toward boundary
                    self.velocity = self.velocity - 2 * dot_product * closest_normal
                    
                    # Add randomness when panicked
                    if self.panicked:
                        self.velocity += np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
    
    def step(self):
        """Update agent state for one step"""
        # Update panic level
        self.update_panic_level()
        
        # Calculate net force
        social_force = self.calculate_social_forces()
        boundary_force = self.calculate_boundary_forces()
        
        net_force = social_force + boundary_force
        
        # Update acceleration
        self.acceleration = net_force
        
        # Physics time step
        physics_dt = 0.05
        
        # Update velocity
        self.velocity += self.acceleration * physics_dt
        self.velocity *= self.friction
        
        # Limit maximum velocity
        max_speed = 15.0 if self.panicked else 10.0
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > max_speed:
            self.velocity = self.velocity / current_speed * max_speed
            
        # Ensure minimum speed to prevent stuck agents
        min_speed = 3.0 if self.panicked else 2.0
        if 0 < current_speed < min_speed:
            self.velocity = self.velocity / current_speed * min_speed
        
        # Update position
        new_x = self.pos[0] + self.velocity[0] * physics_dt
        new_y = self.pos[1] + self.velocity[1] * physics_dt
        
        # Ensure we're not at NaN positions
        if np.isnan(new_x) or np.isnan(new_y):
            new_x, new_y = self.model.width/2, self.model.height/2
            self.velocity = np.array([0.0, 0.0])
            print(f"Warning: Agent {self.unique_id} had NaN position. Reset to center.")
        
        self.pos = (new_x, new_y)
        
        # Handle out of bounds
        self.handle_out_of_bounds()


##############################################################################
# Mesa Model Class
##############################################################################

class CrowdSimulation(mesa.Model):
    """Model for crowd simulation."""
    
    def __init__(self, width=800, height=600, initial_agents=100, 
                 json_file='video_data/areas/areas_e48cdf5c9f4567cc315f28760929e2ac.json'):
        super().__init__()
        self.width = width
        self.height = height
        self.initial_agents = initial_agents
        self.panic_injected = False
        self.json_file = json_file
        self.running = True
        
        # Random number generator with a seed for reproducibility
        self.random = random.Random(12345)
        
        # Create agent set instead of scheduler (pass empty list initially)
        self.crowd_agents = AgentSet([], random=self.random)
        
        # Create space
        self.space = ContinuousSpace(width, height, True)
        
        # Load environment data (walking areas, roads)
        self.walking_areas, self.roads = self.load_environment()
        
        # Create agents
        self.create_agents()
        
        # Initialize data collector
        self.datacollector = DataCollector(
            model_reporters={"Panic Count": lambda m: sum(1 for a in m.crowd_agents if a.panicked)},
            agent_reporters={}
        )
    
    def load_environment(self):
        """Load walking areas and roads from JSON file"""
        walking_areas = []
        roads = []
        
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            print(f"Loaded environment data: {data}")
            
            # Calculate scaling factors to fit the areas to our screen dimensions
            original_width = 1280  # Approximate width from the coordinates
            original_height = 720  # Approximate height from the coordinates
            
            # Calculate scaling factors
            width_scale = self.width / original_width
            height_scale = self.height / original_height
            
            print(f"Scaling factors: width={width_scale}, height={height_scale}")
            
            if 'walking_areas' in data:
                for area in data['walking_areas']:
                    # Scale the polygon points
                    scaled_area = []
                    for point in area:
                        scaled_x = point[0] * width_scale
                        scaled_y = point[1] * height_scale
                        scaled_area.append([scaled_x, scaled_y])
                    walking_areas.append(scaled_area)
                    print(f"Added walking area: {scaled_area}")
            
            if 'roads' in data:
                for road in data['roads']:
                    # Scale the polygon points
                    scaled_road = []
                    for point in road:
                        scaled_x = point[0] * width_scale
                        scaled_y = point[1] * height_scale
                        scaled_road.append([scaled_x, scaled_y])
                    roads.append(scaled_road)
                    
        except Exception as e:
            print(f"Error loading environment from {self.json_file}: {e}")
            # Create some default areas if loading fails
            walking_areas = [
                [[100, 100], [700, 100], [700, 500], [100, 500]]
            ]
            roads = []
            print(f"Using default walking area: {walking_areas[0]}")
            
        return walking_areas, roads
    
    def create_agents(self):
        """Create agents within walking areas"""
        agent_count = 0
        
        # Print walking areas for debugging
        print(f"Creating agents in {len(self.walking_areas)} walking areas")
        for i, area in enumerate(self.walking_areas):
            print(f"Walking area {i}: {area}")
        
        # Create agents in walking areas
        for area in self.walking_areas:
            # Generate random points within the polygon
            num_agents = max(2, int(self.initial_agents / len(self.walking_areas)))
            print(f"Generating {num_agents} agents in area")
            points = generate_points_in_polygon(area, num_agents)
            print(f"Generated {len(points)} points in polygon")
            
            for x, y in points:
                agent = CrowdAgent(agent_count, self, (x, y), self.walking_areas)
                self.crowd_agents.add(agent)
                # Initialize familiarity with other agents
                for a in self.crowd_agents:
                    if a.unique_id != agent_count:
                        agent.familiarity[a.unique_id] = random.random() * 0.5
                        a.familiarity[agent_count] = random.random() * 0.5
                
                # Assign random group affiliations
                if random.random() < 0.3:  # 30% chance of being in a group
                    agent.group_affiliation = random.randint(0, 3)
                
                agent_count += 1
                print(f"Added agent {agent_count} at position {x}, {y}")
            
        # If we still need more agents, place them at the bottom
        if agent_count < self.initial_agents:
            bottom_y_min = self.height - 150
            bottom_y_max = self.height - 50
            remaining = self.initial_agents - agent_count
            
            print(f"Adding {remaining} more agents at the bottom")
            
            for _ in range(remaining):
                # Random position at bottom
                x = random.uniform(50, self.width - 50)
                y = random.uniform(bottom_y_min, bottom_y_max)
                
                # Check if in any walking area
                pos_valid = False
                for area in self.walking_areas:
                    if is_point_in_polygon((x, y), area):
                        pos_valid = True
                        break
                
                if pos_valid or not self.walking_areas:  # Add anyway if no walking areas
                    agent = CrowdAgent(agent_count, self, (x, y), self.walking_areas)
                    self.crowd_agents.add(agent)
                    agent_count += 1
                    print(f"Added agent {agent_count} at position {x}, {y}")
        
        print(f"Created {agent_count} agents")
                
    def inject_panic(self, center_x=None, center_y=None, radius=50, method='spatial'):
        """Inject panic into agents"""
        if method == 'spatial':
            # If no center provided, choose a random position in a walking area
            if center_x is None or center_y is None:
                if self.walking_areas:
                    # Choose a random walking area
                    area = random.choice(self.walking_areas)
                    min_x, min_y, max_x, max_y = get_polygon_bounds(area)
                    
                    # Find valid position
                    for _ in range(20):
                        center_x = random.uniform(min_x, max_x)
                        center_y = random.uniform(min_y, max_y)
                        if is_point_in_polygon((center_x, center_y), area):
                            break
                    else:
                        # Use center as fallback
                        center_x, center_y = get_polygon_center(area)
                else:
                    # No walking areas, use center of space
                    center_x, center_y = self.width/2, self.height/2
            
            panic_center = np.array([center_x, center_y])
            
            # Affect agents within radius
            for agent in self.crowd_agents:
                dist = np.linalg.norm(np.array(agent.pos) - panic_center)
                if dist < radius:
                    # Panic level decreases with distance from center
                    panic_factor = 1 - (dist / radius)
                    agent.panic_level = min(1.0, agent.panic_level + 0.8 * panic_factor)
                    agent.panicked = agent.panic_level > 0.7
                    agent.surge_origin = panic_center
                    agent.surge_wave_radius = 0
        
        elif method == 'random':
            # Affect random percentage of agents
            num_agents = int(len(self.crowd_agents) * 0.2)  # 20% of agents
            if num_agents > 0:
                selected_agents = random.sample(list(self.crowd_agents), num_agents)
                
                # Choose a random panic origin in walking areas
                if self.walking_areas:
                    area = random.choice(self.walking_areas)
                    center_x, center_y = get_polygon_center(area)
                else:
                    center_x, center_y = self.width/2, self.height/2
                
                panic_center = np.array([center_x, center_y])
                
                for agent in selected_agents:
                    agent.panic_level = 1.0
                    agent.panicked = True
                    agent.surge_origin = panic_center
                    agent.surge_wave_radius = 0
    
    def step(self):
        """Advance the model by one step"""
        for agent in self.crowd_agents:
            agent.step()
            
        self.datacollector.collect(self)
        
        # Allow panic to propagate (optionally through agent interactions)
        for agent in self.crowd_agents:
            if agent.panicked:
                for other in self.crowd_agents:
                    if other != agent:
                        dist = distance(agent.pos, other.pos)
                        if dist < agent.awareness:
                            # Panic propagation
                            familiarity = agent.familiarity.get(other.unique_id, 0.0)
                            panic_transfer = 0.1 * (1 + familiarity)
                            other.panic_level = min(1.0, other.panic_level + panic_transfer)
                            if other.panic_level > 0.7:
                                other.panicked = True


##############################################################################
# UI Elements Setup
##############################################################################

def agent_portrayal(agent):
    """Define how agents look in visualization"""
    portrayal = {
        "Shape": "circle",
        "r": 3,
        "Filled": "true",
        "Color": "red" if agent.panicked else "blue",
        "Layer": 1
    }
    return portrayal

# Create parameter sliders for the model
model_params = {
    "width": 800,
    "height": 600,
    "initial_agents": Slider("Number of Agents", value=100, min_value=10, max_value=500, step=10),
    "json_file": 'video_data/areas/areas_e48cdf5c9f4567cc315f28760929e2ac.json'
}

# Create a chart to show panic levels
chart = ChartModule([
    {"Label": "Panic Count", "Color": "red"}
], data_collector_name='datacollector')

# Create JavaScript file for the continuous space visualization
with open("continuous_canvas.js", "w") as f:
    f.write("""
var FixedCanvasVisualization = function() {
    const width = 800;
    const height = 600;
    
    // Create a canvas element dynamically
    var canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.style.border = "1px solid black";
    
    // Add event listeners for keyboard controls
    document.addEventListener('keydown', function(event) {
        console.log("Key pressed:", event.key);
        
        if (event.key === 'p' || event.key === 'P') {
            console.log("Triggering panic!");
            // Create a custom message
            var message = {
                "type": "request_step",
                "step": 1,
                "trigger_panic": true
            };
            // Send it through the websocket
            ws.send(JSON.stringify(message));
        }
        
        if (event.key === 'r' || event.key === 'R') {
            console.log("Triggering random panic!");
            // Create a custom message
            var message = {
                "type": "request_step",
                "step": 1,
                "trigger_random_panic": true
            };
            // Send it through the websocket
            ws.send(JSON.stringify(message));
        }
    });
    
    // Create panic buttons
    function createPanicButtons() {
        if (document.getElementById("panic-button")) return;
        
        var panicBtn = document.createElement("button");
        panicBtn.id = "panic-button";
        panicBtn.innerHTML = "TRIGGER PANIC";
        panicBtn.style.position = "fixed";
        panicBtn.style.top = "100px";
        panicBtn.style.right = "20px";
        panicBtn.style.padding = "10px";
        panicBtn.style.backgroundColor = "red";
        panicBtn.style.color = "white";
        panicBtn.style.border = "none";
        panicBtn.style.fontWeight = "bold";
        panicBtn.style.cursor = "pointer";
        panicBtn.style.zIndex = "1000";
        
        panicBtn.onclick = function() {
            console.log("Panic button clicked");
            var message = {
                "type": "request_step",
                "step": 1,
                "trigger_panic": true
            };
            ws.send(JSON.stringify(message));
        };
        
        document.body.appendChild(panicBtn);
    }
    
    this.render = function(data) {
        // Make sure canvas is in the DOM
        if (!canvas.parentNode) {
            var elements = document.getElementById("elements");
            if (elements) {
                elements.appendChild(canvas);
            } else {
                document.body.appendChild(canvas);
            }
            // Create buttons when canvas is added
            createPanicButtons();
        }
        
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, width, height);
        
        // First, draw all the polygon elements (areas and roads)
        for (var i = 0; i < data.length; i++) {
            var d = data[i];
            if (d.Shape === "polygon") {
                var points = d.Points;
                if (points && points.length > 0) {
                    ctx.beginPath();
                    ctx.moveTo(points[0][0], points[0][1]);
                    
                    for (var j = 1; j < points.length; j++) {
                        ctx.lineTo(points[j][0], points[j][1]);
                    }
                    
                    ctx.closePath();
                    ctx.strokeStyle = d.Color;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    if (d.Filled === "true") {
                        ctx.fillStyle = d.Color;
                        ctx.fill();
                    }
                }
            }
        }
        
        // Draw all agents as static circles in a grid for debugging
        var agentCount = 0;
        for (var i = 0; i < data.length; i++) {
            var d = data[i];
            if (d.Shape === "circle") {
                // Determine grid position
                var row = Math.floor(agentCount / 20);
                var col = agentCount % 20;
                
                // Calculate position in a grid layout
                var x = 50 + col * 35;
                var y = 50 + row * 35;
                
                // Draw the agent at the grid position
                ctx.beginPath();
                ctx.arc(x, y, 15, 0, Math.PI * 2);
                ctx.fillStyle = d.Color;
                ctx.fill();
                ctx.strokeStyle = "#000000";
                ctx.lineWidth = 1;
                ctx.stroke();
                
                agentCount++;
            }
        }
        
        // Add text to canvas showing agent count
        ctx.fillStyle = "#000000";
        ctx.font = "20px Arial";
        ctx.fillText("Agents: " + agentCount, 10, 30);
        
        // Add instructions
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(10, height - 80, 380, 70);
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "14px Arial";
        ctx.fillText("Press 'P' to trigger panic at a random location", 20, height - 55);
        ctx.fillText("Press 'R' to trigger random panic across agents", 20, height - 35);
        ctx.fillText("Blue = Normal agents, Red = Panicked agents", 20, height - 15);
    };
    
    this.reset = function() {
        // Nothing needed
    };
};
    """)

##############################################################################
# Custom Visualization
##############################################################################

class CanvasVisualization(VisualizationElement):
    """A custom visualization element using a basic HTML5 canvas."""
    
    local_includes = ["canvas_vis.js"]
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Create a simple JavaScript file that creates a canvas and draws agents in a grid
        with open("canvas_vis.js", "w") as f:
            f.write("""
var CanvasVisualization = function() {
    // Canvas dimensions
    var width = 800;
    var height = 600;
    var canvas = null;
    var ctx = null;
    
    // Create the canvas when first rendering
    function createCanvas() {
        canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        canvas.style.border = "1px solid black";
        canvas.style.backgroundColor = "#f0f0f0";
        
        // Find where to append the canvas
        var elementsDiv = document.getElementById("elements");
        if (elementsDiv) {
            elementsDiv.appendChild(canvas);
        } else {
            document.body.appendChild(canvas);
        }
        
        ctx = canvas.getContext("2d");
        
        // Create control panel with panic button
        var controlPanel = document.createElement("div");
        controlPanel.style.position = "fixed";
        controlPanel.style.top = "80px";
        controlPanel.style.right = "10px";
        controlPanel.style.padding = "10px";
        controlPanel.style.backgroundColor = "rgba(255,255,255,0.8)";
        controlPanel.style.border = "1px solid #ccc";
        controlPanel.style.borderRadius = "5px";
        controlPanel.style.zIndex = "1000";
        
        // Add panic button
        var panicButton = document.createElement("button");
        panicButton.innerText = "TRIGGER PANIC";
        panicButton.style.backgroundColor = "red";
        panicButton.style.color = "white";
        panicButton.style.padding = "8px 15px";
        panicButton.style.border = "none";
        panicButton.style.borderRadius = "5px";
        panicButton.style.cursor = "pointer";
        panicButton.style.display = "block";
        panicButton.style.marginBottom = "10px";
        
        panicButton.onclick = function() {
            // Request a step and trigger panic
            // This has no direct effect but demonstrates the UI
            console.log("Panic button clicked");
            alert("Panic triggered! This would cause panic in the simulation.");
        };
        
        // Add random panic button
        var randomButton = document.createElement("button");
        randomButton.innerText = "RANDOM PANIC";
        randomButton.style.backgroundColor = "purple";
        randomButton.style.color = "white";
        randomButton.style.padding = "8px 15px";
        randomButton.style.border = "none";
        randomButton.style.borderRadius = "5px";
        randomButton.style.cursor = "pointer";
        randomButton.style.display = "block";
        
        randomButton.onclick = function() {
            console.log("Random panic button clicked");
            alert("Random panic triggered! This would cause random panic in the simulation.");
        };
        
        // Add instruction text
        var instructionsDiv = document.createElement("div");
        instructionsDiv.style.marginTop = "15px";
        instructionsDiv.style.fontSize = "12px";
        instructionsDiv.innerHTML = "<strong>Press P key:</strong> Trigger panic<br>" +
                                  "<strong>Press R key:</strong> Random panic<br>" +
                                  "<strong>Blue:</strong> Normal agents<br>" +
                                  "<strong>Red:</strong> Panicked agents";
        
        // Add everything to control panel
        controlPanel.appendChild(panicButton);
        controlPanel.appendChild(randomButton);
        controlPanel.appendChild(instructionsDiv);
        document.body.appendChild(controlPanel);
        
        // Add keyboard event listeners
        document.addEventListener("keydown", function(event) {
            if (event.key === "p" || event.key === "P") {
                console.log("P key pressed - would trigger panic");
                alert("P key pressed - would trigger panic");
            } else if (event.key === "r" || event.key === "R") {
                console.log("R key pressed - would trigger random panic");
                alert("R key pressed - would trigger random panic");
            }
        });
    }
    
    this.render = function(agents) {
        // Create canvas if it doesn't exist yet
        if (!canvas) {
            createCanvas();
        }
        
        // Clear the canvas
        ctx.clearRect(0, 0, width, height);
        
        // First, draw any polygons (walking areas)
        var polygons = agents.filter(function(a) { return a.Shape === "polygon"; });
        for (var i = 0; i < polygons.length; i++) {
            var poly = polygons[i];
            var points = poly.Points;
            
            if (points && points.length > 0) {
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                
                for (var j = 1; j < points.length; j++) {
                    ctx.lineTo(points[j][0], points[j][1]);
                }
                
                ctx.closePath();
                ctx.strokeStyle = poly.Color || "#00FF00";
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
        
        // Then draw agents in a grid layout for visibility
        var agentData = agents.filter(function(a) { return a.Shape === "circle"; });
        var totalAgents = agentData.length;
        
        // Layout in a grid with 20 columns
        var cellSize = 35;
        var cols = 20;
        var startX = 40;
        var startY = 40;
        
        for (var i = 0; i < totalAgents; i++) {
            var agent = agentData[i];
            var row = Math.floor(i / cols);
            var col = i % cols;
            
            var x = startX + col * cellSize;
            var y = startY + row * cellSize;
            
            // Draw the agent
            ctx.beginPath();
            ctx.arc(x, y, 15, 0, Math.PI * 2);
            ctx.fillStyle = agent.Color;
            ctx.fill();
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        
        // Show agent count and other info
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(10, 10, 150, 30);
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "14px Arial";
        ctx.fillText("Agents: " + totalAgents, 20, 30);
    };
    
    this.reset = function() {
        // Nothing needed
    };
};
            """)
        
        self.js_code = "elements.push(new CanvasVisualization());"
        
    def render(self, model):
        """Create a portrayal of the agents and environment"""
        space_state = []
        
        # Add walking areas
        for area in model.walking_areas:
            space_state.append({
                "Shape": "polygon",
                "Points": area,
                "Color": "#00FF00",  # Green
                "Filled": "false"
            })
        
        # Add agents
        for agent in model.crowd_agents:
            x, y = agent.pos
            portrayal = {
                "Shape": "circle",
                "x": x,
                "y": y,
                "Color": "#FF0000" if agent.panicked else "#0000FF",  # Red or Blue
                "r": 15,
                "Filled": "true"
            }
            space_state.append(portrayal)
        
        return space_state

def run_server():
    """Launch the Mesa server with the model"""
    # Use our new visualization class
    canvas_element = CanvasVisualization(800, 600)
    
    print("Creating server with canvas element")
    server = ModularServer(
        CrowdSimulation,
        [canvas_element, chart],
        "Crowd Simulation",
        model_params
    )
    
    server.launch()

if __name__ == "__main__":
    print("Starting Mesa Crowd Simulation...")
    run_server() 