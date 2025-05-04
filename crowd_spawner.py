import random
import numpy as np

class CrowdSpawner:
    """
    Module for spawning crowds at the edges of the simulation.
    People spawned on one edge will walk towards the opposite edge.
    """
    
    def __init__(self, width, height, walking_areas, particle_class, is_point_in_polygon_func):
        self.width = width
        self.height = height
        self.walking_areas = walking_areas
        self.left_edge = 20  # Buffer from the very edge
        self.right_edge = width - 20  # Buffer from the very edge
        self.spawn_cooldown = 0
        self.spawn_interval = 60  # Reduced from 120 - Frames between spawns (1 second at 60 FPS)
        self.Particle = particle_class
        self.is_point_in_polygon = is_point_in_polygon_func
        
    def update(self, particles):
        """Update spawner state and potentially spawn new people"""
        self.spawn_cooldown -= 1
        
        if self.spawn_cooldown <= 0:
            # Reset cooldown with some randomness
            self.spawn_cooldown = self.spawn_interval + random.randint(-20, 20)
            
            # Decide which side to spawn from
            from_left = random.choice([True, False])
            
            # Decide how many people to spawn (increased from 1-25 to 5-35)
            num_to_spawn = random.randint(5, 35)
            
            # Spawn the people
            new_particles = self.spawn_people(from_left, num_to_spawn)
            particles.extend(new_particles)
            
            return new_particles
        
        return []
    
    def spawn_people(self, from_left, count):
        """Spawn a given number of people at a specific point on either the left or right edge"""
        new_particles = []
        
        x_pos = self.left_edge if from_left else self.right_edge
        x_direction = 1.0 if from_left else -1.0  # Direction to walk (opposite of spawn side)
        
        # Choose a single random y position for this group
        base_y_pos = random.randint(50, self.height - 50)
        
        # Try to find a valid position within walking areas
        position_valid = False
        valid_y_pos = None
        
        # Try up to 10 different y positions
        for _ in range(10):
            test_y = base_y_pos + random.randint(-30, 30)  # Add some variation
            test_y = max(20, min(self.height - 20, test_y))  # Keep within screen bounds
            
            for area in self.walking_areas:
                if self.is_point_in_polygon((x_pos, test_y), area):
                    position_valid = True
                    valid_y_pos = test_y
                    break
                    
            if position_valid:
                break
        
        # If no valid position found, return empty list
        if not valid_y_pos:
            return []
        
        # Spawn people around the valid position
        for i in range(count):
            # Add random offset to create a small group rather than a single point
            offset_y = random.randint(-15, 15)
            y_pos = valid_y_pos + offset_y
            
            # Ensure y position is within bounds
            y_pos = max(20, min(self.height - 20, y_pos))
            
            # Double-check that this specific position is valid
            position_valid = False
            for area in self.walking_areas:
                if self.is_point_in_polygon((x_pos, y_pos), area):
                    position_valid = True
                    break
            
            if position_valid:
                # Create new particle
                particle = self.Particle(x_pos, y_pos)
                
                # Set direction toward opposite edge
                particle.desired_direction = np.array([x_direction, 0.0])
                
                # Set ID and other properties
                particle.id = f"edge_spawned_{len(new_particles)}"
                
                # Add to list
                new_particles.append(particle)
        
        return new_particles 