import mesa
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class SimpleAgent(mesa.Agent):
    """A simple agent in the crowd simulation."""
    
    def __init__(self, unique_id, model, pos, panicked=False):
        # In Mesa 3.0, model is the first parameter
        super().__init__(model)
        # Manually assign the unique_id since we're not using Mesa's auto-generation
        self.unique_id = unique_id
        self.pos = pos
        self.panicked = panicked
        self.velocity = np.array([random.uniform(-1,1), random.uniform(-1,1)])
        self.speed = 1.0
        
    def step(self):
        # Simple movement
        if self.panicked:
            self.speed = 2.0
        else:
            self.speed = 1.0
            
        # Add some randomness to direction
        angle = random.uniform(-0.5, 0.5)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        self.velocity = rotation @ self.velocity
        
        # Normalize and apply speed
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > 0:
            self.velocity = self.velocity / velocity_norm * self.speed
            
        # Update position
        new_x = self.pos[0] + self.velocity[0]
        new_y = self.pos[1] + self.velocity[1]
        
        # Bounce off walls
        if new_x < 0 or new_x > self.model.width:
            self.velocity[0] *= -1
            new_x = np.clip(new_x, 0, self.model.width)
            
        if new_y < 0 or new_y > self.model.height:
            self.velocity[1] *= -1
            new_y = np.clip(new_y, 0, self.model.height)
            
        self.pos = (new_x, new_y)
        
        # Panic spreads to nearby agents
        if self.panicked:
            for agent in self.model.schedule.agents:
                if agent != self and not agent.panicked:
                    distance = np.sqrt((self.pos[0] - agent.pos[0])**2 + 
                                       (self.pos[1] - agent.pos[1])**2)
                    if distance < 5.0:  # Panic radius
                        if random.random() < 0.2:  # 20% chance to spread
                            agent.panicked = True


class SimpleCrowdModel(mesa.Model):
    """A simple model for crowd simulation."""
    
    def __init__(self, width=100, height=100, num_agents=50):
        super().__init__()
        self.width = width
        self.height = height
        self.num_agents = num_agents
        
        # Create scheduler
        self.schedule = mesa.time.RandomActivation(self)
        
        # Create space
        self.space = mesa.space.ContinuousSpace(width, height, True)
        
        # Create agents
        for i in range(self.num_agents):
            x = random.random() * self.width
            y = random.random() * self.height
            agent = SimpleAgent(i, self, (x, y))
            self.schedule.add(agent)
            
        # Setup data collector
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={"Panicked": lambda m: sum(1 for a in m.schedule.agents if a.panicked)}
        )
        
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
    def inject_panic(self, x=None, y=None, radius=10):
        """Inject panic at a specific location"""
        if x is None:
            x = self.width / 2
        if y is None:
            y = self.height / 2
            
        # Panic agents near the location
        for agent in self.schedule.agents:
            distance = np.sqrt((agent.pos[0] - x)**2 + (agent.pos[1] - y)**2)
            if distance < radius:
                agent.panicked = True
                
# Run the model
def run_simple_model():
    print("Creating model...")
    model = SimpleCrowdModel(width=100, height=100, num_agents=50)
    
    print("Running steps...")
    # Run for 50 steps without panic
    for i in range(50):
        if i % 10 == 0:
            print(f"Step {i}")
        model.step()
        
    # Inject panic
    print("Injecting panic...")
    model.inject_panic()
    
    # Run for 50 more steps with panic
    for i in range(50, 100):
        if i % 10 == 0:
            print(f"Step {i}")
        model.step()
            
    # Plot results
    print("Generating plot...")
    results = model.datacollector.get_model_vars_dataframe()
    plt.figure(figsize=(10, 6))
    plt.plot(results["Panicked"])
    plt.title("Number of Panicked Agents")
    plt.xlabel("Step")
    plt.ylabel("Panicked Agents")
    plt.savefig("simple_panic.png")
    plt.close()
    
    print("Done! Results saved to simple_panic.png")

if __name__ == "__main__":
    try:
        run_simple_model()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 