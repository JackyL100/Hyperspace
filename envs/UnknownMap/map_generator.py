import numpy as np
from envs.UnknownMap.obstacle import Obstacle
class MapGenerator:
    def __init__(self,
                 size,
                 min_obstacles_size = 10,
                 max_obstacle_size = 250,
                 seed = 42):
        self.seed = seed
        self.map_size = size
        self.width, self.height = size
        self.min_obstacles_size = min_obstacles_size
        self.max_obstacle_size = max_obstacle_size
        np.random.seed(self.seed)
        
    def generate(self,
                 size,
                 num_obstacles):
        
        widths = np.random.randint(self.min_obstacles_size, self.max_obstacle_size, size=num_obstacles)
        heights = np.random.randint(self.min_obstacles_size, self.max_obstacle_size, size=num_obstacles)
        x = np.random.randint(0, self.width - widths, size=num_obstacles)
        y = np.random.randint(0, self.height - heights, size=num_obstacles)
        obstacles = [Obstacle((widths[i], heights[i]), (x[i], y[i])) for i in range(num_obstacles)]
        return obstacles
