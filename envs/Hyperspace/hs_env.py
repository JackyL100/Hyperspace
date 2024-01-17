import gym
import numpy as np
import matplotlib.pyplot as plt
from envs.Hyperspace.hs_game import HyperSpace
import cv2
import math

class HyperSpaceEnv(gym.Env):
    def __init__(self,
                size,
                locations,
                num_bad_guys = 2):
        assert len(locations) == num_bad_guys
        self.seed = 42
        self.width,self.height = size
        upper = max(self.width, self.height)
        self.game = HyperSpace(size=size,
                               locations=locations,
                               num_bad_guys=num_bad_guys)
        self.observation_shape = (( num_bad_guys + 1 ) * 2)
        self.observation_space = gym.spaces.Box(low=np.zeros(self.observation_shape),
                                                high=np.array([upper for i in range(self.observation_shape)]),
                                                dtype=np.float64)
        self.action_space = gym.spaces.Tuple((gym.spaces.Box(-1,1,shape=(1,)), gym.spaces.Box(-1,1,shape=(1,))))

    def reset(self):
        self.game.restart()
        return self.game.observe(), {}

    def step(self, action): # agent move certain x and y amount
        x, y = action
        mag = math.sqrt(x**2 + y**2)
        x = x / mag
        y = y / mag
        self.game.move(x,y) # environment moves agent position by x and y
        is_terminal, is_truncated = self.game.is_over()
        reward = ((self.height - self.game.player.x) / self.height) ** 2
        if is_terminal and not is_truncated:
            reward = 100
        elif is_terminal and is_truncated:
            reward = -50
        return self.game.observe(), reward, is_terminal, is_truncated, {} # return new game state
        
    def render(self, mode="human", close=False):
        self.game.draw_game()
        cv2.imshow("game",self.game.canvas)