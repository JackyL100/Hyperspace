from envs.Hyperspace.bad_guy import BadGuy
from envs.Hyperspace.player import Player
import numpy as np

class HyperSpace:
    def __init__(self,
                size,
                locations,
                num_bad_guys = 2):
        assert len(locations) == num_bad_guys
        self.bad_guys = [BadGuy(locations[i]) for i in range(len(locations))]
        self.width, self.height = size
        self.size = size
        self.starting_x = self.width / 2
        self.starting_y = self.height / 2
        self.player = Player(self.starting_x, self.starting_y)

        self.canvas = np.zeros(size)

    def restart(self):
        for bad_guy in self.bad_guys:
            bad_guy.restart()
        self.player.moveTo(self.starting_x, self.starting_y)

    def move(self, x, y):
        self.player.moveBy(x, y)
        for bad_guy in self.bad_guys:
            bad_guy.move_towards((self.player.x, self.player.y))

    def observe(self):
        observation = [self.player.x, self.player.y]
        for bad_guy in self.bad_guys:
            observation.append(bad_guy.x)
            observation.append(bad_guy.y)
        obs = np.array(observation, dtype=np.float64)
        return obs

    def draw_game(self):
        self.canvas = np.zeros(self.size)
        self.canvas[int(self.player.draw_x) : int(self.player.draw_x + self.player.width), int(self.player.draw_y) : int(self.player.draw_y + self.player.height)] = 1
        del_color = 1.0 / (float(len(self.bad_guys)) + 1.0)
        color = del_color
        for bad_guy in self.bad_guys:
            self.canvas[int(bad_guy.draw_x) : int(bad_guy.draw_x + bad_guy.width), int(bad_guy.draw_y) : int(bad_guy.draw_y + bad_guy.height)] = color
            color += del_color

    def is_over(self):
        is_terminal = False
        is_truncated = False
        for bad_guy in self.bad_guys:
            if self.player.x + self.player.width >= bad_guy.x and \
            self.player.x <= bad_guy.x + bad_guy.width and\
            self.player.y + self.player.height >= bad_guy.y\
            and self.player.y <= bad_guy.y + bad_guy.height:
                is_truncated = True
                is_terminal = True
        if self.player.x > self.width or self.player.y > self.height or self.player.y < 0:
            is_terminal = True
            is_truncated = True
        if self.player.x <= 0:
            is_terminal = True
            is_truncated = False
            
        return is_terminal, is_truncated