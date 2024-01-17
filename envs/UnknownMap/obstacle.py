class Obstacle:
    def __init__(self,
                 size,
                 location):
        self.x, self.y = location
        self.width, self.height = size

    def has_collided(self,
                     size,
                     location):
        x2, y2 = location
        width2, height2 = size
        if self.x < x2 + width2 and self.x + self.width > x2 and self.y < y2 + height2 and self.y + self.height > y2:
            return True
        return False