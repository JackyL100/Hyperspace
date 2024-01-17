class Player:
    def __init__(self, 
                 x,
                 y,
                 width = 10,
                 height = 10):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.draw_x = self.x - self.width / 2
        self.draw_y = self.y - self.height / 2

    def moveTo(self, x, y):
        self.x = x
        self.y = y
        self.draw_x = self.x - self.width / 2
        self.draw_y = self.y - self.height / 2

    def moveBy(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        self.draw_x = self.x - self.width / 2
        self.draw_y = self.y - self.height / 2