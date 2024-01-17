import numpy as np

class BadGuy:
  def __init__(self,
               location,
               size = (40,40),
               speed = 1,
               speed_adjust = 0.2):
    self.x, self.y = location
    self.width, self.height = size
    self.init_x, self.init_y = location

    self.draw_x = self.x - self.width / 2
    self.draw_y = self.y - self.height / 2

    self.normal_speed = speed
    self.speed_adjust = speed_adjust

  def get_rand_speed(self):
    x = np.random.random() - 0.5
    return self.normal_speed + x * self.speed_adjust
  
  def change_location(self, del_x, del_y):
    self.x += del_x
    self.draw_x += del_x
    self.y += del_y
    self.draw_y += del_y
  
  def move_towards(self, target_location):
    speed = self.get_rand_speed()
    x, y = target_location
    vector_x = x - self.x
    vector_y = y - self.y
    mag = (vector_x**2 + vector_y**2) ** 0.5
    if mag <= speed:
      self.change_location(vector_x, vector_y)
    else:
      unit_x = vector_x / mag
      unit_y = vector_y / mag
      unit_x *= speed
      unit_y *= speed
      self.change_location(unit_x, unit_y)

  def restart(self):
    self.x = self.init_x
    self.draw_x = self.x - self.width / 2
    self.y = self.init_y
    self.draw_y = self.y - self.height / 2
