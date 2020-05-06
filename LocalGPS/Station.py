import numpy as np


c_lum = 3.e8

class Station (object):
   def __init__(self, x,y,z):
      self.x = x
      self.y = y
      self.z = z
      
   def mesure(self, x,y,z,t):
      d = np.sqrt((x-self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
      tm = d/c_lum + t
      return tm + np.random.normal()*1e-9*0
      