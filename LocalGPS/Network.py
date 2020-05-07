import numpy as np
from numpy.linalg import inv

from LocalGPS.Station import Station, c_lum

   
class Network (object):
   """Class that represents a network of emitting stations

   """
   def __init__(self):
      self.lstat = []
      
   def addStation(self, s : Station):
      """Adds a station to the network
      
      Args:
         s
            Station to add to the network
         
      """
      self.lstat.append(s)
   
   def createStation(self, x : float,y : float,z : float) -> Station:
      """Creates a station and adds it to the network
      
      Args:
         m x, y, z
            Cartesian coordinates of the station

      Returns:
         Station created

      """
      stat = Station(x,y,z)
      self.addStation(stat)
      return stat

   def simulate(self, x : float,y : float,z : float,t : float) -> np.array:
      """Simulates a set of measurements. A measurement is the time of 
      
      Args:
         m x, y, z
            Cartesian coordinates of the receiver
         s t
            Local time of the receiver

      Returns:
         Array of measurements

      """
      n = len(self.lstat)
      tk = np.empty((n,4))
      for i in range(n):
         s = self.lstat[i]
         tk[i,:3] = (s.x,s.y,s.z)
         tk[i,3] = s.mesure(x,y,z,t)
      return tk
