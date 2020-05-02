import numpy as np
from numpy.linalg import inv

from LocalGPS.Station import Station, c_lum


def pdot(u,v):
   return u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3]
   
class Network (object):
   def __init__(self):
      self.lstat = []
      
   def addStation(self, s):
      self.lstat.append(self)
   
   def createStation(self, x,y,z):
      self.lstat.append(Station(x,y,z))
      
   def simulate(self, x,y,z,t):
      n = len(self.lstat)
      tk = np.empty(n)
      for i in range(n):
         s = self.lstat[i]
         tk[i] = s.mesure(x,y,z,t)
      return tk
      
   def locate(self, tk):
      nstat = len(self.lstat)
      B = np.empty((nstat,4))
      a = np.empty(nstat)
      e = np.ones(nstat)
      for i in range(nstat):
         s = self.lstat[i]
         B[i,:] = (s.x, s.y, s.z, -c_lum*tk[i])
         a[i] = 1./2.*(s.x**2 + s.y**2 + s.z**2 - c_lum**2*tk[i]**2)
         
      Bp = inv(B.T@B)@B.T
      Bpe = Bp@e
      Bpa = Bp@a

      a2 = pdot(Bpe,Bpe)
      b2 = -2 + 2*pdot(Bpe,Bpa)
      c2 = pdot(Bpa,Bpa)
      
      dlt = b2**2 - 4*a2*c2
      l1 = -b2/(2*a2) + np.sqrt(dlt)/(2*a2)
      l2 = -b2/(2*a2) - np.sqrt(dlt)/(2*a2)
      
      u1 = np.dot(Bp,a+l1*e)
      u2 = np.dot(Bp,a+l2*e)
      
      v1 = np.dot(B,u1)-a-l1*e
      v2 = np.dot(B,u2)-a-l2*e
      j1 = np.dot(v1,v1)
      j2 = np.dot(v2,v2)
      
      if j1 < j2:
         x,y,z,ct = u1
      else:
         x,y,z,ct = u2
      
      return x,y,z,ct/c_lum