import numpy as np
from numpy.linalg import inv

from LocalGPS.Station import Station, c_lum


def pdot(u : np.array,v : np.array) -> float:
   """Pseudo scalar product :

   :math:`x.x'+y.y'+z.z'-t.t'`
   
   Args:
      u
         First quadri-vector
      v
         Second quadri-vector
   
   Returns:
      Pseudo scalar product

   """
   return u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3]

class Receiver (object):
   """Class that represents a receiver

   """
   def __init__(self):
      pass
      
   def locate(self, tk):
      nstat = tk.shape[0]
      B = np.empty((nstat,4))
      a = np.empty(nstat)
      e = np.ones(nstat)
      for i in range(nstat):
         x,y,z,t = tk[i,:]
         B[i,:] = (x, y, z, -c_lum*t)
         a[i] = 1./2.*(x**2 + y**2 + z**2 - c_lum**2*t**2)
         
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