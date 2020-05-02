import numpy as np

from Network import Network


net = Network()
d0 = 1000.

net.createStation(0,0,0)
net.createStation(d0,0,0)
net.createStation(d0,d0,0)
net.createStation(0,0,0)
net.createStation(0,0,d0)
net.createStation(d0,0,d0)
net.createStation(d0,d0,d0)
net.createStation(0,0,d0)

x0=100.
y0=-2*d0
z0=50.
t0=0.

tk = net.simulate(x0,y0,z0,t0)
x,y,z,t = net.locate(tk)
# print("x=%.2fm, y=%.2fm, z=%.2fm, t=%.2fs" % (x,y,z,t))
print("dx=%.2f m, dy=%.2f m, dz=%.2f m, dt=%.2f us" % (x-x0,y-y0,z-z0,1e6*(t-t0)))


