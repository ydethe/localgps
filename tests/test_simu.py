import unittest

import numpy as np

from LocalGPS.Network import Network
from LocalGPS.Receiver import Receiver


class TestSimu (unittest.TestCase):
    def test_simu(self):
        net = Network()
        rec = Receiver()

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
        x,y,z,t = rec.locate(tk)
        
        self.assertAlmostEqual(x,x0, delta=0.01)
        self.assertAlmostEqual(y,y0, delta=0.01)
        self.assertAlmostEqual(z,z0, delta=0.01)
        self.assertAlmostEqual(t,t0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
    