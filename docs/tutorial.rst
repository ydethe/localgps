.. Licensed under the MIT Licensed

.. _tutorial:

========
Tutorial
========

Use-case #1
===========
This case consists in :

* a setpoint, named *stp* :class:`SystemControl.SetPoint.Step`
  This element provides a user command such as a step or a ramp or a Dirac, ...
  Here, we will use a unit step function

* a LQ regulator, named *ctl* :class:`SystemControl.Controller.LQRegulator`

  This element compares the estimated state of the system and compares to the setpoint.
  It controls the system so that it reaches the setpoint

* a LTI system, named *sys* :class:`SystemControl.System.LTISystem`

  This element simulates the system

The command *u* goes from the controller to the system, and modifies its behavior.

.. figure:: UseCase1.png
   :align: center
   :alt: Schematic of the setup

   Simulation diagram of the use-case
   
Let the system model a mass **m** linked to a spring with rate **k**, sliding over a horizontal plane.
We note **x** the position of the masse, and **X** the state vector :math:`X=(x, x')^T`.

:math:`x=0` when the spring is not stressed.

You can also apply a force to the system, named **U**, thanks to an inductive device (for example).

According to `Hooke's law <https://en.wikipedia.org/wiki/Spring_(device)#Hooke%27s_law>`_, the equation of the movement is:

:math:`m.x''=-k.x + U`

This can be rewritten as :

:math:`X'=f(t,X,U)`

With :

:math:`f(t,X,U) = \begin{pmatrix} v\\ -k/m.x + U/m \end{pmatrix}`

So you first implement this behavior with a class named **MySystem**, which implements the method **transition**::

    import numpy as np
    from SystemControl.System import LTISystem

    m = 1. # Mass
    k = 40. # Spring rate
    sys = LTISystem('sys', name_of_outputs=['x'], name_of_states=['x','v'])
    sys.A = np.array([[0,1],[-k/m,0]])
    sys.B = np.array([[0,1/m]]).T
    sys.C = np.array([[1, 0]])
    sys.D = np.array([[0]])
    sys.setInitialState(np.array([-1.,0.]))
    
You can simulate this system alone::

    # 1 s of simulation time. The solver subdivides automatically this time-step.
    # Every subclass of ASystem has an input named 'command'
    # Here, no command is applied.
    sys.reset()
    sys.update(0,1,{'command':np.array([0.])})
    x = sys.getState()
    print(x)
    
This will print::

    [-0.99914262  0.2615665 ]
    
Now, let's use a LQ regulator::

    from SystemControl.Controller import LQRegulator

    A = sys.A
    B = sys.B
    C = np.hstack((B, A@B))
    Q = C.T@C*100
    R = np.eye(1)
    ctl = LQRegulator(name='ctl', name_of_outputs=['u'])
    ctl.computeGain(sys, Q, R)
    
Finally, we have to provide a set point for the controller.
In this use-case, the set point will be 1::

    from SystemControl.SetPoint import Step
    
    stp = Step(name='stp', name_of_outputs=['x_cons'], cons=1)
    
We create the simulation graph::
    
    from SystemControl.Simulation import Simulation
    
    sim = Simulation()
    # We add the elements to the graph
    sim.addElement(sys)
    sim.addElement(ctl)
    sim.addElement(stp)
    # We connect them
    sim.linkElements(src=ctl, dst=sys, src_data_name='output', dst_input_name='command')
    sim.linkElements(src=sys, dst=ctl, src_data_name='state', dst_input_name='estimation')
    sim.linkElements(src=stp, dst=ctl, src_data_name='output', dst_input_name='setpoint')
    
You can draw the simulation graph::

    import matplotlib.pyplot as plt
    fig = plt.figure()
    axe = fig.add_subplot(111)
    sim.drawGraph(axe)
    plt.show()
    
We can now launch the simulation from t = 0 s to t = 1 s, with a time step of 0.05 s::

    tps = np.arange(0, 2, 0.05)
    sim.simulate(tps)

Now we can get the simulation log and plot some lines::

    log = sim.getLogger()
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axe = fig.add_subplot(111)
    log.plot('t', 'x', axe)
    log.plot('t', 'x_cons', axe)
    
    axe = fig.add_subplot(212, sharex=axe); axe.grid(True)
    log.plot('t', 'u', axe)
    
    plt.show()
    
.. figure:: SimuUseCase1.png
   :align: center
   :alt: Simulation of the setup

   Simulation of the setup
   

Use-case #2
===========
This case consists in :

* a setpoint, named *stp* :class:`SystemControl.SetPoint.Step`
  This element provides a user command such as a step or a ramp or a Dirac, ...
  Here, we will use a unit step function

* a LQ regulator, named *ctl* :class:`SystemControl.Controller.LQRegulator`

  This element compares the estimated state of the system and compares to the setpoint.
  It controls the system so that it reaches the setpoint

* a LTI system, named *sys* :class:`SystemControl.System.LTISystem`

  This element simulates the system

* a linear sensor, named *cpt* :class:`SystemControl.Sensors.LinearSensors`
  
  This elements simulates a noisy measurement of position and speed

.. figure:: UseCase2.png
   :align: center
   :alt: Schematic of the setup

   Simulation diagram of the use-case
   
The equations of the system's dynamic are the same as in the use-case #1.
We only show here the implementation of cpt.

For the sensors::

    from SystemControl.Sensors import LinearSensors

    cpt = LinearSensors('cpt', name_of_outputs=['x_mes','v_mes'])
    # Unbiased measurements
    cpt.mean = np.zeros(2)
    # Half the covariance of the sensor
    cpt.cov  = np.eye(2)/200
    # Same system described as the sensor
    cpt.C    = np.eye(2)
    cpt.D    = np.zeros((2,1))

We create the simulation graph::
    
    from SystemControl.Simulation import Simulation
    
    sim = Simulation()
    # We add the elements to the graph
    sim.addElement(sys)
    sim.addElement(ctl)
    sim.addElement(stp)
    sim.addElement(cpt)
    # We connect them
    sim.linkElements(src=stp, dst=ctl, src_data_name='output', dst_input_name='setpoint')
    sim.linkElements(src=ctl, dst=sys, src_data_name='output', dst_input_name='command')
    sim.linkElements(src=sys, dst=cpt, src_data_name='state', dst_input_name='state')
    sim.linkElements(src=ctl, dst=cpt, src_data_name='output', dst_input_name='command')
    sim.linkElements(src=cpt, dst=ctl, src_data_name='state', dst_input_name='estimation')
    
We can now launch the simulation from t = 0 s to t = 1 s, with a time step of 0.05 s::

    tps = np.arange(0, 4, 0.05)
    sim.simulate(tps)

Now we can get the simulation log and plot some lines::

    log = sim.getLogger()
    
    fig = plt.figure()
    axe = fig.add_subplot(111); axe.grid(True)
    log.plot('t', 'x', axe, label="simulation")
    log.plot('t', 'x_cons', axe, label="set point")
    log.plot('t', 'x_mes', axe, linestyle='', marker='+', label="measure")
    axe.legend(loc='best')
    axe.set_xlabel('t (s)')
    
    plt.show()
    
.. figure:: SimuUseCase2.png
   :align: center
   :alt: Simulation of the setup

   Simulation of the setup
   
   
Use-case #3
===========
This case consists in :

* a setpoint, named *stp* :class:`SystemControl.SetPoint.Step`
  This element provides a user command such as a step or a ramp or a Dirac, ...
  Here, we will use a unit step function

* a LQ regulator, named *ctl* :class:`SystemControl.Controller.LQRegulator`

  This element compares the estimated state of the system and compares to the setpoint.
  It controls the system so that it reaches the setpoint

* a LTI system, named *sys* :class:`SystemControl.System.LTISystem`

  This element simulates the system

* a linear sensor, named *cpt* :class:`SystemControl.Sensors.LinearSensors`
  
  This elements simulates a noisy measurement of position

* a Kalman filter, named *kal* :class:`SystemControl.Estimator.AKalmanFilter`
  
  This elements estimates the state of the system with the measurements given to it.

.. figure:: UseCase3.png
   :align: center
   :alt: Schematic of the setup

   Simulation diagram of the use-case
   
The equations of the system's dynamic are the same as in the use-case #1.
We only show here the implementation of cpt and kal.

The class :class:`SystemControl.Estimator.AKalmanFilter` needs you to implement the abstract methods A, B, C, D, Q and R,
with the following definitions:

For each time step, :math:`X_k=X(t=t_k)`. The discrete system is described by :

:math:`X_{k+1}=A_d(t_k,t_{k+1}).X_k+B_d(t_k,t_{k+1}).U_k`

:math:`Y_k=C_d(t_k,t_{k+1}).X_k+D_d(t_k,t_{k+1}).U_k`

To compute the discrete A matrix, you have to exponentiate it. Assuming it constant :
:math:`A_d(t_k,t_{k+1})=e^{A.(t_{k+1}-t_k)}`

and

:math:`B_d(t_k,t_{k+1})=\int_{0}^{t_{k+1}-t_k} e^{A.u} .B.du`

For the Kalman filter::

    from SystemControl.Estimator import AKalmanFilter

    Kk = 1/m
    Ka = np.sqrt(k/m)
    class MyKal(AKalmanFilter):
        def A(self, t1, t2):
            return np.array(
                [
                    [np.cos(Ka * (t2 - t1)), np.sin(Ka * (t2 - t1)) / Ka],
                    [-Ka * np.sin(Ka * (t2 - t1)), np.cos(Ka * (t2 - t1))],
                ]
            )

        def B(self, t1, t2):
            return np.array(
                [
                    [
                        Kk / Ka ** 2 * (1.0 - np.cos(Ka * (t2 - t1))),
                        Kk / Ka * np.sin(Ka * (t2 - t1)),
                    ]
                ]
            ).T

        def C(self, t):
            return np.array([[1,0]])
        def D(self, t):
            return np.zeros((1,1))
        def Q(self, t):
            return np.eye(2)/10000
        def R(self, t):
            return np.eye(1)/100
            
    kal = MyKal('kal', name_of_outputs=["x_est"], name_of_states=["state_x_est","state_v_est"])

For the sensors::

    from SystemControl.Sensors import LinearSensors

    cpt = LinearSensors('cpt', name_of_outputs=['x_mes'])
    # Unbiased measurements
    cpt.mean = np.zeros(1)
    # Half the covariance of the sensor
    cpt.cov  = kal.R(0)/2
    # Same system described as the sensor
    cpt.C    = kal.C(0)
    cpt.D    = kal.D(0)

We create the simulation graph::
    
    from SystemControl.Simulation import Simulation
    
    sim = Simulation()
    # We add the elements to the graph
    sim.addElement(sys)
    sim.addElement(ctl)
    sim.addElement(stp)
    sim.addElement(cpt)
    sim.addElement(kal)
    # We connect them
    sim.linkElements(src=stp, dst=ctl, src_data_name='output', dst_input_name='setpoint')
    sim.linkElements(src=ctl, dst=sys, src_data_name='output', dst_input_name='command')
    sim.linkElements(src=sys, dst=cpt, src_data_name='state', dst_input_name='state')
    sim.linkElements(src=ctl, dst=cpt, src_data_name='output', dst_input_name='command')
    sim.linkElements(src=cpt, dst=kal, src_data_name='output', dst_input_name='measurement')
    sim.linkElements(src=ctl, dst=kal, src_data_name='output', dst_input_name='command')
    sim.linkElements(src=kal, dst=ctl, src_data_name='state', dst_input_name='estimation')
    
We can now launch the simulation from t = 0 s to t = 1 s, with a time step of 0.05 s::

    tps = np.arange(0, 4, 0.05)
    sim.simulate(tps)

Now we can get the simulation log and plot some lines::

    log = sim.getLogger()
    
    fig = plt.figure()
    axe = fig.add_subplot(111); axe.grid(True)
    log.plot('t', 'x', axe, label="simulation")
    log.plot('t', 'x_cons', axe, label="set point")
    log.plot('t', 'state_x_est', axe, label="estimation")
    log.plot('t', 'x_mes', axe, linestyle='', marker='+', label="measure")
    axe.legend(loc='best')
    axe.set_xlabel('t (s)')
    
    plt.show()
    
.. figure:: SimuUseCase3.png
   :align: center
   :alt: Simulation of the setup

   Simulation of the setup
   
   