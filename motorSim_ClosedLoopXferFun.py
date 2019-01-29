# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 09:57:28 2019

@author: aberger
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from pylab import *
from scipy import signal


K_T = 7.5e-3    #torque constant, N*m/A
J = 1.66e-5     #moment of inertia
K_D = 1.33e-8   #damping constant
I_Q = 0.5       #Amps

b = K_T/J
a = K_D/J

"""
motor transfer function for theta
(K_T/J)/(s^2  + (K_D/J)*s) = b/(s^2 + a*s)
"""
#tf_num = [2*K_T*I_Q]
#tf_den = [J, 2*zeta]

"""
motor with PID feedback
"""
k_p = 0.5
k_i = 0.5
k_d = 0.1

s = np.linspace(0,20,500)
tf = (b*k_d*s**2 + b*k_p*s + b*k_i)/(s**3 + (b*k_d + a)*s**2 + b*k_p*s + b*k_i)

#I don't think this works. You can't simply call s = f (i.e. purely real)
plt.figure(1)
plt.plot(s,20*np.log10(tf))
plt.xlabel('frequency (Hz)')
plt.ylabel('response')

#instead, try creating an LTI system (linear time invariant)
system = signal.lti([b*k_d, b*k_p, b*k_i], [1, (b*k_d + a), b*k_p, b*k_i])
omega = 2*pi*s

omega, mag, phase = signal.bode(system, omega)

plt.figure(2)
plt.plot(s, mag)

plt.figure(3)
plt.plot(s, phase)