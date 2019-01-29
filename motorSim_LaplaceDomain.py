# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 09:57:28 2019

@author: aberger
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy import pi, sqrt

K_T = 7.5e-3    #torque constant, N*m/A
J = 1.66e-5     #moment of inertia
K_D = 1.33e-8   #damping constant
I_Q = 0.5       #Amps

zeta = sqrt(K_D*K_T*I_Q)

"""
motor transfer function
2*K_T*I_Q/(J*s  + 2*zeta)
"""
#tf_num = [2*K_T*I_Q]
#tf_den = [J, 2*zeta]

"""
motor with PI feedback
(b*k_p*s + b*k_i)/(s^2 + (a + b*k_p)*s + b*k_i)
"""
b = 2*K_T/J
a = 2*zeta/J
k_p = 0.02
k_i = 0.01

tf_num = [b*k_p, b*k_i]
tf_den = [1, (a + b*k_p), b*k_i]

motor_tf = signal.TransferFunction(tf_num, tf_den)
t1,y1 = signal.step(motor_tf) #t1 in seconds, y1 in rad/s
y1 = y1/(2*pi) #convert y1 to Hz

plt.figure(1)
plt.plot(t1,y1)
plt.xlabel('time (s)')
plt.ylabel('motor freq (Hz)')