# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:44:50 2019

This script creates response plots for a motor + PI control transfer function
of motor speed, derived in notebook #5, pg 98-104

@author: aberger
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy import pi, sqrt, logspace

#assume some final speed in order to approximate damping
f_final = 50     #Hz
omega_final = 2*pi*f_final

K_T = 7.5e-3    #torque constant, N*m/A
J = 1.66e-5     #moment of inertia
K_D = 1.33e-8   #damping constant, N*m*s^2 (assuming damping torque = K_D*omega^2)
Gamma = K_D*omega_final #assuming linear damping, Gamma*omega

#from motor PID tuning. In firmware, these constants are defined with respect
#to motor frequency in Hz. Transfer function is given in terms of s [rad/s]
k_p = 0.02/(2*pi)      #A/Hz
k_i = 0.01/(2*pi)      #A/s*s =A

"""
motor transfer function
In general
H(s) = H_motor(s)*H_PI(s)/(1 + H_motor(s)*H_PI(s))

where H(s) is the transfer function between motor frequency Omega and reference
frequency Omega_ref:
Omega(s) = H(s)*Omega_ref(s)

and
H(s) = K_T*k_p/J*( (s + k_i/k_p) / (s^2 + s*(K_T*k_p + Gamma)/J + K_T*k_i/J) )
"""

tf_num = [K_T*k_p, K_T*k_i]
tf_den = [J, (K_T*k_p + 2*Gamma), K_T*k_i]

motor_tf = signal.TransferFunction(tf_num, tf_den)

f = logspace(-2,3)
omega = 2*pi*f

omega, mag, phase = signal.bode(motor_tf,omega)

plt.figure(1)
plt.semilogx(f, mag);   #Bode magnitude plot
mag_3db = -3*np.ones(len(f));
plt.semilogx(f, mag_3db,linestyle='dashed');
plt.ylabel('Magnitude Response (dB)')
plt.xlabel('Freq (Hz)')
plt.legend(('|H(s)|','-3dB'))

plt.figure(2)
plt.semilogx(f, phase); #Bode phase plot
plt.ylabel('Phase Response (deg)')
plt.xlabel('Freq (Hz)')


t1,y1 = signal.step(motor_tf) #t1 in seconds, y1 is normalized response

plt.figure(3)
plt.plot(t1,y1)
plt.xlabel('time (s)')
plt.ylabel('motor freq response (Hz/Hz)')

print("poles at: "+str(motor_tf.poles)+"rad/s")
print("zeros at: "+str(motor_tf.zeros)+"rad/s")