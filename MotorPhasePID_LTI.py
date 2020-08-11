# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:01 2020

This script creates response plots for a motor + PID control transfer function
of motor phase

@author: aberger
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy import pi, sqrt, logspace

#assume some final speed in order to approximate damping
f_final = 50     #Hz
omega_final = 2*pi*f_final

K_T = 5.55e-3    #torque constant, N*m/A
J = 1.82e-5     #moment of inertia, kg*m^2
K_D = 1.42e-8   #damping constant, N*m*s^2 (assuming damping torque = K_D*omega^2)
Gamma = K_D*omega_final #assuming linear damping, Gamma*omega

#from motor PID tuning. In firmware, these constants are defined with respect
#to motor phase in revs. Transfer function is given in terms of rad

"""
# original values of phase PID
k_p = 0.5/(2*pi)      
k_i = 0.5/(2*pi)      
k_d = 5/(2*pi)
"""

#To include the effect of speed PID:
# k_p = k_p,phase + k_i,speed
# k_i = k_i,phase (no change)
# k_d = k_d,phase + k_p,speed
k_p = (0.8926 + 0.0184)/(2*pi)      
k_i = 3.484/(2*pi)      
k_d = (0.0369 + .0231)/(2*pi)

"""
motor transfer function
In general
H(s) = H_motor(s)*H_PID(s)/(1 + H_motor(s)*H_PID(s))

where H(s) is the transfer function between motor phase Theta(s) and reference
phase Theta_ref(s):
Theta(s) = H(s)*Theta_ref(s)

and
H(s) = ( K_T*k_d*s**2 + K_T*k_p*s + K_T*k_i ) / ( J*s**3 + (Gamma + K_T*k_d)*s**2 + K_T*k_p*s + K_T*k_i )
"""

tf_num = [K_T*k_d, K_T*k_p, K_T*k_i]
tf_den = [J, (Gamma + K_T*k_d), K_T*k_p, K_T*k_i]

motor_tf = signal.TransferFunction(tf_num, tf_den)

f = logspace(-2,3)
omega = 2*pi*f

omega, mag, phase = signal.bode(motor_tf, omega)

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


t = np.linspace(0, 100, 20001)
t1,y1 = signal.step(motor_tf, T=t) #t1 in seconds, y1 is normalized response

plt.figure(3)
plt.plot(t1,y1)
plt.xlabel('time (s)')
plt.ylabel('step response')

print("poles at: "+str(motor_tf.poles)+"rad/s")
print("zeros at: "+str(motor_tf.zeros)+"rad/s")