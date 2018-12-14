# -*- coding: utf-8 -*-
"""
Written by A. Berger on 12/5/2018

This program performs inverse Clarke and Park transforms to convert control 
currents in the DQ reference frame to control voltages in the UVW reference
frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt

motorAngle = np.linspace(0,65535,65536)

V_D = np.dtype('float16')
V_Q = np.dtype('float16')

def invPark(V_D, V_Q, motorAngle):
    V_alpha = np.dtype('float16')
    V_beta = np.dtype('float16')
    theta = motorAngle*2*pi/65536
    
    V_alpha = V_D*np.cos(theta) - V_Q*np.sin(theta)
    V_beta = V_D*np.sin(theta) + V_Q*np.cos(theta)
    
    return [V_alpha,V_beta]

def invClarke(V_alpha, V_beta, motorAngle):
    
    V_U = V_alpha*4096
    V_V = 1/2*(-V_alpha + sqrt(3)*V_beta)*4096
    V_W = 1/2*(-V_alpha - sqrt(3)*V_beta)*4096
    
    return [V_U, V_V, V_W]

V_Q = 1
V_D = 0
    
[V_alpha,V_beta] = invPark(V_D, V_Q, motorAngle)
[V_U, V_V, V_W] = invClarke(V_alpha, V_beta, motorAngle)
theta = motorAngle*2*pi/65536
"""
plt.figure(1)
plt.plot(theta,np.cos(theta))
plt.plot(theta,np.sin(theta))
"""

plt.figure(2)
plt.plot(theta,V_U,'c')
plt.plot(theta,V_V,'m')


plt.figure(3)
plt.plot(theta,V_U,'r')
plt.plot(theta,V_V,'g')
plt.plot(theta,V_W,'b')
