q# -*- coding: utf-8 -*-
"""
Written by A. Berger on 12/5/2018

This program performs inverse Clarke and Park transforms to convert control 
currents in the DQ reference frame to control voltages in the UVW reference
frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt

motorAngle = np.linspace(0,65535,100)

V_D = np.dtype('float16')
V_Q = np.dtype('float16')

filename = r'D:\Documents\Projects\SR544\Data\8Y16_A_currentQ_corr.csv'
Q_corr = np.loadtxt(filename, delimiter=',', usecols=[0], skiprows=1)

def invPark(V_D, V_Q, motorAngle):
    V_alpha = np.dtype('float16')
    V_beta = np.dtype('float16')
    theta = motorAngle*2*pi/65536
    
    V_alpha = V_D*np.cos(theta) - V_Q*np.sin(theta)
    V_beta = V_D*np.sin(theta) + V_Q*np.cos(theta)
    
    return [V_alpha,V_beta]

def invPark_withCorr(V_D, V_Q, motorAngle):
    V_alpha = np.dtype('float16')
    V_beta = np.dtype('float16')
    theta = motorAngle*2*pi/65536
    tickCount = np.floor(motorAngle*100/65536).astype(int)
    
    V_alpha = V_D*np.cos(theta) - (V_Q + 80*Q_corr[tickCount])*np.sin(theta)
    V_beta = V_D*np.sin(theta) + (V_Q + 80*Q_corr[tickCount])*np.cos(theta)
    
    return [V_alpha,V_beta]

def invClarke(V_alpha, V_beta, motorAngle):
    
    V_U = V_alpha
    V_V = 1/2*(-V_alpha + sqrt(3)*V_beta)
    V_W = 1/2*(-V_alpha - sqrt(3)*V_beta)
    
    return [V_U, V_V, V_W]

def RoundTo12BitOutput(arrayIn):
    return np.rint(arrayIn*(2**12))

V_Q = 0.5
V_D = 0
 
#without Q-axis correction   
[V_alpha,V_beta] = invPark(V_D, V_Q, motorAngle)
[V_U, V_V, V_W] = invClarke(V_alpha, V_beta, motorAngle)
Out_U = RoundTo12BitOutput(V_U)
Out_V = RoundTo12BitOutput(V_V)
Out_W = RoundTo12BitOutput(V_W)

#with Q-axis correction
[V_alpha_corr,V_beta_corr] = invPark_withCorr(V_D, V_Q, motorAngle)
[V_U_corr, V_V_corr, V_W_corr] = invClarke(V_alpha_corr, V_beta_corr, motorAngle)
Out_U_corr = RoundTo12BitOutput(V_U_corr)
Out_V_corr = RoundTo12BitOutput(V_V_corr)
Out_W_corr = RoundTo12BitOutput(V_W_corr)

theta = motorAngle*2*pi/65536
"""
plt.figure(1)
plt.plot(theta,np.cos(theta))
plt.plot(theta,np.sin(theta))


plt.figure(2)
plt.plot(theta,V_U,'c')
plt.plot(theta,V_V,'m')
"""

plt.figure(3)
plt.plot(theta,Out_U_corr,'r',marker='.')
plt.plot(theta,Out_V_corr,'g',marker='.')
plt.plot(theta,Out_W_corr,'b',marker='.')
plt.legend(('U','V','W'))
plt.xlabel('motor angle (rad)')
plt.ylabel('DAC output (DAC code)')


