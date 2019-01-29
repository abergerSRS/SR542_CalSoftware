# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:39:43 2019

@author: aberger
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random

dt = 0.004 #s

N = 10000
time = linspace(0,(N-1)*dt,N)

f_0 = 0.1 #Hz
omega_0 = f_0*2*pi #rad/s

A = .025
theta = omega_0*time + A*np.sin(2*omega_0*time)

p = np.polyfit(time,theta,1)
resAngle = theta - np.polyval(p,time)

plt.plot(time,resAngle)