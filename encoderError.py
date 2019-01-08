# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 09:51:58 2019

@author: aberger
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random

theta_m = linspace(0,2*pi,200)

f_m = linspace(4,100,10) #Hz
f_FTM = 14648 #Hz

for i, val in enumerate(f_m):
    plt.plot(180/pi*theta_m, 180/pi*2*np.cos(theta_m + pi*(f_m[i] - 22)/f_FTM)*np.sin(pi*(f_m[i] - 22)/f_FTM),label=str(round(f_m[i],2)))
    
plt.xlabel('motor angle (deg)')
plt.ylabel('encoder error (deg)')
plt.legend(loc=4)