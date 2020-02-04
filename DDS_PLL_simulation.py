# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:21:10 2019

@author: aberger
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

f_ref = 50      #Hz
f_FTM = 60e3    #Hz

mod_FTM = int(64)
f_DDS = f_FTM/mod_FTM

T_ref = 1/f_ref #s
T_FTM = 1/f_FTM #s

total_time = 2*T_ref
dt = T_FTM
N = int(total_time/dt)

t = linspace(0,total_time,N)
n = np.arange(0,N,1,dtype=int)

FTM_CNT = n%(mod_FTM-1)

DDS_phase = np.zeros(N)
linear_phase = (t/T_ref)%1

FTW = (f_ref/f_DDS)

i = 0
for x in FTM_CNT:
    if x == 0:
        DDS_phase[i] = (DDS_phase[i-1] + FTW)%1
    else:
        DDS_phase[i] = DDS_phase[i-1]
    i += 1       
    
phase_error = DDS_phase - linear_phase

plt.plot(t,FTM_CNT/mod_FTM,marker='o')
plt.plot(t,DDS_phase)
plt.plot(t,linear_phase)
plt.xlabel('time (s)')

plt.figure(2)
plt.plot(t,phase_error)
