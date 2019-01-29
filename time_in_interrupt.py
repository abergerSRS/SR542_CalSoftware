# -*- coding: utf-8 -*-
"""
Written by A. Berger on 1/29/2019


"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random

filename = r'F:\scope_183.csv'

file = open(filename)
reader = csv.reader(file,delimiter=',')

time_s= []
ch1_busy = []
ch2_busy = []
ch3_busy = []

#skip the first two rows
next(reader)
next(reader)
        
for row in reader:
    if(row[1] != ''):
        time_s.append(float(row[0]))
        ch1_busy.append(1 if (float(row[1]) > 0.75) else 0)
        ch2_busy.append(1 if (float(row[2]) > 0.75) else 0)
        ch3_busy.append(1 if (float(row[3]) > 0.75) else 0)
    else:
        break
        


time_s = np.asarray(time_s)
ch1_busy = np.asarray(ch1_busy)
ch2_busy = np.asarray(ch2_busy)
ch3_busy = np.asarray(ch3_busy)

N = len(time_s)
ch1_busy_time = np.sum(ch1_busy)
ch2_busy_time = np.sum(ch2_busy)
busy = np.logical_or(ch1_busy,ch2_busy)
total_busy_time = np.sum(busy)

plt.plot(time_s,ch1_busy)
plt.plot(time_s,ch2_busy+1.01)
plt.plot(time_s,busy+2.02)
plt.xlabel('time (s)')
plt.ylabel('state (hi/low)')
plt.legend(('DDS update','motor update','DDS OR motor update'))

print("ch 1 busy percentage = "+str(100*ch1_busy_time/N))
print("ch 2 busy percentage = "+str(100*ch2_busy_time/N))
print("ch 1 OR 2 busy percentage = "+str(100*total_busy_time/N))
   


