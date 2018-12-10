# -*- coding: utf-8 -*-
"""
Written by A. Berger on 12/5/2018

This program reads in CSV data of raw motor angle (measured during constant 
current_Q operation) and residual angle (calculated in IGOR assuming constant
motor speed). The residual angle data is then binned and averaged based on
its ordered pairing with the raw angle data. The average and standard deviation
are calculated based on a two-pass method through the original data.
"""

import numpy as np
import matplotlib.pyplot as plt
#from numpy import sqrt, pi, exp, linspace, random
#from scipy.optimize import curve_fit

filename = r'D:\Documents\Projects\SR544\Data\8Y16_A_residualAngle_trial5.csv'

rawAngle = [] 
residualAngle = [] 

data = np.loadtxt(filename, delimiter=',', usecols=[0,1], skiprows=1)

rawAngle = data[0:,0]
residualAngle = data[0:,1]
#treat these as an ordered pair

numPts = 100
dx = 1/numPts
x = np.linspace(0,1-dx,numPts)
res_avg = np.zeros(numPts)
res_stdev = np.zeros(numPts)
slope = np.zeros(numPts)

compAngle = np.zeros(len(rawAngle))
corrAngle = np.zeros(len(rawAngle))

i = 0 #iterator over x values
for x_n in x:
    
    n = 0
    total = 0
    j = 0 #iterator over rawAngle values
    for value in rawAngle:
        if(x_n <= value < (x_n + dx)):
            total += residualAngle[j]
            n += 1
        j += 1
        
    res_avg[i] = total/n
    #print(res_avg[i])
    
    sum_sq = 0
    j = 0
    for value in rawAngle:
        if(x_n <= value < (x_n + dx)):
            sum_sq += (residualAngle[j] - res_avg[i])**2
        j += 1
        
    res_stdev[i] = np.sqrt(sum_sq/(n - 1))
    #print(res_stdev[i])
    
    i += 1
    
#now that the average residuals have been calculated, calculate the slope 
#between each residual    
i = 0        
for x_n in x:
    if(i == numPts - 1):
        slope[i] = (res_avg[i] - res_avg[i-1])/dx
    else:
        slope[i] = (res_avg[i+1] - res_avg[i])/dx
    
    i += 1
    
#finally, check the calibration
i = 0
for angle in rawAngle:
    index = int(np.floor(100*angle))
    compAngle[i] = res_avg[index] + slope[index]*(angle - np.floor(100*angle)/100)
    corrAngle[i] = residualAngle[i] - compAngle[i]
    i += 1
        
       
np.savetxt(r'D:\Documents\Projects\SR544\Data\residual_averages.txt',np.transpose([x,res_avg,res_stdev,slope]),newline='\r\n',delimiter=',')
    
plt.figure(1)
#plt.errorbar(x,res_avg, yerr=res_stdev,color='g', marker='.',linestyle='None')
plt.plot(rawAngle,residualAngle, color='b',marker='o',linestyle='None')
plt.plot(rawAngle,compAngle, color='r', marker='_', linestyle='None')
plt.plot(rawAngle,corrAngle,color='g',marker='.',linestyle='None')
plt.ylabel('raw angle (revs)')
plt.xlabel('residual angle (revs)')