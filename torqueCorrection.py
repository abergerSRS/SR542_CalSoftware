# -*- coding: utf-8 -*-
"""
Written by A. Berger on 12/17/2018

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random
#from scipy.optimize import curve_fit

filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\angleRecord_8Y16_A_3Hz_angleCorr.txt'

rawAngle = [] 
residualAngle = [] 

data = np.loadtxt(filename, delimiter=' ', usecols=[0,1], skiprows=0)

dt = 0.004369066688
N = len(data[0:,0])
time = linspace(0,(N-1)*dt,N)
rawAngle = 2*pi*data[0:,1]/(2**32) #in rad
uwAngle = np.unwrap(rawAngle) #in rad
#treat these as an ordered pair

def resample(x,y,dec):
    #resample (x,y) such that len(x_out) = len(x)/dec
    #uses a two-pass method to calculate the average of y at the new bin spacing,
    #followed by the standard deviation of y_resamp based on the original y data
    
    Nsamp = int(len(x)/dec)
    dx = max(x)/(Nsamp+1)
    
    x_resamp = linspace(0,max(x)-dx,Nsamp)
    
    
    y_resamp = np.zeros(Nsamp)
    y_stdev = np.zeros(Nsamp)
    
    i = 0 #iterator over x_resamp values
    for x_n in x_resamp:
        
        n = 0
        total = 0
        j = 0 #iterator over y values
        for value in x:
            if(x_n <= value < (x_n + dx)):
                total += y[j]
                n += 1
            j += 1
            
        y_resamp[i] = total/n
        #print(res_avg[i])
        
        #again iterate over y values to calculate standard deviation
        sum_sq = 0
        j = 0
        for value in x:
            if(x_n <= value < (x_n + dx)):
                sum_sq += (y[j] - y_resamp[i])**2
            j += 1
            
        y_stdev[i] = np.sqrt(sum_sq/(n - 1))
        #print(res_stdev[i])
        
        i += 1
        
    return [x_resamp,y_resamp,y_stdev]

omega = np.gradient(uwAngle,dt)
alpha = np.gradient(omega,dt)

[tickCount,alpha_avg,alpha_std] = resample(rawAngle,alpha,10)
plt.plot(rawAngle,alpha,marker='.',linestyle='None')
plt.errorbar(tickCount,alpha_avg, yerr=alpha_std,marker='o',linestyle='None')

#p = np.polyfit(time,uwAngle,3)
#pfit = np.polyval(p,time)

#resAngle = uwAngle - pfit

#plot raw data
#plt.figure(1)
#plt.plot(time,rawAngle)

#plot unwrapped data, along with polynomial fit
"""
plt.figure(2)
plt.plot(time,uwAngle, color='b',marker='o',linestyle='None')
plt.plot(time,pfit)

#plot residual angle
plt.figure(3)
plt.plot(time,resAngle)

#calculate first and second derivative of residual angle data
omega = np.gradient(resAngle,dt)
alpha = np.gradient(omega,dt)

plt.figure(4)
plt.plot(time,alpha)
"""




"""


numPts = 100
dx = 1/numPts
x = np.linspace(0,1-dx,numPts)
res_avg = np.zeros(numPts)
res_stdev = np.zeros(numPts)
slope = np.zeros(numPts)

compAngle = np.zeros(len(rawAngle))
corrAngle = np.zeros(len(rawAngle))
compAngle_noSlope = np.zeros(len(rawAngle))
corrAngle_noSlope = np.zeros(len(rawAngle))
    
#now that the average residuals have been calculated, calculate the slope 
#between each residual    
i = 0        
for x_n in x:
    if(i == numPts - 1):
        slope[i] = (res_avg[i] - res_avg[i-1])/dx
    else:
        slope[i] = (res_avg[i+1] - res_avg[i])/dx
    
    #large slope rejection
"""
#if(np.abs(slope[i]) > 0.005):
#    slope[i] = 0
"""
    i += 1
    
#finally, check the calibration
i = 0
for angle in rawAngle:
    index = int(np.floor(100*angle))
    compAngle[i] = res_avg[index] + slope[index]*(angle - np.floor(100*angle)/100)
    compAngle_noSlope[i] = res_avg[index] #ignoring slope
    corrAngle[i] = residualAngle[i] - compAngle[i]
    corrAngle_noSlope[i] = residualAngle[i] - compAngle_noSlope[i]
    i += 1
    
integerResiduals = [int(x) for x in (res_avg*(2**32))]
        
       
np.savetxt(r'D:\Documents\Projects\SR544\Data\residual_averages.txt',np.transpose([x,res_avg,res_stdev,slope]),newline='\r\n',delimiter=',')
np.savetxt(r'D:\Documents\Projects\SR544\Data\angleCorr_LUT.txt',integerResiduals,newline=',\r\n',fmt='%u')
    
plt.figure(1)
#plt.errorbar(x,res_avg, yerr=res_stdev,color='g', marker='.',linestyle='None')
plt.plot(rawAngle,residualAngle, color='b',marker='o',linestyle='None')
plt.plot(rawAngle,compAngle, color='r', marker='_', linestyle='None')
plt.plot(rawAngle,compAngle_noSlope,color='y',marker='_',linestyle='None')
plt.plot(rawAngle,corrAngle,color='c',marker='.',linestyle='None')
plt.plot(rawAngle,corrAngle_noSlope,color='k',marker='.',lineStyle='None')
plt.legend(('residual angle','comp','comp w/o slope','corrected angle','corrected w/o slope'))
plt.ylabel('raw angle (revs)')
plt.xlabel('residual angle (revs)')
"""