# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:20:07 2020

@author: aberger

This program imports four columns of data:
    1. enc raw count (0-99)
    2. input capture of shaft edges on "free running" 60 MHz FTM counter
    3. chop wheel raw count (0-99 for 100-slot outer track)
    4. input capture of chop edges on "free running" 60 MHz FTM counter
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random

plt.close('all')

filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_80Hz_10100Blade_UVWconnected.txt'

data = np.loadtxt(filename, delimiter=' ', usecols=[0,1,2,3], skiprows=0)

encCount = data[:,0]
encEdge = data[:,1]
chopCount = data[:,2]
chopEdge = data[:,3]

N_samples = len(encCount)
N_enc = 100 #number of ticks on shaft encoder
N_chop = 100 #number of apertures in chopper blade
f_FTM = 60e6 #Hz
FTM_MOD = 4096 #FTM_MOD for the FTM peripheral used to collect these data

dt = FTM_MOD/f_FTM
t = np.linspace(0, dt*N_samples, N_samples)
encoderCount = np.linspace(0, N_enc - 1, N_enc)

def measureCountDeltas(counts, edges, time):
    count1 = np.zeros(0)
    delta = np.zeros(0)
    t1 = np.zeros(0)
    for i, count in enumerate(counts[1:], start=1):
        if counts[i] != counts[i-1]:
            count1 = np.append(count1, counts[i])
            t1 = np.append(t1, time[i])
            delta = np.append(delta, edges[i] - edges[i-1])
            
    return count1, delta, t1


def movingAverage(data, windowSize):
    avg = np.zeros(len(data))
    delta = int(np.floor(windowSize/2))
    
    for i in range(delta):
        avg[i] = data[i]
        
    for i in range(len(data) - delta, len(data)):
        avg[i] = data[i]
        
    for i, datum in enumerate(data[delta:-delta], start=delta):
        avg[i] = np.sum(data[i - delta: i + delta + 1])/(windowSize+1)
    
    return avg

encCountAtDelta, encDelta, encTimeAtDelta = measureCountDeltas(encCount, encEdge, t)
chopCountAtDelta, chopDelta, chopTimeAtDelta = measureCountDeltas(chopCount, chopEdge, t)

encDeltaT_sec = encDelta/f_FTM
chopDeltaT_sec = chopDelta/f_FTM

encSpeed = 1/(N_enc*encDeltaT_sec)
chopSpeed = 1/(N_chop*chopDeltaT_sec)
        
windowSize = 100
avgEncSpeed = movingAverage(encSpeed, windowSize)
avgChopSpeed = movingAverage(chopSpeed, windowSize)

fig1, ax1 = plt.subplots()
ax1.plot(encTimeAtDelta[1:], encSpeed[1:])
ax1.plot(encTimeAtDelta[1:], avgEncSpeed[1:])
ax1.set_ylim(min(encSpeed[1:]), max(encSpeed[1:]))
ax1.set_xlabel('time (s)')
ax1.set_ylabel('speed (rev/s)')
ax1.legend(('shaft encoder','windowed average, N='+str(windowSize)))
ax1.set_title('Free Spindle Decay: speed vs. time')

encTickSpacing = avgEncSpeed*encDeltaT_sec

indexOfZerothTick = np.where(encCountAtDelta == 0)

numRevsToWait = 3
# wait at least numRevsToWait revolutions before considering data
startingIndexOfFirstGoodRev = indexOfZerothTick[0][numRevsToWait]

fig2, ax2 = plt.subplots()
for i in range(3):
    ax2.plot(encoderCount, encTickSpacing[startingIndexOfFirstGoodRev + (i*N_enc): startingIndexOfFirstGoodRev + (i+1)*N_enc])
ax2.plot(encoderCount, 1/N_enc*np.ones(N_enc),'--')
ax2.set_xlabel('encoder count')
ax2.set_ylabel(r'$\Delta \theta$ (revs)')
ax2.legend(('rev '+str(numRevsToWait), 'rev '+str(numRevsToWait + 1), 'rev '+str(numRevsToWait + 2), 'ideal'))
ax2.set_title('Tick Spacing, '+r'$\Delta \theta_i = \bar{f}_i*\Delta t_i$')
fig2.tight_layout()

N_revs = 10
distFromZerothTick = np.zeros((N_enc, N_revs))
i = 0
for index in indexOfZerothTick[0]:
    if i >= N_revs:
        break
    if index > 300:
        distFromZerothTick[:,i] = np.cumsum(encTickSpacing[index:index+N_enc]) - 1/N_enc
        i += 1
        

tickCorrection = np.zeros((N_enc, N_revs))
for i, col in enumerate(distFromZerothTick.T):
    tickCorrection[:,i] = encoderCount/N_enc - col

avgTickCorrection = np.mean(tickCorrection, axis=1)
stdTickCorrection = np.std(tickCorrection, axis=1)

offsetTickCorrection = np.mean(avgTickCorrection)

fig3, ax3 = plt.subplots()
ax3.errorbar(encoderCount, avgTickCorrection - offsetTickCorrection, yerr=stdTickCorrection, marker='.', capsize=4.0)
ax3.set_xlabel('encoder count')
ax3.set_ylabel('tick correction (revs)')
ax3.set_title('Tick correction, '+r'$\langle \theta_i \rangle - \theta_i = \frac{i}{N_{enc}} - \sum_{k=0}^i \Delta \theta_k$', y = 1.03)
fig3.tight_layout()
