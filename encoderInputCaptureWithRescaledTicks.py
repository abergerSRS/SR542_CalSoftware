# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:20:07 2020

@author: aberger

This program imports four columns of data:
    1. enc raw count (0-99)
    2. input capture of shaft edges on "free running" 60 MHz FTM counter
    3. running total of number of encoder edges captured
    4. delta FTM count between most recent two edges, rescaled by tick spacing
    
And is used to validate the shaft encoder calibration performed by encoderInputCapture.py
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random
import os

# custom modules
import fileWriter

plt.close('all')

file_dir = os.path.abspath(r"C:\Users\aberger\Documents\Projects\SR542\Firmware\SR544\tools")
#filename = 'edgesAndCounts_80Hz_10100Blade_UVWconnected.txt'
#filename = 'edgesAndCounts_80Hz_10100Blade_UVWdisconnected.txt'
filename = 'edgesAndCounts_80Hz_10100Blade_ticksRescaled.txt'

full_path = os.path.join(file_dir, filename)

data = np.loadtxt(full_path, delimiter=' ', usecols=[0,1,2,3], skiprows=0)

encCount = data[:,0]
shaftInputCap = data[:,1] #last input captured edge on a free-running FTM counter
numEncEdges = data[:,2] #number of captured encoder edges since last speed measurement
totalCounts = data[:,3] #number of FTM counts, rescaled by tickRescale, since last speed measurement

N_samples = len(encCount)
N_enc = 100 #number of ticks on shaft encoder
N_chop = 100 #number of apertures in chopper blade
f_FTM = 60e6 #Hz
FTM_MOD = 4096 #FTM_MOD for the FTM peripheral used to collect these data

dt = FTM_MOD/f_FTM
t = np.linspace(0, dt*N_samples, N_samples)
encoderCount = np.linspace(0, N_enc - 1, N_enc)

# Look for changes in the input captured FTM value and use that to generate deltas.
# Given: integer counts, FTM input captured CnV values @ edges, time array
# Return: 
#1. rawCount[i] = shaft encoder count at captured edge: counts[i]
#2. deltaCount[i] = change in shaft encoder count: counts[i] - counts[i-1]
#3. deltaFTM[i] = change in input capture value: CnV_i - CnV_(i-1)
#4. t1[i] = time at edge corresponding to CnV_i
def measureCountDeltas(counts, edges, time, maxCount):
    rawCount = np.zeros(0)
    deltaCount = np.zeros(0)
    deltaFTM = np.zeros(0)
    t1 = np.zeros(0)
    for i, edge in enumerate(edges[1:], start=1):
        if edge != edges[i-1]:
            rawCount = np.append(rawCount, counts[i])
            deltaCount = np.append(deltaCount, (counts[i] - counts[i-1])%maxCount)
            t1 = np.append(t1, time[i])
            deltaFTM = np.append(deltaFTM, edge - edges[i-1])
            
    return rawCount, deltaCount, deltaFTM, t1

def extractCountDeltas(FTMcount, time):
    delta = np.zeros(0)
    t1 = np.zeros(0)
    for i, count in enumerate(FTMcount[1:], start=1):
        if (count != 0) and (count != FTMcount[i-1]):                
            t1 = np.append(t1, time[i])
            if(count < FTMcount[i-1]):
                delta = np.append(delta, count)
            else:
                delta = np.append(delta, count - FTMcount[i-1])
            
    return delta, t1

# Given: a 1-D array of data
# Returns: a sliding window average where the window for the i-th average
# is centered on the i-th point (so equally forward- and backward-looking)
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

# First, calculate the delta FTM counts
rawCountAtDelta, rawCountDelta, rawFtmDelta, timeAtRawDelta = measureCountDeltas(encCount, shaftInputCap, t, N_enc)
rescaledFtmDelta, timeAtRescaledDelta = extractCountDeltas(totalCounts, t)

# Because the MeasureShaftSpeed() function periodically resets the runningTotal
# FTM count, there will be instances when the rescaledDeltaFtm is *not* the difference
# between the currentTotal and the previous total
"""
rescaledDeltaFtm = np.zeros(len(rescaledFtmDelta))
for i, currentTotal in enumerate(rescaledFtmDelta):
    if currentTotal > rescaledFtmDelta[i-1]:
        rescaledDeltaFtm[i] = currentTotal - rescaledFtmDelta[i-1]
    else:
        rescaledDeltaFtm[i] = currentTotal
"""

# This can be easily converted to delta t in seconds
rawFtmDeltaT_sec = rawFtmDelta/f_FTM
rescaledDeltaT_sec = rescaledFtmDelta/f_FTM

# Which can be converted to estimated speed as a function of time:
encSpeed = rawCountDelta/(N_enc*rawFtmDeltaT_sec)
rescaledSpeed = 1/(N_enc*rescaledDeltaT_sec)
        
# Calculate the moving average to smooth over the fine-scale variation due to 
# encoder errors (window size >= N_enc)
windowSize = int(5*N_enc/2)
avgEncSpeed = movingAverage(encSpeed, windowSize)

# Plot Speed vs Time
fig1, ax1 = plt.subplots()
ax1.plot(timeAtRawDelta[1:], encSpeed[1:], label='raw encoder speed')
ax1.plot(timeAtRawDelta[1:], avgEncSpeed[1:], label=f'windowed average, N={windowSize}')
ax1.plot(timeAtRescaledDelta[1:], rescaledSpeed[1:], label='rescaled in firmware')
ax1.set_ylim(min(encSpeed[1:]), max(encSpeed[1:]))
ax1.set_xlabel('time (s)')
ax1.set_ylabel('speed (rev/s)')
ax1.set_title('Free Spindle Decay: speed vs. time')
ax1.legend()
