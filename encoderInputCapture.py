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
import os

# custom modules
import fileWriter

plt.close('all')

#filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_80Hz_10100Blade_UVWconnected.txt'
filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_80Hz_10100Blade_UVWdisconnected.txt'

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

# Given: integer counts, FTM input captured CnV values @ edges, time array
# Return: 
#1. integer count at edge corresponding to CnV_i, 
#2. delta = CnV_i - CnV_(i-1)
#3. t1 = time at edge corresponding to CnV_i
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
encCountAtDelta, encDelta, encTimeAtDelta = measureCountDeltas(encCount, encEdge, t)
chopCountAtDelta, chopDelta, chopTimeAtDelta = measureCountDeltas(chopCount, chopEdge, t)

# This can be easily converted to delta t in seconds
encDeltaT_sec = encDelta/f_FTM
chopDeltaT_sec = chopDelta/f_FTM

# Which can be converted to estimated speed as a function of time:
encSpeed = 1/(N_enc*encDeltaT_sec)
chopSpeed = 1/(N_chop*chopDeltaT_sec)
        
# Calculate the moving average to smooth over the fine-scale variation due to 
# encoder errors (window size >= N_enc)
windowSize = 250
avgEncSpeed = movingAverage(encSpeed, windowSize)
avgChopSpeed = movingAverage(chopSpeed, windowSize) #not currently using this data

# Plot Speed vs Time
fig1, ax1 = plt.subplots()
ax1.plot(encTimeAtDelta[1:], encSpeed[1:])
ax1.plot(encTimeAtDelta[1:], avgEncSpeed[1:])
ax1.set_ylim(min(encSpeed[1:]), max(encSpeed[1:]))
ax1.set_xlabel('time (s)')
ax1.set_ylabel('speed (rev/s)')
ax1.legend(('shaft encoder','windowed average, N='+str(windowSize)))
ax1.set_title('Free Spindle Decay: speed vs. time')

# The main step of the calibration is to convert the delta t measurements
# to tick spacing (in revs). This requires a "perfect" estimator of the instanteous
# shaft speed 
encTickSpacing = avgEncSpeed*encDeltaT_sec

# In order to align the tick spacing measurements to encoderCount = 0, find 
# all instances of the encoder count where == 0:
indexOfZerothTick = np.where(encCountAtDelta == 0)

# And wait at least numRevsToWait revolutions before considering data
numRevsToWait = 3
startingIndexOfFirstGoodRev = indexOfZerothTick[0][numRevsToWait]

N_revs = 10
# Find the average and stdev of tick spacing over N_revs revolutions
measTickSpacing = np.zeros((N_enc, N_revs))
i = 0
for index in indexOfZerothTick[0]:
    if i >= N_revs:
        break
    if index > numRevsToWait*N_enc:
        #distFromZerothTick[k, i]:
        measTickSpacing[:,i] = (encTickSpacing[index:index+N_enc])
        i += 1
avgTickSpacing = np.mean(measTickSpacing, axis=1)
stdTickSpacing = np.std(measTickSpacing, axis=1)

fig2, ax2 = plt.subplots()
ax2.errorbar(encoderCount, avgTickSpacing, yerr=stdTickSpacing, marker='.', capsize=4.0, label='average')
ax2.plot(encoderCount, 1/N_enc*np.ones(N_enc), '--', label='ideal')
ax2.set_xlabel('encoder count')
ax2.set_ylabel(r'$\Delta \theta$ (revs)')
ax2.legend()
ax2.set_title('Tick Spacing, '+r'$\Delta \theta_i = \bar{f}_i*\Delta t_i$')
fig2.tight_layout()

tickSpacingRescale = .01/avgTickSpacing
fileWriter.saveDataWithHeader(os.path.basename(__file__), filename, tickSpacingRescale, 'tickRescale')

# For N_revs worth of data, calculate the cumulative distance from tick 0 to tick k
distFromZerothTick = np.zeros((N_enc, N_revs))
i = 0
for index in indexOfZerothTick[0]:
    if i >= N_revs:
        break
    if index > numRevsToWait*N_enc:
        #distFromZerothTick[k, i]:
        distFromZerothTick[:,i] = np.cumsum(encTickSpacing[index:index+N_enc]) - 1/N_enc
        i += 1
        
# The tick correction is then simply the difference between the expected tick position
# (encoderCount/N_enc) and the measured distFromZerothTick
tickCorrection = np.zeros((N_enc, N_revs))
for i, col in enumerate(distFromZerothTick.T):
    tickCorrection[:,i] = encoderCount/N_enc - col

# Calculate the average and standard deviation of the tick corrections to check
# for reproducibility
avgTickCorrection = np.mean(tickCorrection, axis=1)
stdTickCorrection = np.std(tickCorrection, axis=1)

# Assume that the average tick correction over one cycle is zero (otherwise,
# there will be some angle-indepedent offset imparted by the tick correction)
offsetTickCorrection = np.mean(avgTickCorrection)

avgTickCorrection -= offsetTickCorrection

fig3, ax3 = plt.subplots()
ax3.errorbar(encoderCount, avgTickCorrection, yerr=stdTickCorrection, marker='.', capsize=4.0)
ax3.set_xlabel('encoder count')
ax3.set_ylabel('tick correction (revs)')
ax3.set_title('Tick correction, '+r'$\langle \theta_i \rangle - \theta_i = \frac{i}{N_{enc}} - \sum_{k=0}^i \Delta \theta_k$', y = 1.03)
fig3.tight_layout()

"""
Test the correction -----------------------------------------------------------
"""
# Use the average tick spacing to re-scale the speed measurements, 
# where the scaling factor is measTickSpacing/perfectTickSpacing
corrSpeed = encSpeed*(avgTickSpacing[encCountAtDelta.astype(int)]/(1/N_enc))
ax1.plot(encTimeAtDelta[1:], corrSpeed[1:])
ax1.legend(('shaft encoder', 'windowed average, N='+str(windowSize), 'corrected speed'))

fig4, ax4 = plt.subplots()
ax4.plot(encTimeAtDelta[1:], avgEncSpeed[1:] - encSpeed[1:])
ax4.plot(encTimeAtDelta[1:], avgEncSpeed[1:] - corrSpeed[1:])
ax4.legend(('original','corrected'))
ax4.set_xlabel('time (s)')
ax4.set_ylabel('speed error (revs/s)')
ax4.set_title('Speed error comparison')
fig4.tight_layout()

angleCorr_int32 = avgTickCorrection*2**32
fileWriter.saveDataWithHeader(os.path.basename(__file__), filename, angleCorr_int32, 'angleComp')