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
#filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_80Hz_10100Blade_UVWdisconnected.txt'
#filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_80Hz_10100Blade_innerTrackCal.txt'

# 400 count encoder
#filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_35Hz_10100Blade_400CountEnc_innerTrackCal.txt'
filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_35Hz_HeavyBlades_400CountEnc_innerTrackCal.txt'

data = np.loadtxt(filename, delimiter=' ', usecols=[0,1,2,3], skiprows=0)

encCount = data[:,0]
encEdge = data[:,1]
chopCount = data[:,2]
chopEdge = data[:,3]

# I used an 8-bit counter for chopCount. And np.unwrap works on values in radians
chopCount_rad = chopCount/256*2*pi 
uwChopCount = (np.unwrap(chopCount_rad)*256/(2*pi)).astype(int)

N_samples = len(encCount)
N_enc = 400 #number of ticks on shaft encoder
N_chop = 10 #number of apertures in chopper blade
f_FTM = 60e6 #Hz
FTM_MOD = 4096 #FTM_MOD for the FTM peripheral used to collect these data

dt = FTM_MOD/f_FTM
t = np.linspace(0, dt*N_samples, N_samples)
encoderCount = np.linspace(0, N_enc - 1, N_enc)


# TODO: finish building out this class to make the code more object-oriented
class RotaryEncoder():
    def __init__(self, N_ticks):
        self.N_ticks = N_ticks
        self.tickArray = np.linspace(0, self.N_ticks - 1, self.N_ticks)
        self.avgTickSpacing = np.zeros(len(N_ticks))
        self.tickCorrection = np.zeros(len(N_ticks))

# Given: integer counts, FTM input captured CnV values @ edges, time array
# Return: 
#1. integer count at edge corresponding to CnV_i, 
#2. delta = CnV_i - CnV_(i-1)
#3. t1 = time at edge corresponding to CnV_i
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

def findIndexOfNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def findWrapArounds(array):
    wrapIndex = np.zeros(0, dtype='int')
    for i, val in enumerate(array):
        if val - array[i-1] < 0:
            wrapIndex = np.append(wrapIndex, i)
            
    return wrapIndex

# First, calculate the delta FTM counts
encCountAtDelta, encCountDelta, encFtmDelta, encTimeAtDelta = measureCountDeltas(encCount, encEdge, t, N_enc)
chopCountAtDelta, chopCountDelta, chopFtmDelta, chopTimeAtDelta = measureCountDeltas(uwChopCount, chopEdge, t, N_chop)

# This can be easily converted to delta t in seconds
encFtmDeltaT_sec = encFtmDelta/f_FTM
chopFtmDeltaT_sec = chopFtmDelta/f_FTM

# Which can be converted to estimated speed as a function of time:
encSpeed = encCountDelta/(N_enc*encFtmDeltaT_sec)
chopSpeed = chopCountDelta/(N_chop*chopFtmDeltaT_sec)
        
# Calculate the moving average to smooth over the fine-scale variation due to 
# encoder errors (window size >= N_enc)
windowSize = int(5*N_enc/2)
avgEncSpeed = movingAverage(encSpeed, windowSize)

# Plot Speed vs Time
fig1, ax1 = plt.subplots()
ax1.plot(encTimeAtDelta[1:], encSpeed[1:])
ax1.plot(encTimeAtDelta[1:], avgEncSpeed[1:])
#ax1.plot(chopTimeAtDelta[1:], chopSpeed[1:], marker='.')
ax1.set_ylim(min(encSpeed[1:]), max(encSpeed[1:]))
ax1.set_xlabel('time (s)')
ax1.set_ylabel('speed (rev/s)')
ax1.set_title('Free Spindle Decay: speed vs. time')
ax1.legend(('shaft encoder', 'windowed average, N='+str(windowSize)))
#ax1.legend(('shaft encoder', 'windowed average, N='+str(windowSize), 'chop track, N='+str(N_chop)))

# The main step of the calibration is to convert the delta t measurements
# to tick spacing (in revs). This requires a "perfect" estimator of the instanteous
# shaft speed 
encTickSpacing = avgEncSpeed*encFtmDeltaT_sec

# To do the same for the chopper blade requires using the avgEncSpeed *at the 
# same instant in time* as when a chopFtmDeltaT_sec was measured
chopTickSpacing = np.zeros(len(chopFtmDeltaT_sec))
for i, t in enumerate(chopTimeAtDelta):
    chopTickSpacing[i] = avgEncSpeed[findIndexOfNearest(encTimeAtDelta, t)]*chopFtmDeltaT_sec[i]

# create an array against which to plot the chopTickSpacing
chopTrackCount = np.linspace(0, len(chopTickSpacing)-1, len(chopTickSpacing), dtype=int)
# chopCountAtDelta is somewhat unreliable due to non-atomic reads during 
# firmware execution. chopTrackCount provides a uniform counter that is reliably
# incremented for each new measured chopTickSpacing
chopTrackCount += int(chopCountAtDelta[0])

# Find the average and stdev of tick spacing over N_revsToAvg revolutions
def CalculateAvgTickSpacing(N_ticks, N_revsToAvg, N_revsToWait, tickSpacing_revs, rawCountAtDelta):
    measTickSpacing = np.zeros((N_ticks, N_revsToAvg))
    indexOfZerothTick = np.where(rawCountAtDelta == 0)
    i = 0
    for index in indexOfZerothTick[0]:
        if i >= N_revsToAvg:
            break
        if index > N_revsToWait*N_ticks:
            #distFromZerothTick[k, i]:
            measTickSpacing[:,i] = (tickSpacing_revs[index:index+N_ticks])
            i += 1
    avgTickSpacing = np.mean(measTickSpacing, axis=1)
    stdTickSpacing = np.std(measTickSpacing, axis=1)
    
    return (avgTickSpacing, stdTickSpacing)

(avgTickSpacing, stdTickSpacing) = CalculateAvgTickSpacing(N_enc, 10, 5, encTickSpacing, encCountAtDelta)
# find the chop count when the rotor angle == 0
chopCountZero = uwChopCount[findWrapArounds(encCount)][0]
(avgChopSpacing, stdChopSpacing) = CalculateAvgTickSpacing(N_chop, 10, 10, chopTickSpacing, (chopTrackCount-chopCountZero)%N_chop)

fig2, ax2 = plt.subplots()
ax2.errorbar(encoderCount, avgTickSpacing, yerr=stdTickSpacing, marker='.', capsize=4.0, label='average')
ax2.plot(encoderCount, 1/N_enc*np.ones(N_enc), '--', label='ideal')
ax2.set_xlabel('encoder count')
ax2.set_ylabel(r'$\Delta \theta$ (revs)')
ax2.legend()
ax2.set_title('Tick Spacing, '+r'$\Delta \theta_i = \bar{f}_i*\Delta t_i$')
fig2.tight_layout()

tickSpacingRescale = 1/(N_enc*avgTickSpacing)
#fileWriter.saveDataWithHeader(os.path.basename(__file__), filename, tickSpacingRescale, 'float', '1.7f', 'tickRescale')

# For N_revsToAvg worth of data, calculate the cumulative distance from tick 0 to tick k
def ConvertSpacingToCorrections(N_ticks, N_revsToAvg, N_revsToWait, tickSpacing_revs, rawCountAtDelta):
    distFromZerothTick_revs = np.zeros((N_ticks, N_revsToAvg))
    indexOfZerothTick = np.where(rawCountAtDelta == 0)
    i = 0
    for index in indexOfZerothTick[0]:
        if i >= N_revsToAvg:
            break
        if index > N_revsToWait*N_ticks:
            #distFromZerothTick[k, i]:
            distFromZerothTick_revs[:,i] = np.cumsum(tickSpacing_revs[index:index+N_ticks]) - 1/N_ticks
            i += 1
            
    # The tick correction is then simply the difference between the expected tick position
    # (encoderCount/N_enc) and the measured distFromZerothTick
    tickCorrection_revs = np.zeros((N_ticks, N_revsToAvg))
    encoderCount = np.linspace(0, N_ticks - 1, N_ticks)
    for i, col in enumerate(distFromZerothTick_revs.T):
        tickCorrection_revs[:,i] = encoderCount/N_ticks - col
        
    # Calculate the average and standard deviation of the tick corrections to check
    # for reproducibility
    avgTickCorrection = np.mean(tickCorrection_revs, axis=1)
    stdTickCorrection = np.std(tickCorrection_revs, axis=1)
    
    return (avgTickCorrection, stdTickCorrection)

(avgTickCorrection, stdTickCorrection) = ConvertSpacingToCorrections(N_enc, 10, 3, encTickSpacing, encCountAtDelta)
(avgChopCorrection, stdChopCorrection) = ConvertSpacingToCorrections(N_chop, 10, 10, chopTickSpacing, (chopTrackCount-chopCountZero)%N_chop)
            
# Assume that the average tick correction over one cycle is zero (otherwise,
# there will be some angle-indepedent offset imparted by the tick correction)
offsetTickCorrection = np.mean(avgTickCorrection)
offsetChopCorrection = np.mean(avgChopCorrection)

avgTickCorrection -= offsetTickCorrection
avgChopCorrection -= offsetChopCorrection

fig3, ax3 = plt.subplots()
ax3.errorbar(encoderCount/N_enc*360, avgTickCorrection, yerr=stdTickCorrection, marker='.', capsize=4.0)
ax3.errorbar(np.linspace(0, N_chop-1, N_chop)/N_chop*360, avgChopCorrection, yerr=stdChopCorrection, marker='.', capsize=4.0)
ax3.set_xlabel('rotor angle (deg)')
ax3.set_ylabel('tick error (mech. revs)')
ax3.set_title('Tick error, '+r'$\langle \theta_i \rangle - \theta_i = \frac{i}{N_{enc}} - \sum_{k=0}^i \Delta \theta_k$', y = 1.03)
ax3.legend(('shaft encoder', 'chop track, N='+str(N_chop)))
fig3.tight_layout()

"""
Test the correction -----------------------------------------------------------
"""
# Use the average tick spacing to re-scale the speed measurements, 
# where the scaling factor is measTickSpacing/perfectTickSpacing
corrSpeed = encSpeed*(avgTickSpacing[encCountAtDelta.astype(int)]/(1/N_enc))
corrChopSpeed = chopSpeed*(avgChopSpacing[(chopTrackCount-chopCountZero)%N_chop]/(1/N_chop))
ax1.plot(encTimeAtDelta[1:], corrSpeed[1:])
#ax1.plot(chopTimeAtDelta[1:], corrChopSpeed[1:], marker='.')
ax1.legend(('shaft encoder', 'windowed average, N='+str(windowSize), 'corrected encoder speed'))

fig4, ax4 = plt.subplots()
ax4.plot(encTimeAtDelta[1:], avgEncSpeed[1:] - encSpeed[1:])
ax4.plot(encTimeAtDelta[1:], avgEncSpeed[1:] - corrSpeed[1:])
ax4.legend(('original','corrected'))
ax4.set_xlabel('time (s)')
ax4.set_ylabel('speed error (revs/s)')
ax4.set_title('Speed error comparison')
fig4.tight_layout()

angleCorr_int32 = avgTickCorrection*2**32
#fileWriter.saveDataWithHeader(os.path.basename(__file__), filename, angleCorr_int32.astype(int), 'int32_t', 0, 'angleComp')