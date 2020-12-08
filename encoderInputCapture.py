# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:20:07 2020

@author: aberger

This program imports three columns of data:
    1. enc raw count (0-399)
    2. input capture of shaft edges on "free running" 60 MHz FTM counter
    3. rotor phase as a floating point value [0,1). If uncalibrated, these
    are just 1/N_enc * i
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# custom modules
import fileWriter

plt.close('all')

# For SRS laptop
#file_dir = os.path.abspath(r"C:\Users\aberger\Documents\Projects\SR542\Firmware\SR544\tools")

# For lock-in lab desktop
file_dir = os.path.abspath(r"D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\CalData")

#filename = "edgesAndCounts_35Hz_10-100blade_400CountShaftCal_CW_newTickScaling_6.txt" #CW rotation, new tick scaling
#filename = "encoderCal_20460001_35Hz.txt"
#filename = "encoderCal_20460001_35Hz_postCal.txt"
filename = "encoderCal_20460001_35Hz_heavyBlades2.txt"
#filename = "encoderCal_20460001_35Hz_vertMount.txt"

postcal = False

full_path = os.path.join(file_dir, filename)

data = np.loadtxt(full_path, delimiter=',', usecols=[0,1,2], skiprows=0)

encCount = data[:,0]
ftmCount = data[:,1]
phaseInRevs = data[:,2]

N_samples = len(encCount)
N_enc = int(max(encCount)) + 1
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

# Look for changes in the input captured FTM value and use that to generate deltas.
# Given: integer counts, FTM input captured CnV values @ edges, time array
# Return: 
#1. rawCount[i] = shaft encoder count at captured edge: counts[i]
#2. deltaCount[i] = change in shaft encoder count: counts[i] - counts[i-1]
#3. deltaFTM[i] = change in input capture value: CnV_i - CnV_(i-1)
#4. t1[i] = time at edge corresponding to CnV_i
def extractDeltas(encCount, ftmCount, phase, time, maxCount):
    rawCount = np.zeros(0)
    dCount = np.zeros(0)
    dFTM = np.zeros(0)
    dPhase = np.zeros(0)
    t1 = np.zeros(0)
    for i, cnt in enumerate(encCount[1:], start=1):
        if cnt != encCount[i-1]:
            rawCount = np.append(rawCount, encCount[i])
            dCount = np.append(dCount, (encCount[i] - encCount[i-1])%maxCount)
            dFTM = np.append(dFTM, ftmCount[i]- ftmCount[i-1])
            dPhase = np.append(dPhase, (phase[i] - phase[i-1])%1)
            t1 = np.append(t1, time[i])
            
            
    return rawCount, dCount, dFTM, dPhase, t1

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
encCountAtDelta, encCountDelta, encFtmDelta, revsDelta, encTimeAtDelta = extractDeltas(encCount, ftmCount, phaseInRevs, t, N_enc)
numRevsCaptured = len(np.where(encCountAtDelta == 0)[0])

# This can be easily converted to delta t in seconds
encFtmDeltaT_sec = encFtmDelta/f_FTM

# Which can be converted to estimated speed as a function of time:
encSpeed = encCountDelta/(N_enc*encFtmDeltaT_sec)
calSpeed = revsDelta/encFtmDeltaT_sec
        
# Calculate the moving average to smooth over the fine-scale variation due to 
# encoder errors (window size >= N_enc)
windowSize = int(5*N_enc/2)
avgEncSpeed = movingAverage(encSpeed, windowSize)

# Plot Speed vs Time
fig1, ax1 = plt.subplots()
ax1.plot(encTimeAtDelta[1:], encSpeed[1:], label='from raw encoder count')
ax1.plot(encTimeAtDelta[1:], avgEncSpeed[1:], label=f'windowed average, N={windowSize}')
if(postcal):
    ax1.plot(encTimeAtDelta[1:], calSpeed[1:], label="from cal'd phase LUT")
ax1.set_ylim(min(encSpeed[1:]), max(encSpeed[1:]))
ax1.set_xlabel('time (s)')
ax1.set_ylabel('speed (rev/s)')
ax1.set_title('Free Spindle Decay: speed vs. time')

# The main step of the calibration is to convert the delta t measurements
# to tick spacing (in revs). This requires a "perfect" estimator of the instanteous
# shaft speed (for which I use the moving average avgEncSpeed)

# tick spacing calculated *without* circular closure constraint
encTickSpacing = avgEncSpeed*encFtmDeltaT_sec

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

(avgTickSpacing_NC, stdTickSpacing_NC) = CalculateAvgTickSpacing(N_enc, 10, 12, encTickSpacing, encCountAtDelta)

# Use Least Squares with Circular Closure to determine tick spacing
# 1. Want to solve A*x = b, subject to least squares such that we minimize ||b - A*x||
# 2. In our case, A = I*(1/omega), with a final row of ones, where omega is the estimate of "true" speed
# 3. x = encoder tick spacing, which we are solving for
# 4. b = measured Delta T's, with a final element of one
# 5. The augmented elements of A and b enforce circular closure such that:
# sum_i x_i = 1 (the sum of all tick spacings over one revolution should equal one revolution)
def LeastSquaresTickSpacing(N_ticks, N_revsToAvg, N_revsToWait, startCount, rawCountAtDelta, speed, deltaT):
    measTickSpacing = np.zeros((N_ticks, N_revsToAvg))
    indexOfZerothTick = np.where(rawCountAtDelta == startCount)
    i = 0
    for index in indexOfZerothTick[0]:
        if i >= N_revsToAvg:
            break
        if index > N_revsToWait*N_ticks:
            A = np.identity(N_ticks)*1/speed[index:index+N_ticks]
            A = np.append(A, np.ones((1, N_ticks)), axis=0)
            b = deltaT[index:index+N_ticks]
            b = np.append(b, 1)
            lstsqsol = np.linalg.lstsq(A, b)
            measTickSpacing[:,i] = lstsqsol[0]
            i += 1
            
    avgTickSpacing = np.mean(measTickSpacing, axis=1)
    stdTickSpacing = np.std(measTickSpacing, axis=1)
    
    return (np.roll(avgTickSpacing, startCount), stdTickSpacing)

# The LeastSquaresTickSpacing treats circular closure only as another data point to be fitted,
# rather than as a firm constraint.    
# Instead, use scipy.optimize.minimize to enforce the constraint that the sum of tick spacings = 1 revolution
# See above description of the linear algebraic equation we are solving: A*x = b
def targetFun(x, A, b):
    return np.sum((b - A*x)**2)

# Define a constraint that requires the sum(x) for all x to add up to 1
def constraint(x):
    return np.sum(x) - 1

cons = [{'type': 'eq', 'fun': constraint}]

def ConstrainedTickSpacing(N_ticks, N_revsToAvg, N_revsToWait, startCount, rawCountAtDelta, speed, deltaT):
    """ 
    Also a least-squares minimization, but utilizes constraint to enforce
    circular closure, instead of using circular closure as a data point
    to-be-fitted
    """
    measTickSpacing = np.zeros((N_ticks, N_revsToAvg))
    indexOfZerothTick = np.where(rawCountAtDelta == startCount)
    i = 0
    for index in indexOfZerothTick[0]:
        if i >= N_revsToAvg:
            break
        if index > N_revsToWait*N_ticks:
            A = np.identity(N_ticks)*1/speed[index:index+N_ticks]
            b = deltaT[index:index+N_ticks]
            sol = minimize(targetFun, x0 = avgTickSpacing_NC, args = (A, b), method='SLSQP', tol=1e-12, constraints=cons)
            measTickSpacing[:,i] = sol['x']
            i += 1
            
    avgTickSpacing = np.mean(measTickSpacing, axis=1)
    stdTickSpacing = np.std(measTickSpacing, axis=1)
    
    return (avgTickSpacing, stdTickSpacing)

# Choose between LeastSquaresTickSpacing and ConstrainedTickSpacing
#(lsAvgTickSpacing, lsStdTickSpacing) = LeastSquaresTickSpacing(N_enc, 10, 3, 0, encCountAtDelta, avgEncSpeed, encFtmDeltaT_sec)
#(lsAvgTickSpacing_startMid, lsStdTickSpacing_startMid) = LeastSquaresTickSpacing(N_enc, 10, 3, int(N_enc/2), encCountAtDelta, avgEncSpeed, encFtmDeltaT_sec)
(lsAvgTickSpacing, lsStdTickSpacing) = ConstrainedTickSpacing(N_enc, 10, 12, 0, encCountAtDelta, avgEncSpeed, encFtmDeltaT_sec)
(lsAvgTickSpacing_startMid, lsStdTickSpacing_startMid) = ConstrainedTickSpacing(N_enc, 10, 12, int(N_enc/2), encCountAtDelta, avgEncSpeed, encFtmDeltaT_sec)
# make sure that circular closure was properly enforced
print(f'Sum of the extracted tick spacings over 1 rev is: {np.sum(lsAvgTickSpacing)} revs')

fig2, ax2 = plt.subplots()
ax2.errorbar(encoderCount, avgTickSpacing_NC, yerr=stdTickSpacing_NC, marker='.', capsize=4.0, label='no circular closure', zorder=0)
ax2.errorbar(encoderCount, lsAvgTickSpacing, yerr=lsStdTickSpacing, marker='.', capsize=4.0, label='circular closure, start = 0', zorder=0)
ax2.plot(encoderCount, 1/N_enc*np.ones(N_enc), '--', label='ideal', zorder=1)
ax2.set_xlabel('encoder count')
ax2.set_ylabel(r'$\Delta \theta$ (revs)')
ax2.legend()
ax2.set_title('Tick Spacing, '+r'$\Delta \theta_i = \bar{f}_i*\Delta t_i$')
fig2.tight_layout()

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

def ConvertLSSpacingToCorrections(N_ticks, lsTickSpacing_revs):
    distFromZerothTick = np.cumsum(lsTickSpacing_revs) - lsTickSpacing_revs[0]
    
    encoderCount = np.linspace(0, N_ticks - 1, N_ticks)
    tickCorrection_revs = encoderCount/N_ticks - distFromZerothTick
    
    return tickCorrection_revs

# Tick Corrections *without* circular closure
(avgTickCorrection, stdTickCorrection) = ConvertSpacingToCorrections(N_enc, 10, 3, encTickSpacing, encCountAtDelta)

# Assume that the average tick correction over one cycle is zero (otherwise,
# there will be some angle-indepedent offset imparted by the tick correction)
offsetTickCorrection = np.mean(avgTickCorrection)
avgTickCorrection -= offsetTickCorrection

# Tick Corrections *with* circular closure
lsTickCorrection = ConvertLSSpacingToCorrections(N_enc, lsAvgTickSpacing)
lsTickCorrection_startMid = ConvertLSSpacingToCorrections(N_enc, lsAvgTickSpacing_startMid)
#lsOffset = np.mean(lsTickCorrection)
#lsTickCorrection -= lsOffset

calibratedTickPositions = (np.cumsum(lsAvgTickSpacing) - lsAvgTickSpacing[0]).astype('float32')
# Save data as a .c file for incorporation into firmware:
#fileWriter.saveDataWithHeader(os.path.basename(__file__), filename, calibratedTickPositions, 'float', 'e', f'tickRescale{N_enc}')
# Save data as a simple csv for uploading via serial communications:
#np.savetxt(os.path.join(file_dir, 'tickPos.csv'), calibratedTickPositions, newline='\n', fmt='%.6e', delimiter=',')
tickError = encoderCount/N_enc - calibratedTickPositions

fig3, ax3 = plt.subplots()
#ax3.plot(encoderCount/N_enc*360, lsTickCorrection, marker='.', label = 'circular closure, start = 0')
#ax3.plot(encoderCount/N_enc*360, lsTickCorrection_startMid - np.mean(lsTickCorrection_startMid), label = f'circular closure, start = {int(N_enc/2)}')
#ax3.errorbar(encoderCount/N_enc*360, avgTickCorrection, yerr=stdTickCorrection, marker='.', capsize=4.0, label='no circular closure')
ax3.plot(encoderCount, tickError, marker='.')
ax3.set_xlabel('rotor angle (deg)')
ax3.set_ylabel('tick error (mech. revs)')
ax3.set_title('Tick error, '+r'$\langle \theta_i \rangle - \theta_i = \frac{i}{N_{enc}} - \sum_{k=0}^i \Delta \theta_k$', y = 1.03)
fig3.tight_layout()


#------------------------------------------------------------------------------
# Test the correction ---------------------------------------------------------
#------------------------------------------------------------------------------

# Use the average tick spacing to re-scale the speed measurements, 
# where the scaling factor is measTickSpacing/perfectTickSpacing
tickSpacingRescale = np.float32(N_enc*lsAvgTickSpacing)
corrSpeed = encSpeed*tickSpacingRescale[encCountAtDelta.astype(int)]

# Or applying the LUT directly instead of by applying a scale factor:    
calibratedDeltaPhase = (calibratedTickPositions[encCountAtDelta.astype(int)] - calibratedTickPositions[(encCountAtDelta.astype(int)-1)%400])%1
corrSpeed = calibratedDeltaPhase/encFtmDeltaT_sec
ax1.plot(encTimeAtDelta[1:], corrSpeed[1:], label='corrected encoder speed')
ax1.legend()

fig4, ax4 = plt.subplots()
numPtsToIgnore = int(windowSize/2)
uncaldSpeedError = avgEncSpeed[numPtsToIgnore:-numPtsToIgnore] - encSpeed[numPtsToIgnore:-numPtsToIgnore]
caldSpeedError = avgEncSpeed[numPtsToIgnore:-numPtsToIgnore] - calSpeed[numPtsToIgnore:-numPtsToIgnore]
newCaldSpeedError = avgEncSpeed[numPtsToIgnore:-numPtsToIgnore] - corrSpeed[numPtsToIgnore:-numPtsToIgnore:]

ax4.plot(encTimeAtDelta[numPtsToIgnore:-numPtsToIgnore], uncaldSpeedError, label='from raw encoder count')
if(postcal):
    ax4.plot(encTimeAtDelta[numPtsToIgnore:-numPtsToIgnore], caldSpeedError, label="from firwmare phase LUT")
ax4.plot(encTimeAtDelta[numPtsToIgnore:-numPtsToIgnore], newCaldSpeedError, label="from new cal'd tick positions")
ax4.legend()
ax4.set_xlabel('time (s)')
ax4.set_ylabel('speed error (revs/s)')
ax4.set_title('Speed error comparison')
fig4.tight_layout()

def rms(x):  
    return np.sqrt(np.mean(x**2))

print(f'Improvement in rms speed error = {rms(uncaldSpeedError)/rms(caldSpeedError):.2f}x')
# TODO: what is acceptable to "pass" the cal?