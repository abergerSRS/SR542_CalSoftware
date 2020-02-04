# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:05:16 2020

@author: aberger

This program attempts to simulate optical encoder error
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

plt.close('all')

class GenericEncoder(object):
    def getCountsPerRevolution(self):
        return len(self.points)
    
    # Given: an array of actual positions (in revs)
    # Return: an ordered pair of encoder position count, and number of revolutions
    def measureRaw(self, actualPos):
        numRevolutions = ((actualPos - self.points[0])//1).astype(int)        
        positionInRevolution = np.array(actualPos) - numRevolutions
        posCount = (np.searchsorted(self.points, positionInRevolution, side='right'))-1        
        return (posCount, numRevolutions)
    
    # Given: an array of actual positions (in revs)
    # Return: the cumulative position count
    def measure(self, actualPos):
        (posCount, numRevolutions) = self.measureRaw(actualPos)
        return posCount + numRevolutions*self.getCountsPerRevolution()
    
    def getTickPositions(self):
        return self.points
    
    # Given: an array of actual positions (in revs)
    # Return: index of locations of changes in encoder count (for use with 
    # simulated FTM input capture of edges)
    def findEdges(self, actualPos):
        (encCount, numRevolutions) = self.measureRaw(actualPos)
        edgeIndices = np.zeros(0, dtype=int)
        encCountAtEdge = np.zeros(0, dtype=int)
                    
        for i, count in enumerate(encCount[1:], start=1):
            if encCount[i] != encCount[i-1]:
                edgeIndices = np.append(edgeIndices, i)
                encCountAtEdge = np.append(encCountAtEdge, encCount[i])
                
        return (encCountAtEdge, edgeIndices)
    
class PerfectEncoder(GenericEncoder):
    def __init__(self, N):
        self.points = np.arange(0, 1, 1.0/N)
        
class USDigitalE2Encoder(GenericEncoder):
    def __init__(self):
        # angle error is generated from normal random distribution with mean
        # of 0 degs, and std deviation of ~sqrt(3)
        angleError_elecDegs = np.array([
        2.44186501,  2.5379438 , -0.27091587, -3.57686419, -4.41229792,
        0.49112343, -0.01359262, -0.9317193 ,  1.25871155, -0.02554087,
       -0.57320613, -1.54170247, -0.95057039, -0.08583924,  2.3392125 ,
       -0.57041181,  0.12843141,  0.56701995,  1.35714331, -1.33452506,
        1.1001491 ,  0.08990817, -1.1962842 ,  0.68031007,  3.6967314 ,
       -0.13271095,  0.37133305,  1.09290887,  1.79342285,  1.26576171,
        1.6303808 , -2.04181488, -1.09090484, -0.49614661, -2.15543667,
       -1.12846149, -0.02378523, -0.76464322, -1.71317722,  0.43139991,
        1.23706493,  2.4038922 ,  2.04273708,  0.24238397, -2.0975687 ,
        0.33688082, -0.10736362, -0.44575841,  0.6438165 ,  5.42320809,
       -1.21645277,  2.03419924, -0.9019025 ,  1.6062626 , -0.34027706,
        0.29959941, -0.95887284, -1.97150881,  2.7655532 , -0.01800716,
        0.53045341,  2.23368319,  0.02850127, -0.9069917 ,  1.8876487 ,
       -0.2227745 , -1.25584448, -0.2526032 , -2.29790783, -0.18987051,
       -0.13744477,  1.17742555,  1.73009864,  0.80588236,  0.53433823,
        0.80013399, -2.51117443, -1.11728092, -1.2002701 , -0.62981861,
       -3.46344264, -0.87406476, -0.14335899,  0.84020046,  0.17259158,
        0.97984206, -1.89372792, -1.68195365, -2.36321172, -0.03754612,
        2.18551477, -1.74108652, -0.20605028, -1.04449301, -0.65542215,
       -2.01900357,  2.52012054,  2.41968657,  0.96446608,  0.43393069])
    
        N_enc = len(angleError_elecDegs)
        encoderCount = np.linspace(0, N_enc-1, N_enc)
        self.points = encoderCount/N_enc
        # Add the random angle error
        self.points += angleError_elecDegs/(360*len(angleError_elecDegs))
        # Add sinusoidal run-out
        runOutMagnitude_revs = 2e-4
        self.points += runOutMagnitude_revs*np.sin((encoderCount - 27)/N_enc*2*np.pi)
        
        # Calculate spacing between adjacent ticks
        self.tickSpacing = np.zeros(len(self.points))
        
        self.tickSpacing[0] = self.points[0] + 1.0 - self.points[99]        
        for i, tick in enumerate(self.points[1:], start=1):
            self.tickSpacing[i] = self.points[i] - self.points[i-1]
        
class FTMCounter():
    def __init__(self, freq, modulus, time_array_s):
        self.freq = freq
        self.modulus = modulus
        self.time = time_array_s        
        #self.count = (time_array_s*self.freq % self.modulus).astype(int)
        self.count = (time_array_s*self.freq).astype(int)
        
    # Returns: At some point in time "time_s", what is the FTM counter value?
    # (works with array inputs too)
    def getCount(self, time_s):
        return self.count[np.searchsorted(self.time, time_s)]
    
    def getCountDeltas(self, time_s):
        count = self.getCount(time_s)
        deltaCount = np.zeros(0, dtype = int)
        
        for i in range(1, len(count)):
            deltaCount = np.append(deltaCount, count[i] - count[i-1])
        return deltaCount
        
# Start the procedure here
omega_0 = 85 #Hz
theta_0 = 0.3 #revs

dt = 1/120e6 #seconds (twice as fast as CPU sampling rate, just to ensure no artifacts)
numRevs = 3
t_final = numRevs/omega_0
numPoints = int(t_final/dt)
t = np.linspace(0, t_final, numPoints)

gamma = .073 #damping parameter, in Hz/s
#omega = omega_0*np.exp(-gamma*t)
omega = omega_0*np.ones(len(t))

theta_actual = scipy.integrate.cumtrapz(omega, t, initial = theta_0)
theta_actual[1:] += theta_0

"""
plt.subplot(2,1,1)
plt.plot(t, omega)
plt.ylabel('speed (revs/s)')

plt.subplot(2,1,2)
plt.plot(t, theta_actual % 1.0)
plt.ylabel('angle (revs)')
plt.xlabel('time (s)')
"""

ftm = FTMCounter(60e6, 4096, t)
shaftEncoder = USDigitalE2Encoder()
N_enc = shaftEncoder.getCountsPerRevolution()
perfectEncoder = PerfectEncoder(N_enc)

actualPosInCounts = theta_actual*N_enc

measuredPosInCounts = shaftEncoder.measure(theta_actual)
(measPosAtEdge, measuredEdgeIndices) = shaftEncoder.findEdges(theta_actual)
measuredCountDeltas = ftm.getCountDeltas(t[measuredEdgeIndices])
measuredDeltaT_secs = measuredCountDeltas/ftm.freq
measuredSpeed = 1/(N_enc*measuredDeltaT_secs)

perfectPosInCounts = perfectEncoder.measure(theta_actual)
(perfPosAtEdge, perfectEdgeIndices) = perfectEncoder.findEdges(theta_actual)
perfectCountDeltas = ftm.getCountDeltas(t[perfectEdgeIndices])
perfectDeltaT_secs = perfectCountDeltas/ftm.freq
perfectSpeed = 1/(N_enc*perfectDeltaT_secs)

encoderCount = np.linspace(0, N_enc-1, N_enc)
measuredTickSpacing = perfectSpeed*measuredDeltaT_secs

fig1, ax1 = plt.subplots()
ax1.plot(t, actualPosInCounts)
ax1.plot(t, measuredPosInCounts)
ax1.plot(t, perfectPosInCounts)
ax1.legend(('actual pos', 'shaft encoder', 'perfect encoder'))
ax1.set_ylabel('position (encoder count)')
ax1.set_xlabel('time (s)')

fig2, ax2 = plt.subplots()
ax2.plot(t, omega)
ax2.plot(t[measuredEdgeIndices[1:]], measuredSpeed, marker='o')
ax2.plot(t[perfectEdgeIndices[1:]], perfectSpeed, marker='o')
ax2.legend(('actual pos', 'shaft encoder', 'perfect encoder'))
ax2.set_ylabel('measured speed (revs/s)')
ax2.set_xlabel('time (s)')

fig3, ax3 = plt.subplots()
ax3.plot(measPosAtEdge[1:], measuredTickSpacing, marker='o', linestyle='none')
ax3.plot(encoderCount, shaftEncoder.tickSpacing)
ax3.legend(('measured', 'actual'))
ax3.set_ylabel('tick spacing (revs)')
ax3.set_xlabel('encoder count')

"""
plt.figure(2)
plt.plot(t, actualPositionInCounts - measuredPositionInCounts)
plt.ylabel('actual - shaft encoder (counts)')
plt.xlabel('time (s)')
"""


