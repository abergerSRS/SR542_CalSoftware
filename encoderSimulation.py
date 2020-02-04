# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:05:16 2020

@author: aberger

This program attempts to simulate optical encoder error
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

class GenericEncoder(object):
    def getCountsPerRevolution(self):
        return len(self.points)
    
    # Given: an array of actual positions (in revs)
    # Return: an ordered pair of encoder position count, and number of revolutions
    def measureRaw(self, actualPos):
        positionInRevolution = np.array(actualPos) % 1.0
        numRevolutions = np.round(actualPos - positionInRevolution).astype(int)
        posCount = np.searchsorted(self.points, positionInRevolution)
        return (posCount, numRevolutions)
    
    # Given: an array of actual positions (in revs)
    # Return: the cumulative position count
    def measure(self, actualPos):
        (posCount, numRevolutions) = self.measureRaw(actualPos)
        return posCount + numRevolutions*self.getCountsPerRevolution()
    
    def getTickPositions(self):
        return self.points
    
    # Given: an array of actual positions (in revs)
    # Return: locations (in time) of changes in encoder count (for use with 
    # simulated FTM input cpature of edges)
    def findEdges(self, actualPos):
        (posCount, numRevolutions) = self.measureRaw(actualPos)
        edges = np.zeros(0, dtype=int)
        
        #first handle the zero-th element
        if posCount[0] != posCount[-1]:
            edges = np.append(edges, 0)
            
        for i, count in enumerate(posCount[1:], start=1):
            if posCount[i] != posCount[i-1]:
                edges = np.append(edges, i)
                
        return edges
    
class PerfectEncoder(GenericEncoder):
    def __init__(self, N):
        self.points = np.arange(0, 1, 1.0/N)
        
class USDigitalE2Encoder(GenericEncoder):
    def __init__(self):
        angleError_degs = np.array([
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
    
        self.points = np.arange(0, 1, 1.0/len(angleError_degs)) 
        self.points += angleError_degs/(360*len(angleError_degs))
        
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
        counts = self.getCount(time_s)
        deltaCount = np.zeros(0, dtype = int)
        
        for i in range(len(counts[0:-1])):
            deltaCount = np.append(deltaCount, counts[i+1] - counts[i])
            
        return deltaCount
        
# Start the procedure here
omega_0 = 85 #Hz
theta_0 = 0 #revs

dt = 1/120e6 #seconds (twice as fast as CPU sampling rate, just to ensure no artifacts)
numRevs = 3
t_final = numRevs/omega_0
numPoints = int(t_final/dt)
t = np.linspace(0, t_final, numPoints)

gamma = .073 #damping parameter, in Hz/s
omega = omega_0*np.exp(-gamma*t)

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
perfectEncoder = PerfectEncoder(100)

actualPosInCounts = theta_actual*shaftEncoder.getCountsPerRevolution()
measuredPosInCounts = shaftEncoder.measure(theta_actual)
measuredEdgeIndices = shaftEncoder.findEdges(theta_actual)
measuredCountDeltas = ftm.getCountDeltas(t[measuredEdgeIndices])

"""
plt.figure(2)
plt.plot(t, actualPositionInCounts - measuredPositionInCounts)
plt.ylabel('actual - shaft encoder (counts)')
plt.xlabel('time (s)')
"""

perfectPosInCounts = perfectEncoder.measure(theta_actual)
perfectEdgeIndices = perfectEncoder.findEdges(theta_actual)
perfectCountDeltas = ftm.getCountDeltas(t[perfectEdgeIndices])
