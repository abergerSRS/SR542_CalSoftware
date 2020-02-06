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

filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\edgesAndCounts_80Hz_10100Blade_UVWconnected.txt'

data = np.loadtxt(filename, delimiter=' ', usecols=[0,1,2,3], skiprows=0)

encCount = data[:,0]
encEdge = data[:,1]
chopCount = data[:,2]
chopEdge = data[:,3]

dt = 4096/60e6
t = np.linspace(0, dt*len(encCount), len(encCount))

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

encCountAtDelta, encDelta, encTimeAtDelta = measureCountDeltas(encCount, encEdge, t)
chopCountAtDelta, chopDelta, chopTimeAtDelta = measureCountDeltas(chopCount, chopEdge, t)

encSpeed = 60e6/(100*encDelta)
chopSpeed = 60e6/(100*chopDelta)

def movingAverage(data, windowSize):
    avg = np.zeros(len(data))
    delta = int(np.floor(windowSize/2))
    
    for i in range(delta):
        avg[i] = data[i]
        
    for i, datum in enumerate(data[delta:-delta], start=delta):
        avg[i] = np.sum(data[i - delta: i + delta + 1])/(windowSize+1)
    
    return avg
        

