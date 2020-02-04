# -*- coding: utf-8 -*-
"""
Written by A. Berger on 1/28/2020

This program is derived from torqueCorrection.py. Instead of importing angular
position data, the program requires ordered pairs of motor.phase and 
instr.phase_PID.output. This program simply resamples instr.phase_PID.output
for use as a look-up-table for feedforward corrections of phase PID
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random

#data with angle correction, for torque calibration:
filename = r'D:\Documents\Projects\SR544\Data\phasePIDoutout_vs_motorPhase.csv'

data = np.loadtxt(filename, delimiter=',', usecols=[0,1], skiprows=1)
                    
N = len(data[4972:9969,0])
rawAngle = 2*pi*data[4972:9969,0]/(2**32) #in rad
output = data[4972:9969,1]

def resample(x,y,numPts):
    """
    resample (x,y) such that len(x_out) = len(x)/dec
    
    uses a two-pass method to calculate the average of y at the new bin spacing,
    followed by the standard deviation of y_resamp based on the scatter 
    in the original y data
    """
    
    dx = max(x)/(numPts+1)
    
    x_resamp = linspace(0,max(x)-dx,numPts)
    
    
    y_resamp = np.zeros(numPts)
    y_stdev = np.zeros(numPts)
    
    i = 0 #iterator over x_resamp values
    for x_n in x_resamp:
        
        n = 0
        total = 0
        j = 0 #iterator over y values
        for value in x:
            if(x_n - dx/2 <= value < x_n + dx/2):
                total += y[j]
                n += 1
            j += 1
        
        y_resamp[i] = total/n
        #print(y_resamp[i])
        
        #again iterate over y values to calculate standard deviation
        sum_sq = 0
        j = 0
        for value in x:
            if(x_n - dx/2 <= value < x_n + dx/2):
                sum_sq += (y[j] - y_resamp[i])**2
            j += 1
            
        y_stdev[i] = np.sqrt(sum_sq/(n - 1))
        #print(res_stdev[i])
        
        i += 1
        
    return [x_resamp,y_resamp,y_stdev]

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    # convolved signal has longer output length
    #return y
    # crop convolved signal
    return y[int(window_len/2):-int(window_len/2)]



#downsample the smoothed data to create a look-up-table based on the 100-point encoder count
N_encoder = 100
[tickCount, output_avg, output_std] = resample(rawAngle, output, N_encoder)
meanOutput = np.mean(output_avg)
output_avg -= meanOutput

#convert current (as a float) to a frac16_t
current_Q_F16 = 0x8000*output_avg

np.savetxt(r'D:\Documents\Projects\SR544\Data\torqueCorr_LUT.txt',current_Q_F16,newline=',\r\n',fmt='%d')


plt.figure(1)
plt.plot(rawAngle, output - meanOutput, marker='o',linestyle='none')
plt.errorbar(tickCount, output_avg, yerr=output_std, color='orange', marker='.', linestyle='None')
plt.xlabel('motor angle (rad)')
plt.ylabel('output (frac of FS)')

fig, ax1 = plt.subplots()
ax1.set_xlabel('tick count')
ax1.set_ylabel('current correction (FRAC16)')
ax1.plot(current_Q_F16)

ax2 = ax1.twinx()
ax2.set_ylabel('current correction/full scale')
ax2.plot(output_avg)
