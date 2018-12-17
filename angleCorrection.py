# -*- coding: utf-8 -*-
"""
Written by A. Berger on 12/17/2018


"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random

filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\angleRecord_8Y16_A_22.txt'

data = np.loadtxt(filename, delimiter=' ', usecols=[0,1], skiprows=0)

dt = 0.004369066688 #assumes data is being sampled at ~4 ms
                    #based on an FTM3PERIOD_S = 6.8266667e-5
                    #and a PID_PRESCALE = 64
                    
N = len(data[1:,0]) #discarding first point
time = linspace(0,(N-1)*dt,N)
rawAngle = 2*pi*data[1:,1]/(2**32) #in 

   
uwAngle = np.unwrap(rawAngle) #in rad

p = np.polyfit(time,uwAngle,20)
    
resAngle = uwAngle - np.polyval(p,time)
    
plt.figure(1)
plt.plot(rawAngle,resAngle,marker='o',linestyle='none')

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
            if(x_n <= value < x_n + dx):
                total += y[j]
                n += 1
            j += 1
            
        y_resamp[i] = total/n
        #print(res_avg[i])
        
        #again iterate over y values to calculate standard deviation
        sum_sq = 0
        j = 0
        for value in x:
            if(x_n <= value < x_n + dx):
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

# the first and last several points deviate from the majority behavior. Throw them away
discardPts = 5
#plt.plot(rawAngle[discardPts:-discardPts],alpha_smth[discardPts:-discardPts],marker='.',linestyle='none')

#downsample the smoothed data to create a look-up-table based on the 100-point encoder count
N_encoder = 100
[tickCount,resAngle_avg,resAngle_std] = resample(rawAngle[discardPts:-discardPts],resAngle[discardPts:-discardPts],N_encoder)

plt.errorbar(tickCount,resAngle_avg, yerr=resAngle_std,color='orange', marker='.',linestyle='None')

#np.savetxt(r'D:\Documents\Projects\SR544\Data\torqueCorr_LUT.txt',current_Q_F16,newline=',\r\n',fmt='%d')
