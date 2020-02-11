# -*- coding: utf-8 -*-
"""
Written by A. Berger on 2/11/2020

This program imports angular position and speed data acquired at nominally-constant 
shaft speed (i.e. fixed Q-axis current) for small shaft frequencies (<~3 Hz)
where the angular deviations are dominated by torque non-uniformity

The program calculates the angular acceleration using:
    1. second derivative of the motor phase
    2. first derivative of the motor speed
and converts this to an applied current that can be used to correct for the
torque non-uniformity
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random

plt.close('all')

#data with angle correction, for torque calibration:
filename = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\phaseAndSpeed_constQ_2.6Hz.txt'

data = np.loadtxt(filename, delimiter=' ', usecols=[0,1,2,3], skiprows=0)

N_enc = 100
f_FTM = 60e6
FTM_MOD = 4096
dt = FTM_MOD*64/f_FTM 
#data is being sampled at ~4 ms
#based on an FTM3PERIOD_S = 6.8266667e-5
#and a samping prescale factor of 64                    

N = len(data[:,0]) 
time = linspace(0,(N-1)*dt,N)

rotorAngle_rawCount = data[:,0]
shaftInputCapture = data[:,1]
numEdges = data[:,2]
rawPhase = 2*pi*data[:,3]/(2**32) #in rad
uwPhase = np.unwrap(rawPhase) #in rad

#moment = 1.21e-5 #kg*m^2, from I_zz Fusion 360 model of encoder disc
moment = 1.7e-5 #kg*m^2, from impulse measurements
#torqueConst = 5.55e-3 #N*m/A, from Nuelectronics motor spec sheet
torqueConst = 7.48e-3 #N*m/A, from Elinco motor spec sheet

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


# differentiate unwrapped angle twice to calculate acceleration
dAngle_dt = np.gradient(uwPhase, dt)
# calculate smoothed speed
#omega_smth = smooth(omega, window_len = 17, window = 'hanning')

d2Angle_dt2 = np.gradient(dAngle_dt, dt)

deltaT_sec = shaftInputCapture/(f_FTM*numEdges)
inputCapSpeed = 1/(N_enc*deltaT_sec)

fig1, ax1 = plt.subplots()
ax1.plot(time, dAngle_dt)
ax1.plot(time, inputCapSpeed*2*np.pi)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('shaft speed (rad/s)')
ax1.legend((r'$\theta(t)$', r'$\Delta_{FTM}(t)$'))

inputCapAccel = np.gradient(inputCapSpeed*2*np.pi, dt)

fig2, ax2 = plt.subplots()
ax2.plot(time, d2Angle_dt2)
ax2.plot(time, inputCapAccel)
ax2.set_xlabel('time (s)')
ax2.set_ylabel('accel rad/s^2')
ax2.legend((r'$\theta(t)$', r'$\Delta_{FTM}(t)$'))

fig3, ax3 = plt.subplots()
ax3.plot(rawPhase, d2Angle_dt2, marker='.', linestyle='none')
ax3.plot(rawPhase, inputCapAccel, marker='.', linestyle='none')
ax3.set_xlabel('rotor angle (rad)')
ax3.set_ylabel('accel (rad/s^2)')
ax3.legend((r'$\theta(t)$', r'$\Delta_{FTM}(t)$'))

"""
# calculate smoothed data
#alpha_smth = smooth(alpha,window_len=11,window='hanning')

# the first and last several points deviate from the majority behavior. Throw them away
discardPts = 30
#plt.plot(rawAngle[discardPts:-discardPts],alpha_smth[discardPts:-discardPts],marker='.',linestyle='none')

"""
#downsample the smoothed data to create a look-up-table based on the 100-point encoder count
[tickCount_ang, alpha_ang_avg, alpha_ang_std] = resample(rawPhase, d2Angle_dt2, N_enc)
[tickCount_spd, alpha_spd_avg, alpha_spd_std] = resample(rawPhase, inputCapAccel, N_enc)

fig4, ax4 = plt.subplots()
ax4.plot(tickCount_ang, alpha_ang_avg)
ax4.plot(tickCount_spd, alpha_spd_avg)
ax4.set_xlabel('rotor angle (rad)')
ax4.set_ylabel('accel (rad/s^2)')
ax4.legend((r'$\theta(t)$', r'$\Delta_{FTM}(t)$'))
"""
#convert angular acceleration (alpha) to torque, and then current
torque = moment*alpha_avg
current_corr = torque/torqueConst #in Amps

#currently, the output current is scaled such that full-scale = 1.65 A
current_corr = current_corr/1.65 #as a float

#convert current (as a float) to a frac16_t
current_Q_F16 = 0x8000*current_corr

np.savetxt(r'D:\Documents\Projects\SR544\Data\torqueCorr_LUT.txt',current_Q_F16,newline=',\r\n',fmt='%d')


plt.figure(1)
plt.plot(rawAngle,alpha_smth,marker='o',linestyle='none')
plt.errorbar(tickCount,alpha_avg, yerr=alpha_std,color='orange', marker='.',linestyle='None')
plt.xlabel('motor angle (rad)')
plt.ylabel('alpha (rad/s^2)')

fig, ax1 = plt.subplots()
ax1.set_xlabel('tick count')
ax1.set_ylabel('current correction (FRAC16)')
ax1.plot(current_Q_F16)

ax2 = ax1.twinx()
ax2.set_ylabel('current correction/full scale')
ax2.plot(current_Q_F16/(2**15))
"""