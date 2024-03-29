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
from numpy import pi

import matplotlib.pyplot as plt
import os

# custom modules
import fileWriter

plt.close('all')

# For SR1454 desktop
file_dir = os.path.abspath(r"C:\Users\aberger\Documents\Projects\SR542\SR542-firmware\tools\CalData\Accel Measurements")
filename = "torqueCal_20460001_3Hz.txt"

# For SRS laptop
#file_dir = os.path.abspath(r"C:\Users\aberger\Documents\Projects\SR542\Firmware\SR544\tools")

#filename = r"torqueCal_constQ_3Hz_10-100blade_CW.txt"
#filename = r"torqueCal_constQ_3Hz_10-100blade_CW_verify.txt"
#filename = r"torqueCal_constQ_3Hz_noBlade_CW.txt"
#filename = r"torqueCal_constQ_CW_newTickScaling.txt"

# For lock-in lab desktop
#file_dir = os.path.abspath(r"D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\tools\CalData")
#filename = "torqueCal_2046001_4Hz.txt"
#filename = "torqueCal_2046001_3Hz_postCal.txt"
#filename = "torqueCal_2046001_4Hz_torqueScale=0.txt"

full_path = os.path.join(file_dir, filename)

data = np.loadtxt(full_path, delimiter=',', usecols=[0,1,2], skiprows=0)

encCount = data[:,0]
ftmCount = data[:,1]
phaseInRad = 2*pi*data[:,2] #in rad
uwPhase = np.unwrap(phaseInRad) #in rad

N_enc = int(max(encCount)) + 1
f_FTM = 60e6
FTM_MOD = 4096
dt = FTM_MOD*128/f_FTM 
#data is being sampled at ~8.7 ms
#based on an FTM3PERIOD_S = 6.8266667e-5
#and a samping prescale factor of 128  
time_s = (ftmCount - ftmCount[0])/f_FTM                  

N = len(encCount) 
time = np.linspace(0,(N-1)*dt,N)

moment = 1.7e-5 #kg*m^2, from impulse measurements
#UWE 10-100 Blade (Stainless Steel)
#moment = 0.648*1.829e-5 #oz*in^2 from I_zz Fusion 360 model, to kg*m^2
torqueConst = 5.55e-3 #N*m/A, from Nuelectronics motor spec sheet
#torqueConst = 7.48e-3 #N*m/A, from Elinco motor spec sheet

def resample(x, y, numPts):
    """
    resample (x,y) such that len(y_resamp) = numPts
    
    uses a two-pass method to calculate the average of y at the new bin spacing,
    followed by the standard deviation of y_resamp based on the scatter 
    in the original y data
    """
    
    dx = max(x)/(numPts+1) # new bin spacing
    
    x_resamp = np.linspace(0, max(x)-dx, numPts) # new bins
    
    
    y_resamp = np.zeros(numPts)
    y_stdev = np.zeros(numPts)
    
    i = 0 #iterator over x_resamp values
    for x_n in x_resamp:
        
        n = 0       # total number of y-values in the i-th bin
        total = 0   # sum of y-values in the i-th bin
        j = 0       # iterator over y values
        for value in x: # iterate over the entire array of original x values
            if(x_n - dx/2 <= value < x_n + dx/2):
                # and if x is in this bin, the y value at this index should be
                # counted in this bin
                total += y[j]
                n += 1
            j += 1
        
        if(n != 0):
            y_resamp[i] = total/n
        else: 
            y_resamp[i] = 0
        
        #again iterate over y values to calculate standard deviation
        sum_sq = 0
        j = 0
        for value in x:
            if(x_n - dx/2 <= value < x_n + dx/2):
                sum_sq += (y[j] - y_resamp[i])**2
            j += 1
            
        if(n != 0):
            y_stdev[i] = np.sqrt(sum_sq/(n))
        else:
            y_stdev[i] = 0
        
        i += 1
        
    return [x_resamp,y_resamp,y_stdev]

def smooth(x, window_len=11, window='hanning'):
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

    t=np.linspace(-2,2,0.1)
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


# Calculate speed using num of FTM counts (ftmCount) and 
# number of Captured Edges
inputCapSpeed = np.gradient(uwPhase, time_s)
inputCapAccel = np.gradient(inputCapSpeed, time_s)



fig1, ax1 = plt.subplots()
ax1.plot(time, inputCapSpeed)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('shaft speed (rad/s)')
ax1.set_title('Shaft speed vs time')

fig2, ax2 = plt.subplots()
ax2.plot(time, inputCapAccel)
ax2.set_xlabel('time (s)')
ax2.set_ylabel('accel rad/s^2')
ax2.set_title('Shaft accel vs time')

# throw away first and last data points
fig3, ax3 = plt.subplots()
ax3.plot(phaseInRad[2:-2], inputCapAccel[2:-2], marker='.', linestyle='none')
ax3.set_xlabel('rotor angle (rad)')
ax3.set_ylabel('accel (rad/s^2)')
ax3.set_title('Shaft accel vs rotor angle')


# the first and last several points deviate from the majority behavior. Throw them away
#discardPts = 30
#plt.plot(rawAngle[discardPts:-discardPts],alpha_smth[discardPts:-discardPts],marker='.',linestyle='none')


# Downsample the acceleration data to create an n_sample look-up-table
n_sample = 400
[phase_resampled, alpha_avg, alpha_std] = resample(phaseInRad[2:-2], inputCapAccel[2:-2], n_sample)

# calculate smoothed data
alpha_smth = smooth(alpha_avg, window_len=11, window='hanning')

# Use spline fit to smooth data
from scipy.interpolate import splev, splrep

nz = np.nonzero(alpha_avg)
spl = splrep(phase_resampled[nz], alpha_avg[nz], s=35)
#smooth factor of 30 gave empirically acceptable results
#only use nonzero values of alpha_spd_avg to evaluate spline fit
angle = np.linspace(0, 2*np.pi, 400)
alpha_smth = splev(angle, spl)


fig4, ax4 = plt.subplots()
#ax4.errorbar(tickCount_ang, alpha_ang_avg, yerr=alpha_ang_std, marker='.', capsize=3, linestyle='none', label='from motor.phase')
ax4.errorbar(phase_resampled[nz], alpha_avg[nz], yerr=alpha_std[nz], marker='.', capsize=3, linestyle='none', label=r'from $\Delta_{FTM}$', zorder=0)
#ax4.plot(tickCount_ang, alpha_smth, label='smoothed, from motor.phase', color='#d62728')
ax4.plot(angle, alpha_smth, label='spline fit')
ax4.set_xlabel('rotor angle (rad)')
ax4.set_ylabel('accel (rad/s^2)')
ax4.legend()
ax4.set_title(f'Shaft accel vs rotor angle, down-sampled to {n_sample} elements')

def rms(x):  
    return np.sqrt(np.mean(x**2))

#convert angular acceleration (alpha) to torque, and then current
torque = moment*alpha_smth
current_corr = torque/torqueConst #in Amps

#currently, the output current is scaled such that full-scale = 1.65 A
current_corr = np.float32(current_corr/1.65) #as a float

#convert current (as a float) to a frac16_t
#current_Q_F16 = 0x8000*current_corr

# Save data as a .c file for incorporation into firmware:
#fileWriter.saveDataWithHeader(os.path.basename(__file__), filename, current_corr, 'float', 'e', 'currentQcomp')
# Save data as a simple csv for uploading via serial communications:
#np.savetxt(os.path.join(file_dir, 'currCorr_2.csv'), current_corr, newline='\n', fmt='%.6e', delimiter=',')