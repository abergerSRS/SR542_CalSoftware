# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:07:25 2018

@author: aberger
"""

import numpy as np

def frac16(floatIn):
    if floatIn < 0.999969482421875:
        if floatIn >= -1:
            out = floatIn*0x8000
        else:
            out = 0x8000
    else:
        out = 0x7FFF
        
    return out

filename = r'D:\Documents\Projects\SR544\Data\8Y16_A_currentQ_corr_withAngleCorr.csv'
floatCorr = np.loadtxt(filename, delimiter=',', usecols=[0], skiprows=1)
frac16Corr = np.zeros(len(floatCorr),dtype='int16')

i = 0
for element in floatCorr:
    frac16Corr[i] = frac16(65*element)
    i += 1
    
np.savetxt(r'D:\Documents\Projects\SR544\Data\currentQCorr_LUT.txt',frac16Corr,newline=',\r\n',fmt='%u')

    