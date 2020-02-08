# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:20:01 2020

@author: aberger
"""
import datetime

filepath = r'D:\Documents\MCUXpressoIDE_10.1.0_589\workspace\SR544\utilities\angleComp.c'

def saveDataWithHeader(script, dataFileName, dataArray):
    file = open(filepath, 'w')
    
    file.write('/*\n * Automatically generated with ' + script + ' on ' + str(datetime.datetime.now())+'\n')
    file.write(' * using data from\n')
    file.write(' * '+dataFileName)
    file.write('\n*/\n\n')
    file.write('#include "angleComp.h"\n\n')
    file.write('const uint32_t angleComp[100] = {\n')
    for i, element in enumerate(dataArray):
        if i < (len(dataArray)-1):
            file.write('    '+str(int(element))+',\n')
        else:            
            file.write('    '+str(int(element))+'\n')
            
    file.write('};')
    
    file.close()