# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:20:01 2020

@author: aberger
"""
import datetime

def saveDataWithHeader(scriptName, dataFileName, dataArray, cDataType, width, destFilename):
    
    fileDir = 'D:\\Documents\\MCUXpressoIDE_10.1.0_589\\workspace\\SR544\\utilities\\'
    filepath = fileDir + destFilename + '.c'

    file = open(filepath, 'w')
    
    file.write('/*\n * Automatically generated with ' + scriptName + ' on ' + str(datetime.datetime.now())+'\n')
    file.write(' * using data from\n')
    file.write(' * '+dataFileName)
    file.write('\n*/\n\n')
    file.write('#include "'+destFilename+'.h"\n\n')
    file.write('const '+cDataType+' '+destFilename+'['+str(len(dataArray))+'] = {\n')
    for i, element in enumerate(dataArray):
        if i < (len(dataArray)-1):
            file.write(f'     {element:{width}},\n')
        else:            
            file.write(f'     {element:{width}}\n')
            
    file.write('};')
    
    file.close()