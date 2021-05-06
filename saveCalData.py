import datetime
import numpy as np
import os
import pandas as pd
from glob import glob

def createHeader(sn, measurementName, trial, columns, sep='\t'):    
    headerStr = f'sn: {sn}\nmeasurement: {measurementName}\ntrial: {trial}\n{sep.join(columns)}'
    return headerStr

def saveMeasurement(dest, sn, measurementName, data, columns, header=None, fmt='%.6e'):
    folder = os.path.join(dest, measurementName)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # search for existing measurements of this type and increment the measurement counter
    filesForThisSn = glob(os.path.join(folder, f'{sn}_{measurementName}*.csv'))
    trial = 1 + len(filesForThisSn)
    fname = f'{sn}_{measurementName}_{trial}.csv'
    if header==None:
        hdr = createHeader(sn, measurementName, trial, columns)
    np.savetxt(os.path.join(dest, measurementName, fname), np.transpose(data), delimiter='\t', header=hdr, fmt=fmt)