import datetime
import numpy as np
import os
import pandas as pd
from glob import glob

def create_header(sn, measurement_name, trial, columns, sep='\t'):    
    header_string = f'sn: {sn}\nmeasurement: {measurement_name}\ntrial: {trial}\n{sep.join(columns)}'
    return header_string

def save_measurement(dest, sn, measurement_name, data, columns, header=None, fmt='%.6e'):
    folder = os.path.join(dest, measurement_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # search for existing measurements of this type and increment the measurement counter
    files = glob(os.path.join(folder, f'{sn}_{measurement_name}*.csv'))
    trial = 1 + len(files)
    fname = f'{sn}_{measurement_name}_{trial}.csv'
    if header==None:
        hdr = create_header(sn, measurement_name, trial, columns)
    np.savetxt(os.path.join(dest, measurement_name, fname), np.transpose(data), delimiter='\t', header=hdr, fmt=fmt)

def load_measurement(dest, sn, measurement_name):
    folder = os.path.join(dest, measurement_name)
    files = sorted(glob(os.path.join(folder, f'{sn}_{measurement_name}*.csv')), key=os.path.getmtime)    

    if not files:        
        return None
    else:
        print(f'Loading {files[-1]}')
        return np.loadtxt(files[-1], comments='#')
    