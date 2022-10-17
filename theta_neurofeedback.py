# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:11:12 2021


Acquisition System : Brainvision actiCHamp Plus
Connect electrodes
On Brainvision Recorder app, enable remote data access in configuration tab
Download rda2ft (transports data from an RDA server to a FieldTrip buffer)
From terminal, cd to folder containing rda2ft executable, rda2ft localhost 51244
Run code: Get data from the buffer, Provide theta neurofeedback

Links
https://www.fieldtriptoolbox.org/development/realtime/rda/#standalone-interface-with-rda2ft
https://github.com/fieldtrip/fieldtrip/tree/68835fdcb885304369f49f93aa47d08434972079/realtime/src/acquisition/brainproducts
https://mne.tools/mne-realtime/auto_examples/plot_ftclient_rt_average.html

"""

import os.path as op
import matplotlib.pyplot as plt
import subprocess
import mne
from mne.utils import running_subprocess
from mne.io import read_raw_brainvision
from mne_realtime import FieldTripClient, RtEpochs
import time
import numpy as np
from mne.time_frequency import psd_welch

# from fieldtrip2mne import read_raw
# pip install fieldtrip2mne

print(__doc__)


#the Fieldtrip buffer does not contain all the measurement information required by the 
#MNE real-time processing pipeline,so an info dictionary must be provided to instantiate FieldTripClient.
raw_fname = '.\Test20210819.vhdr'
raw = read_raw_brainvision(raw_fname, preload=True).pick('eeg')
info = raw.info
bads = []

import os
os.chdir('.\sruthi')
#%%

fig, ax = plt.subplots(1)

# speedup = 10
command = ["rda2ft", "localhost", '51244']

theta_values= []

#command only runs where rda2ft is placed -> Downloads folder currently
with running_subprocess(command, after='kill',
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    with FieldTripClient(host='localhost', port=1972,
                     tmax=30, wait_max=5, info=info) as rt_client:
        # get measurement info guessed by MNE-Python
        raw_info = rt_client.get_measurement_info()

        # n_fft = 256  # the FFT size. Ideally a power of 2
        n_samples = 2048  # time window on which to compute FFT

        # make sure at least one epoch is available
        time.sleep(n_samples / info['sfreq'])
        sfreq = int(raw_info['sfreq'])
        print(raw_info)
        
        # for ii in range(10):
        #     plt.cla()
        #     epoch = rt_client.get_data_as_epoch(n_samples=sfreq)
        #     #continuous_data = epoch.get_data()
        #     epoch.filter(2,100)
        #     x =epoch._data[0,0,:]  #1st channel
        #     plt.plot(epoch.times,x)
        #     plt.draw()
        #     plt.pause(1)
            
            
        for ii in range(10):
            plt.cla()
            epoch = rt_client.get_data_as_epoch(n_samples=sfreq)
            epoch.filter(2,100)
            x =epoch._data[0,0,:]  #1st channel
            y = np.fft.fft(x,1000)  #sfreq =1000?
            freq = np.fft.fftfreq(1000)*sfreq  
            y = y[freq>0]
            freq = freq[freq>0]
            #plt.plot(freq, abs(y))
            
            #theta = y[freq>4 and freq<8]
            theta = np.logical_and((freq>4),(freq<8))
            theta_y = y[theta]
            theta_mean = abs(theta_y).mean()
            print(theta_mean)
            theta_values.append(theta_mean)
            plt.plot(theta_values)
            plt.xlim((0,5))
            plt.draw()
            plt.pause(1)
            

    
