# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 11:17:08 2022

@author: Active 64 CCS

Experimental paradigm to measure interoceptive awareness
1. 10s blocks - some synced, some no-synced
2. Within the block, - calculate global system time at start - while loop - till 10s have passed
3. In the block, keep calling from buffer continuously (for min samples to capture QRS peak) 
    - let it be overlapping blocks - no need of pause time
4. If R peak is detected, produce sound and skip calling from buffer for say 250ms
5. Write code for no-sync blocks - introduce random delay
6. analyze the recorded ECG to see delay between sound marker and R peak

Steps
- connect to fieldtrip buffer
- Find min samples for scipy peak detection algorithm to find R peak    #-----10
- Code for the peak detection feedback block
- Main program code

To check before testing 
- Baseline time
- baseline peak prominence threshold
- Trial type (sync/ non-sync)
- Trial duration

"""

#%% import libs


import mne.externals.FieldTrip as FieldTrip
from mne.utils import logger
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal #peak detection
# import mne   #if filtering

import random
from threading import Timer
import pygame
import joblib

#%%  'connect fieldtrip client'
# First run Brainvision Recorder app - enable remote data access in config, 
# in the app, file - Open Workspace - C:\Vision\Workfiles\DEMO2_2ch.rwksp  (F3 channel)
# connect ground, and electrode no 1,2 
# In terminal, cd to Downloads folder or wherever the rda2ft file is
# then rda2ft (rda2ft localhost 51244), then 

wait_max=10
ft_client = FieldTrip.Client()
logger.info("FieldTripClient: Waiting for server to start")
start_time, current_time = time.time(), time.time()
success = False
while current_time < (start_time + wait_max):
    try:
        ft_client.connect('localhost', 1972)
        logger.info("FieldTripClient: Connected")
        success = True
        break
    except Exception:
        current_time = time.time()
        time.sleep(0.1) 
if not success:
    raise RuntimeError('Could not connect to FieldTrip Buffer')

# retrieve header
logger.info("FieldTripClient: Retrieving header")
start_time, current_time = time.time(), time.time()
while current_time < (start_time + wait_max):
    ft_header = ft_client.getHeader()
    sfreq = ft_header.fSample
    print(ft_header)
    print(ft_header.labels)
    if ft_header is None:
        current_time = time.time()
        time.sleep(0.1)
    else:
        break
if ft_header is None:
    raise RuntimeError('Failed to retrieve Fieldtrip header!')
else:
    logger.info("FieldTripClient: Header retrieved")
    
# End buffer
# ft_client.disconnect()
   
#############################################################################################################
#%% Find min samples for scipy peak detection algorithm to find R peak - chunk_size - minimum 10 needed 

paths='D:/CCS_Users/sruthi/ecg_test3/'
ecg_data = pd.read_csv(paths+'saved_ecg_190822.csv',index_col=0)
ecg_data_array = np.array(ecg_data)*1e-6
ecg_data_array=ecg_data_array.reshape(len(ecg_data_array))
sf=250

#baseline - find peak threshold
data = ecg_data_array#[:300]
peaks, _ = signal.find_peaks(data)
prominences = signal.peak_prominences(ecg_data_array, peaks)[0]
prom_thres=np.percentile(prominences, 99)*0.5
signal_rpeaks = signal.find_peaks(ecg_data_array,prominence=prom_thres)[0]
plt.plot(data,'g')
plt.scatter(signal_rpeaks,data[signal_rpeaks],color='r')

# Testing if the find_peak function can detect peaks in small overlapping chunks 
plt.plot(ecg_data_array[:500])
chunk_size=10
total_peaks=0
# for i in range(1,len(ecg_data_array[:500]),int(chunk_size/2)):
i= 0
while(i<len(ecg_data_array[:500])):
    data = ecg_data_array[i:i+chunk_size]
    signal_rpeaks = signal.find_peaks(data,prominence=prom_thres)[0]
    plt.plot(np.arange(i,i+chunk_size),data,'g')
    plt.scatter(i+signal_rpeaks,data[signal_rpeaks],color='r')
    plt.pause(0.1)
    if len(signal_rpeaks)>0:
        total_peaks+=1
        i+=int(sf/3)  # skip for next 200something ms as thre wont be a rpeak
    i+=int(chunk_size/2)  # overlapping
print(total_peaks)
# chunk_size=9 detected all peaks within ecg_data_array[:500]  but not all after that - non overlapping- try overlapping
# 10 perf for overlapping , 8 didnt work either

#############################################################################################################
#%% Baseline 5s - calculate Rpeak threshold - capture heart rate - random delay values should be within 70% of this
'Stream baseline data'

# pygame.mixer.music.play()
start_baseline = time.time()
# Get 5s baseline data
time.sleep(5)
H = ft_client.getHeader()
curr_index = H.nSamples - 1   #Starting sample of entire recording
sample_start = curr_index-5*sfreq
data_baseline = ft_client.getData([sample_start,curr_index]).reshape(-1).astype('float64')*-1
print(data_baseline.shape)

# # filter baseine data     
# data_baseline_filt = mne.filter.filter_data(data_baseline,sfreq,None,30)  # not inplace right?
# # print(data_baseline[1],print(data_baseline_filt[1]))
# data_baseline_filt = mne.filter.filter_data(data_baseline_filt,sfreq,3,None)  #*-1
        
#get peak threshold from baseline filtered data         --------- Testing - data_baseline = ecg_data_array[:2000]
peaks, _ = signal.find_peaks(data_baseline) #_filt
prominences = signal.peak_prominences(data_baseline, peaks)[0] 
prom_threshold=np.percentile(prominences, 98)# *0.995
peak_baseline = signal.find_peaks(data_baseline,prominence=prom_threshold)[0] 

# calculate heart rate
# 100bpm means 1.6 beats per second  -> time between Rpeaks = 60/100 = 0.6s = 0.6*sfreq =150samples
peakdiff = np.diff(peak_baseline)
sample_avg=np.average(peakdiff) 
time_bw_peaks = sample_avg/sfreq  # time between rpeaks = sample_avg/sfreq (seconds)
time_delay_min=0.1 * time_bw_peaks
time_delay_max=0.70 * time_bw_peaks

# random_delay = random.uniform(time_delay_min, time_delay_max)

plt.figure(figsize=(20,10))
plt.plot(data_baseline)
plt.scatter(peak_baseline,data_baseline[peak_baseline],color="red")

#############################################################################################################
#%% Function for producing stimuli when rpeak is detected 
# Start streaming data in size of mini sample
# Get last one second of data (big enough to filter)- every call from buffer - repeats many times per second 
#(Now overlapping to prevent peaks being missed)
# Filter the data, find peaks 
# If peaks present in mini_sample_size (last few samples to get real time), then produce sound 
#- This is done because buffer is not updated every sample - it may come in blocks

mini_sample_size = 10 #int(sfreq/10)
# time_trial=10
    
def find_rpeak(time_trial,sync=True): 
    count=0
    total_peaks=0
    soundmarkers_ft = []
    pygame.mixer.music.play()
    time.sleep(1)
    start_trial=time.time()
    start_time, current_time = time.time(), time.time()
    peak_detected=False
    
    while current_time < (start_time + time_trial):
        # plt.cla()
        pygame.mixer.music.set_volume(0.1)
        # If Rpeak detected previously, let process sleep for some time since Rpeak wont be there in next few 100ms 
	#- avoid duplicate peak detection since we overlap the data
        if peak_detected==True:
            time.sleep(time_delay_max)
        peak_detected=False
        count+=1
        
        idx_curr = ft_client.getHeader().nSamples-1 #-1
        # data_curr = ft_client.getData([idx_curr - mini_sample_size,idx_curr]) #.reshape(-1).astype('float64') 
	# since filtering required, take longer data from buffer instead of appending operations
        data_1s = ft_client.getData([idx_curr - 2*sfreq,idx_curr]).reshape(-1).astype('float64') #Get 1s data if need to filter
        # data_1s_filt = mne.filter.filter_data(data_1s,sfreq,None,30,verbose=False)
        # data_1s_filt = mne.filter.filter_data(data_1s,sfreq,3,None,verbose=False)
        data_1s = data_1s*-1
    
        # Find all prominent peaks in 2s data - produce sound only if peak is there in last few samples
        peak_1s = signal.find_peaks(data_1s,prominence=prom_threshold)[0]
        peak_curr=peak_1s[(peak_1s> data_1s.shape[0]-mini_sample_size)]
        # peak_curr = signal.find_peaks(data_curr,prominence=prom_thres)[0]
        
        # Plotting
        # plt.plot(data_1s)
        # plt.scatter(peak_1s,data_1s[peak_1s],color="red")
        
        if peak_curr.shape[0]!=0:
            total_peaks+=1
	    print('r-wave')
            if sync==False:
                random_delay = int(random.uniform(time_delay_min, time_delay_max)*1000)
                # time.sleep(random_delay)   
                pygame.time.wait(random_delay)
                pygame.mixer.music.set_volume(1)
                soundmarkers_ft.append(ft_client.getHeader().nSamples-1)
            else:
                pygame.mixer.music.set_volume(1)
                soundmarkers_ft.append(ft_client.getHeader().nSamples-1) # or idx_last - more delay
            peak_detected=True
            
        time.sleep(0.01) # remove to call from the buffer more times per second - may be duplicate data since sf only 250
        current_time=time.time() 
        
        # plt.draw()
        # plt.axvline(x=data_1s.shape[0]-mini_sample_size,color='r')
        # plt.pause(0.01)
    
    pygame.mixer.music.stop()
    print(count)   
    end_trial=time.time()
    trial_time_taken = np.round(end_trial-start_trial,3)
    print(trial_time_taken)
    print(total_peaks)
    
    return soundmarkers_ft,trial_time_taken

#############################################################################################################
#%% Main: call sync - no sync trials

pygame.mixer.init()
pygame.mixer.music.load(r"D:\CCS_Users\sruthi\ecg\sine10s.wav")
# pygame.mixer.music.load(r"D:\CCS_Users\sruthi\ecg\sample-15s.mp3")
pygame.mixer.music.set_volume(0.1)
# pygame.mixer.music.play()

# random array of 1s and 0s - 5 in sync trials, 5 non-sync trials
trial_choice = np.zeros(4)
randomlist_for_sync = random.sample(range(0,len(trial_choice)), len(trial_choice)//2)
trial_choice[randomlist_for_sync]=1
time_trial=10
# trial_choice=[1,1,0,0,1] 

soundmarker_ft_sync=[]
soundmarker_ft_nosync=[]
trial_times_taken = []

keyboard_responses=[]

for i in trial_choice:
    if i==1:
        smarkers,trial_time = find_rpeak(time_trial=time_trial,sync=True)
        soundmarker_ft_sync.append(smarkers)
    else:
        smarkers,trial_time = find_rpeak(time_trial=time_trial,sync=False)
        soundmarker_ft_nosync.append(smarkers)

    trial_times_taken.append(trial_time)
    #keyboard response
    r=input('Trial ended. \nEnter 1 if synced to your ecg else 9. Make sure to press "Enter" after your choice\n')
    keyboard_responses.append(r)
    time.sleep(2)
    
# end=time.time()
print(" Times taken : ",trial_times_taken)    

#%%'Save complete data from buffer'
final_header = ft_client.getHeader()
print(final_header)  #dir(raw_header)
sample_end = final_header.nSamples-1 
data_whole_rec = ft_client.getData([sample_start,sample_end]).reshape(-1).astype('float64')  # looks like buffer size is 600,000 samples
df = pd.DataFrame(data_whole_rec)

paths='D:/CCS_Users/sruthi/test_feedback/'
df.to_csv(paths+'saved_ecg_200822.csv')

soundmarker_ft_nosync1 = [np.array(sm)-sample_start for sm in soundmarker_ft_nosync]
soundmarker_ft_sync1 = [np.array(sm)-sample_start for sm in soundmarker_ft_sync]
print("synced markers : ",soundmarker_ft_sync1,"\n random markers :",soundmarker_ft_nosync1)

joblib.dump(soundmarker_ft_nosync1,paths+'soundmarkers_nosync.pkl')
joblib.dump(soundmarker_ft_sync1,paths+'soundmarkers_sync.pkl')
joblib.dump(trial_choice,paths+'true_markers.pkl')
joblib.dump(keyboard_responses,paths+'responses.pkl')
# r=joblib.load('D:/CCS_Users/sruthi/ecg/'+'responses.pkl')


# End buffer
ft_client.disconnect()

#%% END
