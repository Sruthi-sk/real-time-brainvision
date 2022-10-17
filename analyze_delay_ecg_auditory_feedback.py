# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:49:52 2022

Measure delay between auditory feedback and the actual R-peak in the saved data in terms of sample numbers
Synced markers should show minimal variation for every peak-sound combination, 
 while non-synced markers should show high variation
Observation: The soundmarkers have delay of less than 10 samples ~ real-time

"""
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import neurokit2 as nk

paths='./test_feedback/'
ecg_data = pd.read_csv(paths+'saved_ecg_200822.csv',index_col=0)

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx #array[idx]

ecg_data_array = np.array(ecg_data)*1e-6
ecg_data_array=ecg_data_array.reshape(len(ecg_data_array))
ecg_data_array=ecg_data_array*-1

keyboard_responses=joblib.load(paths+'responses.pkl')
trial_choice=joblib.load(paths+'true_markers.pkl')
soundmarkers_sync=joblib.load(paths+'soundmarkers_sync.pkl')
soundmarkers_nosync=joblib.load(paths+'soundmarkers_nosync.pkl')


#convert to int
soundmarkers_sync_int = []
for sblock in soundmarkers_sync:
    soundmarkers_sync_int.append( np.array([int(sm) for sm in sblock] ))
soundmarkers_nosync_int = []
for sblock in soundmarkers_nosync:
    soundmarkers_nosync_int.append( np.array([int(sm) for sm in sblock] ))

##############################################################################################################
#%% plot all sync soundmarkers

soundmarkers_sync_all = np.hstack(soundmarkers_sync)
# convert to int values
soundmarkers_sync_all=np.array([int(s) for s in soundmarkers_sync_all])
# soundmarkers_nosync=np.array([int(s) for s in soundmarkers_nosync_all])

# plot soundmarkers
plt.plot(ecg_data_array,'g')
plt.scatter(soundmarkers_sync_all,ecg_data_array[soundmarkers_sync_all],color='b',zorder=3)
# Plot with time
plt.figure(figsize=(30,10))
times = np.linspace(0,len(ecg_data_array)/256,len(ecg_data_array))
plt.plot(times,ecg_data_array,'g')
plt.scatter(soundmarkers_sync_all/256,ecg_data_array[soundmarkers_sync_all],color='b',zorder=3)

##############################################################################################################
#%% rpeak detection (scipy)

import scipy.signal as signal
peaks, _ = signal.find_peaks(ecg_data_array)
prominences = signal.peak_prominences(ecg_data_array, peaks)[0]
prom_thres=np.percentile(prominences, 99)*0.7
signal_rpeaks = signal.find_peaks(ecg_data_array,prominence=prom_thres)[0]
plt.plot(ecg_data_array,'g')
plt.scatter(signal_rpeaks,ecg_data_array[signal_rpeaks],color='r')
# plt.scatter(nk_rpeaks,ecg_data_array[nk_rpeaks])   # some ecg peaks, neurokit2 didnt detect

#%%Neurokit detect rpeaks

_, nk_rpeaks_all = nk.ecg_peaks(ecg_data_array, sampling_rate=250)  #[:188] minimum samples?
nk_rpeaks = nk_rpeaks_all['ECG_R_Peaks']

# hrv = nk.hrv_time(nk_rpeaks, sampling_rate=250, show=True)
# hrv = nk.hrv_nonlinear(nk_rpeaks, sampling_rate=250, show=True)                   
hrv_indices = nk.hrv(nk_rpeaks, sampling_rate=100, show=True
                     
# plot rpeaks
plot = nk.events_plot(nk_rpeaks_all['ECG_R_Peaks'], ecg_data_array)
plt.plot(ecg_data_array,'g')
plt.scatter(nk_rpeaks,ecg_data_array[nk_rpeaks])   # some ecg peaks, neurokit2 didnt detect

##############################################################################################################
#%%get sample differences for synced block - between soundmarkers and rpeaks
# We have 3 sync blocks 

# get ecg data and soundmarker and rpeaks of each group 
# nk_rpeak_groups=[]
ecg_idxs =[]
signal_rpeak_groups = []
for group in soundmarkers_sync_int:
    # print(g)
    # Take the start and end indices of the soundmarker block
    start,end=group[0]-100,group[-1]+100  
    print(start,end)
    #find nk_rpeaks within this group
    # nk_closest_idx = np.array([find_nearest_idx(nk_rpeaks,start),find_nearest_idx(nk_rpeaks,end)])
    # nk_rpeak_groups.append(nk_rpeaks[nk_closest_idx[0]:nk_closest_idx[1]])
    # find signal's rpeaks within this group
    signal_closest_idx = np.array([find_nearest_idx(signal_rpeaks,start),find_nearest_idx(signal_rpeaks,end)])
    signal_rpeak_groups.append(signal_rpeaks[signal_closest_idx[0]:signal_closest_idx[1]])
    #find ecg data within this group
    ecg_idxs.append([start,end])
    # ecg_groups.append(ecg_data_array[start:end])

# visualize one of the blocks
i=0
start,end = int(ecg_idxs[i][0]),int(ecg_idxs[i][1])
plt.plot(ecg_data_array[start:end],color='g')
# plt.scatter(nk_rpeak_groups[i]-start,ecg_data_array[nk_rpeak_groups[i]],color='r')
plt.scatter(signal_rpeak_groups[i]-start,ecg_data_array[signal_rpeak_groups[i]],color='r')
plt.scatter(soundmarkers_sync_int[i]-start,ecg_data_array[soundmarkers_sync_int[i]],color='b',zorder=5)

# signal_rpeak_groups[i]
# nk_rpeak_groups[i] #.shape
# soundmarkers_sync_int[i] #.shape

# since some rpeaks were not detected, get sample difference as soundmarkers - closest rpeak
#Neurokit rpeaks
# sample_diffs_sync = []
# for i in range(len(soundmarkers_sync_int)):
#     for soundmk in soundmarkers_sync_int[i]:
#         nearest_rpeaks_idx = find_nearest_idx(nk_rpeak_groups[i],soundmk)
#         sample_diff = soundmk - nk_rpeak_groups[i][nearest_rpeaks_idx]
#         sample_diffs_sync.append(sample_diff)
# for signal
sample_diffs_sync = []
for i in range(len(soundmarkers_sync_int)):
    for soundmk in soundmarkers_sync_int[i]:
        nearest_rpeaks_idx = find_nearest_idx(signal_rpeak_groups[i],soundmk)
        sample_diff = soundmk - signal_rpeak_groups[i][nearest_rpeaks_idx]
        sample_diffs_sync.append(sample_diff)
    
##############################################################################################################
#%%get sample differences for NON synced block

# get ecg data and soundmarker and rpeaks of each group 
# nk_rpeak_groups=[]
ecg_idxs =[]
signal_rpeak_groups = []
for group in soundmarkers_nosync_int:
    # print(g)
    # Take the start and end indices of the soundmarker block
    start,end=group[0]-100,group[-1]+100  
    print(start,end)
    # find signal's rpeaks within this group
    signal_closest_idx = np.array([find_nearest_idx(signal_rpeaks,start),find_nearest_idx(signal_rpeaks,end)])
    signal_rpeak_groups.append(signal_rpeaks[signal_closest_idx[0]:signal_closest_idx[1]])
    #find ecg data within this group
    ecg_idxs.append([start,end])
    # ecg_groups.append(ecg_data_array[start:end])

# visualize one of the blocks
i=0
start,end = int(ecg_idxs[i][0]),int(ecg_idxs[i][1])
plt.plot(ecg_data_array[start:end],color='g')
# plt.scatter(nk_rpeak_groups[i]-start,ecg_data_array[nk_rpeak_groups[i]],color='r')
plt.scatter(signal_rpeak_groups[i]-start,ecg_data_array[signal_rpeak_groups[i]],color='r')
plt.scatter(soundmarkers_nosync_int[i]-start,ecg_data_array[soundmarkers_nosync_int[i]],color='b',zorder=5)


# since some rpeaks were not detected, get sample difference as soundmarkers - closest rpeak
sample_diffs_nosync = []
for i in range(len(soundmarkers_nosync_int)):
    for soundmk in soundmarkers_nosync_int[i]:
        nearest_rpeaks_idx = find_nearest_idx(signal_rpeak_groups[i],soundmk)
        sample_diff = soundmk - signal_rpeak_groups[i][nearest_rpeaks_idx]
        sample_diffs_nosync.append(sample_diff)

##############################################################################################################
#%% detrend data
# import scipy
# data_detrend = signal.detrend(ecg_data_array)  #,axis=-1,type='linear',bp=0
# plt.figure(figsize=(30,10))
# plt.plot(data_detrend,'g')




