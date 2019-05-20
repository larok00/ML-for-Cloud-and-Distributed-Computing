#%% [markdown]
# # Dataset Introduction

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

#%%
np.random.seed(19)
cpu_data_exists = False
mem_data_exists = False

#%%
if not cpu_data_exists:
    cpu_data = np.load('google-cpu-full.npy')
    np.random.shuffle(cpu_data)
cpu_data_exists = True

#%%
if not mem_data_exists:
    mem_data = np.load('google-cpu-full.npy')
    np.random.shuffle(mem_data)
mem_data_exists = True

#%% [markdown]
# There are 12476 machines, each with 5351 datapoints.

#%%
print(cpu_data.shape)
print(mem_data.shape)

assert cpu_data.shape == mem_data.shape

no_of_machines = cpu_data.shape[0]
no_of_timestamps = cpu_data.shape[1]

#%% [markdown]
# We will be focusing on cpu and memory usage data.

#%%
plt.subplot(121).plot(cpu_data[0])
plt.subplot(122).plot(mem_data[0])
plt.show()

#%%
def ccf(x, y, no_lag=False):
    correlation = (
        np.correlate(y - np.mean(y), x - np.mean(x))
        if no_lag else ss.correlate(y - np.mean(y), x - np.mean(x)))
    
    return correlation / (np.std(y) * np.std(x) * len(y))

#%%
spatial_sample_size = 200
cpu_spatial_sample = cpu_data[:spatial_sample_size]
mem_spatial_sample = mem_data[:spatial_sample_size]

cpu_spatial_correlations = np.empty((spatial_sample_size, spatial_sample_size-1))
mem_spatial_correlations = np.empty((spatial_sample_size, spatial_sample_size-1))
spatial_correlations = [cpu_spatial_correlations, mem_spatial_correlations]

#%%
temporal_sample_size = 200
cpu_temporal_sample = cpu_data[:temporal_sample_size]
mem_temporal_sample = mem_data[:temporal_sample_size]

cpu_temporal_correlations = np.empty(
    (temporal_sample_size, 2*no_of_timestamps-1))
mem_temporal_correlations = np.empty(
    (temporal_sample_size, 2*no_of_timestamps-1))
temporal_correlations = [cpu_temporal_correlations, mem_temporal_correlations]

#%%
for i in range(spatial_sample_size):
    k=0
    for j in range(spatial_sample_size):
        if i != j:
            machine_x = cpu_spatial_sample[i]
            machine_y = cpu_spatial_sample[j]
            cpu_spatial_correlations[i, k] = ccf(machine_x,
                                                 machine_y,
                                                 no_lag=True)
            machine_x = mem_spatial_sample[i]
            machine_y = mem_spatial_sample[j]
            mem_spatial_correlations[i, k] = ccf(machine_x,
                                                 machine_y,
                                                 no_lag=True)
            k += 1
    i += 1

#%%
for correlations in spatial_correlations:
    abs_correlations = np.abs(correlations)
    print('Maximum:', np.amax(abs_correlations))
    print('Minimum:', np.amin(abs_correlations))
    print('Average:', np.average(correlations))
    plt.hist(correlations, bins=[n/10 for n in range(-10, 11)])
    plt.show()

#%%
for i in range(temporal_sample_size):
    cpu_temporal_correlations[i] = ccf(
        cpu_temporal_sample[i],
        cpu_temporal_sample[i])
    mem_temporal_correlations[i] = ccf(
        mem_temporal_sample[i],
        mem_temporal_sample[i])

#%%
for correlations in temporal_correlations:
    for correlation in correlations[:5]:
        smoothed = ss.medfilt(correlation, kernel_size=101)
        plt.plot(
            range(len(smoothed)), smoothed, '-',
            2*[no_of_timestamps-1], [0, 1], '--',
            2*[no_of_timestamps-1-24*60/5], [0, 1], '--',
            2*[no_of_timestamps-1+24*60/5], [0, 1], '--',
            2*[no_of_timestamps-1-7*24*60/5], [0, 1], '--',
            2*[no_of_timestamps-1+7*24*60/5], [0, 1], '--',
            2*[no_of_timestamps-1-2*7*24*60/5], [0, 1], '--',
            2*[no_of_timestamps-1+2*7*24*60/5], [0, 1], '--',
            2*[no_of_timestamps-1-3*7*24*60/5], [0, 1], '--',
            2*[no_of_timestamps-1+3*7*24*60/5], [0, 1], '--',
            )
        plt.show()

#%%
max_coeff = 0
for i in range(len(temporal_correlations[1][1])):
    coeff = temporal_correlations[1][1][i]
    if coeff > max_coeff:
        max_coeff = coeff
        index = i
print(index)

#%%
for correlations in temporal_correlations:
    avg_correlation = np.average(correlations, axis=0)
    smoothed = ss.medfilt(avg_correlation, kernel_size=25)
    plt.plot(
        range(len(smoothed)), smoothed, '-',
        2*[no_of_timestamps-1], [0, 1], '--',
        2*[no_of_timestamps-1-24*60/5], [0, 1], '--',
        2*[no_of_timestamps-1+24*60/5], [0, 1], '--',
        2*[no_of_timestamps-1-7*24*60/5], [0, 1], '--',
        2*[no_of_timestamps-1+7*24*60/5], [0, 1], '--',
        2*[no_of_timestamps-1-2*7*24*60/5], [0, 1], '--',
        2*[no_of_timestamps-1+2*7*24*60/5], [0, 1], '--',
        2*[no_of_timestamps-1-3*7*24*60/5], [0, 1], '--',
        2*[no_of_timestamps-1+3*7*24*60/5], [0, 1], '--',
        )
    plt.show()

#%%
for correlations in temporal_correlations:
    avg_correlation = np.average(correlations, axis=0)
    print(np.average(avg_correlation))
    plt.hist(avg_correlation, bins=[n/10 for n in range(-10, 11)])
    plt.show()

#%%
