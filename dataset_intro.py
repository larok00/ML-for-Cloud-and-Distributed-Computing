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
def ccf(x, y, lag_max = 100):

    result = (
        ss.correlate(y - np.mean(y), x - np.mean(x), method='direct')
        / (np.std(y) * np.std(x) * len(y)))
    length = (len(result) - 1) // 2
    lo = length - lag_max
    hi = length + (lag_max + 1)

    return result[lo:hi]

#%%
sample_size = 500
cpu_sample = cpu_data[:sample_size]
mem_sample = mem_data[:sample_size]
correlations = np.empty((sample_size, sample_size))
for i in range(sample_size):
    k=0
    for j in range(sample_size):
        machine_x = cpu_data[i][:1000]
        machine_y = cpu_data[j][:1000]
        if i != j:
            correlations[i, k] = ccf(machine_x, machine_y, lag_max=0)
            k += 1
    i += 1

#%%
print(correlations)
print(np.amax(correlations))
print(np.amin(correlations))


#%%
