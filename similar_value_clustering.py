#%% [markdown]
# # Similar Value Clustering

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
np.random.seed(19)

#%%
CPU_DATA_EXISTS = False
MEM_DATA_EXISTS = False

#%%
if not CPU_DATA_EXISTS:
    CPU_DATA = np.load('google-cpu-full.npy')
    np.random.shuffle(CPU_DATA)
CPU_DATA_EXISTS = True

if not MEM_DATA_EXISTS:
    MEM_DATA = np.load('google-mem-full.npy')
    np.random.shuffle(MEM_DATA)
MEM_DATA_EXISTS = True

print(CPU_DATA.shape)
print(MEM_DATA.shape)

assert CPU_DATA.shape == MEM_DATA.shape

NO_OF_MACHINES = CPU_DATA.shape[0]
NO_OF_TIMESTAMPS = CPU_DATA.shape[1]

#%%
SPATIAL_SAMPLE_SIZE = 200
CPU_SPATIAL_SAMPLE = CPU_DATA[:SPATIAL_SAMPLE_SIZE]
MEM_SPATIAL_SAMPLE = MEM_DATA[:SPATIAL_SAMPLE_SIZE]

CPU_SPATIAL_CORRELATIONS = np.empty(
    (SPATIAL_SAMPLE_SIZE, SPATIAL_SAMPLE_SIZE-1))
MEM_SPATIAL_CORRELATIONS = np.empty(
    (SPATIAL_SAMPLE_SIZE, SPATIAL_SAMPLE_SIZE-1))
SPATIAL_CORRELATIONS = {
    'CPU': CPU_SPATIAL_CORRELATIONS, 'MEM': MEM_SPATIAL_CORRELATIONS}

TEMPORAL_SAMPLE_SIZE = 200
CPU_TEMPORAL_SAMPLE = CPU_DATA[:TEMPORAL_SAMPLE_SIZE]
MEM_TEMPORAL_SAMPLE = MEM_DATA[:TEMPORAL_SAMPLE_SIZE]

CPU_TEMPORAL_CORRELATIONS = np.empty(
    (TEMPORAL_SAMPLE_SIZE, 2*NO_OF_TIMESTAMPS-1))
MEM_TEMPORAL_CORRELATIONS = np.empty(
    (TEMPORAL_SAMPLE_SIZE, 2*NO_OF_TIMESTAMPS-1))
TEMPORAL_CORRELATIONS = {
    'CPU': CPU_TEMPORAL_CORRELATIONS, 'MEM': MEM_TEMPORAL_CORRELATIONS}

#%%
plt.subplot(211).plot(CPU_DATA[0][:2*24*60//5])
plt.title('CPU data')
plt.subplot(212).plot(MEM_DATA[0][:2*24*60//5])
plt.title('MEM data')
plt.tight_layout()
plt.show()

#%%
clusters = dict()
utilisation_bins = [(i/10, i/10+0.1) for i in range(10)]

# 5 minutes,
# 10 minutes,
# 15 minutes,
# 1 hour,
# 1 day,
# 1 week.
# Unit of time in timestamps is 5 minutes.
trend_lengths = [1, 2, 3, 60//5, 1*DAYS_TO_MINUTES//5, 7*DAYS_TO_MINUTES//5]
trends = {INCRESING, STABLE, DECREASING}

for bin in utilisation_bins:minutesdaysytoiminutesDAYSTOMdaysEtoinutes
    for length in trend_lengths:
        clusters[bin, length, ]
