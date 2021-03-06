#%% [markdown]
# # Similar Value Clustering Finalised

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import time

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
DATA = {'CPU': CPU_DATA, 'MEM': MEM_DATA}

#%%
SPATIAL_SAMPLE_SIZE = NO_OF_MACHINES
CPU_SPATIAL_SAMPLE = CPU_DATA[:SPATIAL_SAMPLE_SIZE]
MEM_SPATIAL_SAMPLE = MEM_DATA[:SPATIAL_SAMPLE_SIZE]
SPATIAL_SAMPLES = {'CPU': CPU_SPATIAL_SAMPLE, 'MEM': MEM_SPATIAL_SAMPLE}

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
NO_OF_BINS = 10
BIN_INTERVAL = 1 / NO_OF_BINS
UTILISATION_BINS = {(i / NO_OF_BINS,
                     (i + 1) / NO_OF_BINS) for i in range(NO_OF_BINS)}

# 5 minutes,
# 10 minutes,
# 15 minutes,
# 1 hour,
# 1 day,
# 1 week.
# Unit of time in timestamps is 5 minutes.
TREND_LENGTHS = [1, 2, 3, 1 * 24 * 60 // 5]
TREND_VALS = {'INCREASING', 'STABLE', 'DECREASING'}

class Cluster(object):
    def __init__(self, t, bin, trends):
        self.members = []
        self.value = None
        self.t = t
        self.bin = bin
        self.previous_values = dict()
        self.percent_changes = dict()
        self.trends = trends
        for trend in self.trends:
            self.percent_changes[trend] = None
            self.previous_values[trend] = None
        
        self.history = None
    
    def add_member(self, new_member):
        assert new_member.t == self.t
        self.members.append(new_member)
        cluster_size = len(self.members)
        if self.value is None:
            self.value = new_member.value
        else:
            self.value = (((cluster_size-1) * self.value + new_member.value)
                            / cluster_size)
        for trend in self.trends:
            if self.previous_values[trend] is None:
                self.previous_values[trend] = new_member.previous_values[trend]
            elif new_member.previous_values[trend] is not None:
                self.previous_values[trend] = (
                    ((cluster_size-1) * self.previous_values[trend]
                    + new_member.previous_values[trend]) / cluster_size)

            if self.percent_changes[trend] is None:
                self.percent_changes[trend] = new_member.percent_changes[trend]
            elif new_member.percent_changes[trend] is not None:
                self.percent_changes[trend] = (
                    ((cluster_size-1) * self.percent_changes[trend]
                    + new_member.percent_changes[trend]) / cluster_size)
        
        if self.history is None:
            self.history = new_member.history
        else:
            self.history = (
                ((cluster_size-1) * self.history + new_member.history)
                / cluster_size)

class ClusterMember(object):
    def __init__(self, data_type, machine_no, t):
        samples = SPATIAL_SAMPLES[data_type]
        self.value = samples[machine_no, t]
        self.machine_no = machine_no
        self.t = t

        if self.value < 0:
            self.bin = (0.0, 0.1)
        elif self.value > 1:
            self.bin = (0.9, 1.0)
        else:
            lo = self.value // BIN_INTERVAL / NO_OF_BINS
            hi = round(lo + BIN_INTERVAL, 2)
            self.bin = (lo, hi)
        assert self.bin in UTILISATION_BINS, '{} is not in {} {}.'.format(
            self.bin, UTILISATION_BINS)
        self.previous_values = dict()
        self.percent_changes = dict()
        self.trends = dict()

        for length in TREND_LENGTHS:
            if length > t:
                previous_value = None
                percent_change = None
                trend = 'STABLE'
            else:
                previous_value = samples[machine_no, t - length]

                if previous_value == 0.0:
                    percent_change = -1.0
                else:
                    percent_change = self.value / previous_value - 1

                if percent_change > 1/3:
                    trend = 'INCREASING'
                elif percent_change < -1/3:
                    trend = 'DECREASING'
                else:
                    trend = 'STABLE'
            assert trend in TREND_VALS, '{} is not in {}'.format(trend,
                                                                 TREND_VALS)

            self.previous_values[length] = previous_value
            self.percent_changes[length] = percent_change
            self.trends[length] = trend
        
        self.history = None if t == 0 else samples[machine_no, :t]

CLUSTERS_AT_T = dict()
for t in range(NO_OF_TIMESTAMPS)[19:20]:
    CLUSTERS_AT_T[t] = dict()

#%%
start = time.process_time()
for data_type in SPATIAL_SAMPLES:
    samples = SPATIAL_SAMPLES[data_type]
    for machine_no in range(SPATIAL_SAMPLE_SIZE):
        for t in range(NO_OF_TIMESTAMPS)[19:20]:
            clusters = CLUSTERS_AT_T[t]
            member = ClusterMember(data_type, machine_no, t)
            key = [member.bin]
            for trend_length in TREND_LENGTHS:
                key.append(member.trends[trend_length])
            key = tuple(key)
            if key not in clusters:
                clusters[key] = (
                    Cluster(member.t, member.bin, member.trends))
            clusters[key].add_member(member)
elapsed_time = time.process_time() - start

#%%
print(elapsed_time)
print('\n')
sum = 0
j = 1
for i in CLUSTERS_AT_T[19]:
    cluster = CLUSTERS_AT_T[19][i]
    sum += len(cluster.members)
    plt.subplot(40, 10, j).plot(cluster.history)
    j += 1
sum
plt.show()

#%%
for key in CLUSTERS_AT_T[19].keys():
    print(len(key))
    break
print(j)
print(len(TREND_LENGTHS)*len(TREND_VALS)*NO_OF_BINS)
sum/NO_OF_MACHINES

#%%

