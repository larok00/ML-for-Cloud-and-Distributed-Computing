#%% [markdown]
# # Similar Value Clustering

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
NO_OF_BINS = 10
BIN_INTERVAL = 1 / NO_OF_BINS
UTILISATION_BINS = {(i / NO_OF_BINS,
                     i / NO_OF_BINS + BIN_INTERVAL) for i in range(10)}

# 5 minutes,
# 10 minutes,
# 15 minutes,
# 1 hour,
# 1 day,
# 1 week.
# Unit of time in timestamps is 5 minutes.
TREND_LENGTHS = [1, 2, 3, 60 // 5, 1 * 24 * 60 // 5, 7 * 24 * 60 // 5]
TREND_VALS = {'INCREASING', 'STABLE', 'DECREASING'}

class Cluster(object):
    def __init__(self, bin_trends_tup):
        self.bin = bin_trends_tup[0]
        self.trends = bin_trends_tup[1]
        self.members = []
        self.average = None
    
    def add_member(self, new_member):
        self.members.append(new_member)
        cluster_size = len(self.members)
        if self.average is None:
            self.average = new_member
        else:
            self.average = (((cluster_size-1) * self.average + new_member)
                            / cluster_size)

class ClusterMember(object):
    def __init__(self, data_type, machine_no, t):
        corr = TEMPORAL_CORRELATIONS[data_type]
        self.value = corr[machine_no, t]

        lo = self.value // BIN_INTERVAL / NO_OF_BINS
        hi = lo + BIN_INTERVAL
        self.bin = (lo, hi)
        assert self.bin in UTILISATION_BINS, '{} is not in {}'.format(
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
                previous_value = corr[machine_no, t - length]

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

CLUSTERS = dict()

#%%
start = time.process_time()
CORR_CLUSTERINGS = dict()
for data_type in TEMPORAL_CORRELATIONS:
    corr = TEMPORAL_CORRELATIONS[data_type]
    CORR_CLUSTERINGS[data_type] = np.empty(corr.shape, dtype=tuple)
    for machine_no in range(SPATIAL_SAMPLE_SIZE):
        for t in range(NO_OF_TIMESTAMPS):
            lo = corr[machine_no, t] // BIN_INTERVAL / NO_OF_BINS
            hi = lo + BIN_INTERVAL
            bin = (lo, hi)
            assert bin in UTILISATION_BINS, '{} is not in {}'.format(
                bin, UTILISATION_BINS)

            trends = list()
            for length in TREND_LENGTHS:
                if length > t:
                    trend = 'STABLE'
                else:
                    if corr[machine_no, t - length] == 0.0:
                        percent_change = 1.0
                    else:
                        percent_change = (corr[machine_no, t]
                                        / corr[machine_no, t - length]
                                        - 1)

                    if percent_change > 1/3:
                        trend = 'INCREASING'
                    elif percent_change < -1/3:
                        trend = 'DECREASING'
                    else:
                        trend = 'STABLE'

                assert trend in TREND_VALS, '{} is not in {}'.format(
                    trend, TREND_VALS)
                trends.append(trend)

            CORR_CLUSTERINGS[data_type][machine_no, t] = (bin, trends)
elapsed_time = time.process_time() - start

#%%
start = time.process_time()
for data_type in TEMPORAL_CORRELATIONS:
    corr = TEMPORAL_CORRELATIONS[data_type]
    clusters = CORR_CLUSTERINGS[data_type]
    for machine_no in range(SPATIAL_SAMPLE_SIZE):
        for t in range(NO_OF_TIMESTAMPS):
            cluster = clusters[machine_no, t]
            if cluster not in CLUSTERS:
                CLUSTERS[cluster] = Cluster(cluster)
            CLUSTERS[cluster].add_member()

#%%
start = time.process_time()
CORR_CLUSTERINGS = dict()
for data_type in TEMPORAL_CORRELATIONS:
    corr = TEMPORAL_CORRELATIONS[data_type]
    CORR_CLUSTERINGS[data_type] = np.empty(corr.shape, dtype=tuple)
    for machine_no in range(SPATIAL_SAMPLE_SIZE):
        for t in range(NO_OF_TIMESTAMPS):
            member = ClusterMember(data_type, machine_no, t)

            CORR_CLUSTERINGS[data_type][machine_no, t] = (bin, trends)
elapsed_time = time.process_time() - start

#%%
elapsed_time
