#%% [markdown]
# # Spatial Correlation Clustering

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

np.random.seed(19)
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

assert CPU_DATA.shape == MEM_DATA.shape

NO_OF_MACHINES = CPU_DATA.shape[0]
NO_OF_TIMESTAMPS = CPU_DATA.shape[1]
days_to_minutes = 24*60

SPATIAL_SAMPLE_SIZE = 200
CPU_SPATIAL_SAMPLE = CPU_DATA[:SPATIAL_SAMPLE_SIZE]
MEM_SPATIAL_SAMPLE = MEM_DATA[:SPATIAL_SAMPLE_SIZE]

#%%
def ccf(x, y, no_lag=False):
    '''Normalized cross-correlation function,
    similar to ccf in the R language.
    
    Parameters:
    x -- first time series
    y --  second time series
    
    Optional:
    no_lag -- False by default. If true, return a list with every time-shift
        possible instead.
    
    Returns:
        A float with 0 time-shift or a list of floats that represent the
        cross-correlation for every possible time-shift.
    '''
    if np.std(y) * np.std(x) == 0:
        return 0
    else:
        correlation = (
            np.correlate(y - np.mean(y), x - np.mean(x))
            if no_lag else ss.correlate(y - np.mean(y), x - np.mean(x)))
        return correlation / (np.std(y) * np.std(x) * len(y))

#%%
time_windows = [2*[NO_OF_TIMESTAMPS//2], 2*[NO_OF_TIMESTAMPS//4],
                2*[1*days_to_minutes//5], 2*[7*days_to_minutes//5],
                2*[2], 2*[4], 2*[10]]

class TwoCorrelationWindows(object):
    def __init__(self):
        self.first = np.empty((SPATIAL_SAMPLE_SIZE, SPATIAL_SAMPLE_SIZE-1))
        self.second = np.empty((SPATIAL_SAMPLE_SIZE, SPATIAL_SAMPLE_SIZE-1))


CPU_SPATIAL_CORRELATIONS = MEM_SPATIAL_CORRELATIONS = dict()
for window in time_windows:
    CPU_SPATIAL_CORRELATIONS[tuple(window)] = (
        TwoCorrelationWindows())
    MEM_SPATIAL_CORRELATIONS[tuple(window)] = (
        TwoCorrelationWindows())
SPATIAL_CORRELATIONS = {
    'CPU': CPU_SPATIAL_CORRELATIONS, 'MEM': MEM_SPATIAL_CORRELATIONS}

#%%
for window in time_windows:
    for i in range(SPATIAL_SAMPLE_SIZE):
        k=0
        for j in range(SPATIAL_SAMPLE_SIZE):
            #If the first and second machine are the same one, skip.
            if i != j:
                for sample_corr_tup in [(CPU_SPATIAL_SAMPLE,
                                         CPU_SPATIAL_CORRELATIONS),
                                        (MEM_SPATIAL_SAMPLE,
                                         MEM_SPATIAL_CORRELATIONS)]:
                    machine_x = sample_corr_tup[0][i][:window[0]]
                    machine_y = sample_corr_tup[0][j][:window[0]]
                    (sample_corr_tup[1][tuple(window)].first)[i, k] = (
                        ccf(machine_x, machine_y, no_lag=True))

                    machine_x = sample_corr_tup[0][i][
                        window[0] + 1 :
                        window[0] + window[1]]
                    machine_y = sample_corr_tup[0][j][
                        window[0] + 1 :
                        window[0] + window[1]]
                    (sample_corr_tup[1][tuple(window)].second)[i, k] = (
                        ccf(machine_x, machine_y, no_lag=True))
                k += 1
        i += 1

#%%
for window_widths in time_windows:
    print(window_widths)
    window = tuple(window_widths)
    i = 1
    for correlation in SPATIAL_CORRELATIONS:
        corr = SPATIAL_CORRELATIONS[correlation]
        avg = np.average(corr[window].first)
        high_corr = (corr[window].first[corr[window].first>avg]
                     - corr[window].second[corr[window].first>avg])

        corr = corr[window].first - corr[window].second
        abs_correlations = np.abs(corr)
        values, base = np.histogram(high_corr,
                                    bins=[n/100 for n in range(-85, 99)])

        ax0 = plt.subplot(len(SPATIAL_CORRELATIONS), 1, i)
        ax0.plot(base[:-1], values)
        ax1 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.plot(base[:-1], np.cumsum(values), '-',
                2*[np.average(corr)], [0, 40000], '--')
        plt.xticks([n/4 for n in range (-2, 4)])
        plt.title(correlation + ' Spatial Correlation Drift')
        i += 1
        print(avg, np.average(abs_correlations), np.average(corr))

    plt.tight_layout()
    plt.show()

#%%

time_windows = [
    [NO_OF_TIMESTAMPS//2, NO_OF_TIMESTAMPS//4],
    [7*days_to_minutes//5, 1*days_to_minutes//5],
    [14*days_to_minutes//5, 1*days_to_minutes//5],
    [14*days_to_minutes//5, 1*days_to_minutes//(24*5)],
]

CPU_SPATIAL_CORRELATIONS = MEM_SPATIAL_CORRELATIONS = dict()
for window in time_windows:
    CPU_SPATIAL_CORRELATIONS[tuple(window)] = (
        TwoCorrelationWindows())
    MEM_SPATIAL_CORRELATIONS[tuple(window)] = (
        TwoCorrelationWindows())
SPATIAL_CORRELATIONS = {
    'CPU': CPU_SPATIAL_CORRELATIONS, 'MEM': MEM_SPATIAL_CORRELATIONS}

#%%
for window in time_windows:
    for i in range(SPATIAL_SAMPLE_SIZE):
        k=0
        for j in range(SPATIAL_SAMPLE_SIZE):
            #If the first and second machine are the same one, skip.
            if i != j:
                for sample_corr_tup in [(CPU_SPATIAL_SAMPLE,
                                         CPU_SPATIAL_CORRELATIONS),
                                        (MEM_SPATIAL_SAMPLE,
                                         MEM_SPATIAL_CORRELATIONS)]:
                    machine_x = sample_corr_tup[0][i][:window[0]]
                    machine_y = sample_corr_tup[0][j][:window[0]]
                    (sample_corr_tup[1][tuple(window)].first)[i, k] = (
                        ccf(machine_x, machine_y, no_lag=True))

                    machine_x = sample_corr_tup[0][i][
                        window[0] + 1 :
                        window[0] + window[1]]
                    machine_y = sample_corr_tup[0][j][
                        window[0] + 1 :
                        window[0] + window[1]]
                    (sample_corr_tup[1][tuple(window)].second)[i, k] = (
                        ccf(machine_x, machine_y, no_lag=True))
                k += 1
        i += 1

#%%
for window_widths in time_windows:
    print(window_widths)
    window = tuple(window_widths)
    i = 1
    for correlation in SPATIAL_CORRELATIONS:
        corr = SPATIAL_CORRELATIONS[correlation]
        avg = np.average(corr[window].first)
        high_corr = (corr[window].first[corr[window].first>avg]
                     - corr[window].second[corr[window].first>avg])

        corr = corr[window].first - corr[window].second
        abs_correlations = np.abs(corr)
        values, base = np.histogram(high_corr,
                                    bins=[n/100 for n in range(-85, 99)])

        ax0 = plt.subplot(len(SPATIAL_CORRELATIONS), 1, i)
        ax0.plot(base[:-1], values)
        ax1 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.plot(base[:-1], np.cumsum(values), '-',
                2*[np.average(corr)], [0, 40000], '--')
        plt.xticks([n/4 for n in range (-2, 4)])
        plt.title(correlation + ' Spatial Correlation Drift')
        i += 1
        print(avg, np.average(abs_correlations), np.average(corr))

    plt.tight_layout()
    plt.show()


#%%
time_windows = [2*[1*days_to_minutes//5], 2*[1*days_to_minutes//5],
                2*[1*days_to_minutes//5], ]

CPU_SPATIAL_CORRELATIONS = MEM_SPATIAL_CORRELATIONS = dict()
for window in time_windows:
    CPU_SPATIAL_CORRELATIONS[tuple(window)] = (
        TwoCorrelationWindows())
    MEM_SPATIAL_CORRELATIONS[tuple(window)] = (
        TwoCorrelationWindows())
SPATIAL_CORRELATIONS = {
    'CPU': CPU_SPATIAL_CORRELATIONS, 'MEM': MEM_SPATIAL_CORRELATIONS}

#%%
for window in time_windows:
    r = np.random.uniform()
    print(r)
    for i in range(SPATIAL_SAMPLE_SIZE):
        k=0
        for j in range(SPATIAL_SAMPLE_SIZE):
            #If the first and second machine are the same one, skip.
            if i != j:
                for sample_corr_tup in [(CPU_SPATIAL_SAMPLE,
                                         CPU_SPATIAL_CORRELATIONS),
                                        (MEM_SPATIAL_SAMPLE,
                                         MEM_SPATIAL_CORRELATIONS)]:
                    machine_x = sample_corr_tup[0][i][int(r*NO_OF_TIMESTAMPS):int(r*NO_OF_TIMESTAMPS)+window[0]]
                    machine_y = sample_corr_tup[0][j][int(r*NO_OF_TIMESTAMPS):int(r*NO_OF_TIMESTAMPS)+window[0]]
                    (sample_corr_tup[1][tuple(window)].first)[i, k] = (
                        ccf(machine_x, machine_y, no_lag=True))

                    machine_x = sample_corr_tup[0][i][
                        int(r*NO_OF_TIMESTAMPS) + window[0] + 1 :
                        int(r*NO_OF_TIMESTAMPS) + window[0] + window[1]]
                    machine_y = sample_corr_tup[0][j][
                        int(r*NO_OF_TIMESTAMPS) + window[0] + 1 :
                        int(r*NO_OF_TIMESTAMPS) + window[0] + window[1]]
                    (sample_corr_tup[1][tuple(window)].second)[i, k] = (
                        ccf(machine_x, machine_y, no_lag=True))
                k += 1
        i += 1

#%%
for window_widths in time_windows:
    print(window_widths)
    window = tuple(window_widths)
    i = 1
    for correlation in SPATIAL_CORRELATIONS:
        corr = SPATIAL_CORRELATIONS[correlation]
        avg = np.average(corr[window].first)
        high_corr = (corr[window].first[corr[window].first>avg]
                     - corr[window].second[corr[window].first>avg])

        corr = corr[window].first - corr[window].second
        abs_correlations = np.abs(corr)
        values, base = np.histogram(high_corr,
                                    bins=[n/100 for n in range(-85, 99)])

        ax0 = plt.subplot(len(SPATIAL_CORRELATIONS), 1, i)
        ax0.plot(base[:-1], values)
        ax1 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.plot(base[:-1], np.cumsum(values), '-',
                2*[np.average(corr)], [0, 40000], '--')
        plt.xticks([n/4 for n in range (-2, 4)])
        plt.title(correlation + ' Spatial Correlation Drift')
        i += 1
        print(avg, np.average(abs_correlations), np.average(corr))

    plt.tight_layout()
    plt.show()


#%%
