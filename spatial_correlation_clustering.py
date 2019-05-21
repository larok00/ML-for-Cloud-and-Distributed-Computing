#%% [markdown]
# # Spatial Correlation Clustering

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

np.random.seed(19)
cpu_data_exists = False
mem_data_exists = False

#%%
if not cpu_data_exists:
    cpu_data = np.load('google-cpu-full.npy')
    np.random.shuffle(cpu_data)
cpu_data_exists = True

if not mem_data_exists:
    mem_data = np.load('google-mem-full.npy')
    np.random.shuffle(mem_data)
mem_data_exists = True

assert cpu_data.shape == mem_data.shape

no_of_machines = cpu_data.shape[0]
no_of_timestamps = cpu_data.shape[1]
days_to_minutes = 24*60

spatial_sample_size = 200
cpu_spatial_sample = cpu_data[:spatial_sample_size]
mem_spatial_sample = mem_data[:spatial_sample_size]

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
time_windows = [no_of_timestamps//2, no_of_timestamps//4,
                1*days_to_minutes//5, 7*days_to_minutes//5]

class TwoCorrelationWindows(object):
    def __init__(self):
        self.first = np.empty((spatial_sample_size, spatial_sample_size-1))
        self.second = np.empty((spatial_sample_size, spatial_sample_size-1))


cpu_spatial_correlations = mem_spatial_correlations = dict()
for window_width in time_windows:
    cpu_spatial_correlations[window_width] = TwoCorrelationWindows()
    mem_spatial_correlations[window_width] = TwoCorrelationWindows()
spatial_correlations = {
    'CPU': cpu_spatial_correlations, 'MEM': mem_spatial_correlations}

#%%
for window in time_windows:
    for i in range(spatial_sample_size):
        k=0
        for j in range(spatial_sample_size):
            #If the first and second machine are the same one, skip.
            if i != j:
                for sample_corr_tup in [(cpu_spatial_sample,
                                         cpu_spatial_correlations),
                                        (mem_spatial_sample,
                                         mem_spatial_correlations)]:
                    machine_x = sample_corr_tup[0][i][:window]
                    machine_y = sample_corr_tup[0][j][:window]
                    (sample_corr_tup[1][window].first)[i, k] = ccf(machine_x,
                                                                   machine_y,
                                                                   no_lag=True)

                    machine_x = sample_corr_tup[0][i][window + 1 : 2 * window]
                    machine_y = sample_corr_tup[0][j][window + 1 : 2 * window]
                    (sample_corr_tup[1][window].second)[i, k] = ccf(machine_x,
                                                                machine_y,
                                                                no_lag=True)
                k += 1
        i += 1

#%%
for window in time_windows:
    print(window)
    i = 1
    for correlation in spatial_correlations:
        corr = spatial_correlations[correlation]
        avg = np.average(corr[window].first)
        high_corr = (corr[window].first[corr[window].first>avg]
                     - corr[window].second[corr[window].first>avg])

        corr = corr[window].first - corr[window].second
        abs_correlations = np.abs(corr)
        values, base = np.histogram(high_corr, bins=[n/100 for n in range(-85, 99)])

        ax0 = plt.subplot(len(spatial_correlations), 1, i)
        ax0.plot(base[:-1], values)
        ax1 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.plot(base[:-1], np.cumsum(values), '-',
                2*[np.average(corr)], [0, 40000], '--')
        plt.xticks([n/4 for n in range (-2, 4)])
        plt.title(correlation + ' Spatial Correlation Drift')
        i += 1
        print(avg, np.average(abs_correlations))

    plt.tight_layout()
    plt.show()

#%%
