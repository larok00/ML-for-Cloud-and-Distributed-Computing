#%% [markdown]
# # Temporal Correlation

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import pickle
import time

np.random.seed(19)
DATA_DIR = 'data/'
PICKLE_DIR = 'pickles/'
DAYS_TO_MINUTES = 24*60
DATA_TYPES = {'CPU', 'MEM'}
SAMPLE_SIZE = 1250//10

#%%
DATA = dict()
SAMPLES = dict()

#%%
for data_type in DATA_TYPES:
    if data_type not in DATA:
        DATA[data_type] = np.load(DATA_DIR + data_type + '.npy')
        np.random.shuffle(DATA[data_type])

second_data_type = None
for first_data_type in DATA:
    if second_data_type is not None:
        assert DATA[first_data_type].shape == DATA[second_data_type].shape
    second_data_type = first_data_type

assert second_data_type is not None
NO_OF_MACHINES = DATA[second_data_type].shape[0]
NO_OF_TIMESTAMPS = DATA[second_data_type].shape[1]

SAMPLES = dict()
for data_type in DATA:
    SAMPLES[data_type] = DATA[data_type][:SAMPLE_SIZE]

#%%
try:
    pickle_in = open("temporal_correlations.pickle","rb")
    TEMPORAL_CORRELATIONS = pickle.load(pickle_in)
except FileNotFoundError:
    CPU_TEMPORAL_CORRELATIONS = np.full(
        (SAMPLE_SIZE, 2*NO_OF_TIMESTAMPS-1), np.nan)
    MEM_TEMPORAL_CORRELATIONS = np.full(
        (SAMPLE_SIZE, 2*NO_OF_TIMESTAMPS-1), np.nan)
    TEMPORAL_CORRELATIONS = {
        'CPU': CPU_TEMPORAL_CORRELATIONS, 'MEM': MEM_TEMPORAL_CORRELATIONS}

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

#%% [markdown]
# Calculate the temporal correlation of each time series with itself,
# at every possible time-shift.

#%%
for i in range(SAMPLE_SIZE):
    for data_type in DATA_TYPES:
        TEMPORAL_CORRELATIONS[data_type][i] = ccf(
            SAMPLES[data_type][i],
            SAMPLES[data_type][i])

#%% [markdown]
# Have a look at the average temporal correlation for each time shift.

#%%
i = 1
zero_shift_timestamp = NO_OF_TIMESTAMPS-1
DAYS_TO_MINUTES = 24*60
minute_vertical_range = [0, 1]
daily_vertical_range = [0, 1]
weekly_vertical_range = [-0.1, 0.5]
for data_type in TEMPORAL_CORRELATIONS:
    avg_correlation = np.average(TEMPORAL_CORRELATIONS[data_type], axis=0)

    # Demonstrate 5-minute periodicity.

    # One unit of time equals 5 minutes.
    time_window = (
        zero_shift_timestamp - 4 * 5 // 5,
        zero_shift_timestamp + 4 * 5 // 5)
    print(time_window)
    plt.plot(
        # One unit of time equals 5 minutes.
        [5 * (x - zero_shift_timestamp)
         for x in range(len(avg_correlation))][time_window[0]:time_window[1]],
        avg_correlation[time_window[0]:time_window[1]], '-',
        # zero time-shift
        2*[0], daily_vertical_range, '--',
        # +/- a few days
        2*[1*5], daily_vertical_range, '--',
        2*[-1*5], daily_vertical_range, '--',
        2*[2*5], daily_vertical_range, '--',
        2*[-2*5], daily_vertical_range, '--',
        2*[3*5], daily_vertical_range, '--',
        2*[-3*5], daily_vertical_range, '--')
    plt.title(data_type + ' 5 minute Periodicity')
    plt.xlabel('Time-shift (Minutes)')
    plt.ylabel('Cross-correlation')
    plt.show()


    # Demonstrate daily periodicity.

    # One unit of time equals 5 minutes.
    time_window = (
        zero_shift_timestamp - 4 * DAYS_TO_MINUTES // 5,
        zero_shift_timestamp + 4 * DAYS_TO_MINUTES // 5)
    plt.plot(
        # One unit of time equals 5 minutes.
        [5 * (x - zero_shift_timestamp) / DAYS_TO_MINUTES
         for x in range(len(avg_correlation))][time_window[0]:time_window[1]],
        avg_correlation[time_window[0]:time_window[1]], '-',
        # zero time-shift
        2*[0], daily_vertical_range, '--',
        # +/- a few days
        2*[1], daily_vertical_range, '--',
        2*[-1], daily_vertical_range, '--',
        2*[2], daily_vertical_range, '--',
        2*[-2], daily_vertical_range, '--',
        2*[3], daily_vertical_range, '--',
        2*[-3], daily_vertical_range, '--')
    plt.title(data_type + ' Daily Periodicity')
    plt.xlabel('Time-shift (Days)')
    plt.ylabel('Cross-correlation')
    plt.show()
    
    # Demonstrate weekly periodicity.

    # One unit of time equals 5 minutes.
    time_window = (
        zero_shift_timestamp - 4 * 7 * DAYS_TO_MINUTES // 5,
        zero_shift_timestamp + 4 * 7 * DAYS_TO_MINUTES // 5)
    smoothed = ss.medfilt(avg_correlation, kernel_size=249)
    plt.plot(
        # One unit of time equals 5 minutes.
        [5 * (x - zero_shift_timestamp) / (7 * DAYS_TO_MINUTES)
         for x in range(len(smoothed))][time_window[0]:time_window[1]],
        smoothed[time_window[0]:time_window[1]], '-',
        # zero time-shift
        2*[0], weekly_vertical_range, '--',
        # +/- a few weeks
        2*[4/7], weekly_vertical_range, '--',
        2*[-4/7], weekly_vertical_range, '--',
        2*[1], weekly_vertical_range, '--',
        2*[-1], weekly_vertical_range, '--',
        2*[2], weekly_vertical_range, '--',
        2*[-2], weekly_vertical_range, '--',
        2*[3], weekly_vertical_range, '--',
        2*[-3], weekly_vertical_range, '--')
    plt.title(data_type + ' Weekly Periodicity')
    plt.xlabel('Time-shift (Weeks)')
    plt.ylabel('Cross-correlation')
    plt.show()


    plt.plot(
        # One unit of time equals 5 minutes.
        [5 * (x - zero_shift_timestamp) / DAYS_TO_MINUTES
         for x in range(len(avg_correlation))][zero_shift_timestamp:],
        avg_correlation[zero_shift_timestamp:], '-',
        # zero time-shift
        2*[0], daily_vertical_range, '--',
        # +/- a few days
        2*[1/24], daily_vertical_range, '--',
        2*[-1/24], daily_vertical_range, '--',
        2*[2/24], daily_vertical_range, '--',
        2*[-2/24], daily_vertical_range, '--',
        2*[6/24], daily_vertical_range, '--',
        2*[-6/24], daily_vertical_range, '--',
        2*[4/24], daily_vertical_range, '--',
        2*[-4/24], daily_vertical_range, '--',
        2*[1], daily_vertical_range, '--',
        2*[-1], daily_vertical_range, '--',
        2*[2], daily_vertical_range, '--',
        2*[-2], daily_vertical_range, '--',
        2*[3], daily_vertical_range, '--',
        2*[-3], daily_vertical_range, '--',
        2*[4], daily_vertical_range, '--',
        2*[-4], daily_vertical_range, '--',
        2*[5], daily_vertical_range, '--',
        2*[-5], daily_vertical_range, '--',
        2*[6], daily_vertical_range, '--',
        2*[-6], daily_vertical_range, '--',
        2*[7], daily_vertical_range, '--',
        2*[-7], daily_vertical_range, '--',
        [0, 10], 2*[0], '--')
    plt.title(data_type + ' Daily Periodicity')
    plt.xlabel('Time-shift (Days)')
    plt.ylabel('Cross-correlation')
    plt.xscale('log')
    plt.show()
    i += 1


#%%
for data_type in TEMPORAL_CORRELATIONS:
    avg_correlation = np.average(TEMPORAL_CORRELATIONS[data_type], axis=0)
    print(data_type, 'Average Correlation:', np.average(avg_correlation))
    print(data_type, 'RMS Correlation:',
          np.sqrt(np.mean(avg_correlation**2)))
    plt.hist(avg_correlation, bins=[n/10 for n in range(-10, 11)])
    plt.title(data_type + ' Temporal Correlations')
    plt.show()

#%%
for data_type in TEMPORAL_CORRELATIONS:
    avg_correlation = np.average(TEMPORAL_CORRELATIONS[data_type], axis=0)
    daily_timestamps = (
        [zero_shift_timestamp + n * DAYS_TO_MINUTES // 5 for n in range(-1, 2)])
    daily_timestamps = daily_timestamps[:1] + daily_timestamps [2:]
    daily_corr = np.array(
        [avg_correlation[timestamp] for timestamp in daily_timestamps])
        
    print(data_type, 'Average Correlation:', np.average(daily_corr))
    print(data_type, 'RMS Correlation:',
          np.sqrt(np.mean(daily_corr**2)))
    plt.hist(daily_corr, bins=[n/10 for n in range(-10, 11)])
    plt.title(data_type + ' Temporal Correlations')
    plt.show()

#%%
