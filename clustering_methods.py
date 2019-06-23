#%% [markdown]
# # Clustering Methods

#%% [markdown]
# ## Setup

#%%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import time

#%%
np.random.seed(19)
DATA_DIR = 'data/'
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

#%% [markdown]
# ## Methods

#%% [markdown]
# ### Baseline - K=1 averaging

#%%
def average_all_machines(data):
    clusters = dict()

    for data_type in data:
        clusters[data_type] = np.array([np.average(data[data_type], axis=0)])
    return clusters

#%%
avg = average_all_machines(DATA)
print(avg['CPU'])

#%% [markdown]
# ### Unweighted KMeans

#%%
def kmeans(data, n_clusters, return_labels=False):
    clusters, labels = dict(), dict()

    for data_type in data:
        kmeans = KMeans(
                n_clusters=n_clusters, random_state=0).fit(data[data_type])

        clusters[data_type] = kmeans.cluster_centers_
        labels[data_type] = kmeans.labels_
    if return_labels:
        return clusters, labels
    else:
        return clusters

#%% [markdown]
# ### Weighted KMeans

#%%
PICKLE_DIR = 'pickles/'
pickle_in = open(PICKLE_DIR+"temporal_correlations.pickle", "rb")
TEMPORAL_CORRELATIONS = pickle.load(pickle_in)
avg_correlation = dict()
for data_type in TEMPORAL_CORRELATIONS:
    avg_correlation[data_type] = np.average(
        TEMPORAL_CORRELATIONS[data_type], axis=0)
ZERO_SHIFT_TIMESTAMP = NO_OF_TIMESTAMPS-1

def weighted_kmeans(data, n_clusters, return_labels=False):
    clusters, labels = dict(), dict()

    for data_type in data:
        if ZERO_SHIFT_TIMESTAMP > data[data_type].shape[1]:
            weights = avg_correlation[data_type][
                ZERO_SHIFT_TIMESTAMP - data[data_type].shape[1] :
                ZERO_SHIFT_TIMESTAMP
            ]
        else:
            weights = [0.01 for _ in range(data[data_type].shape[1])]
            weights[-ZERO_SHIFT_TIMESTAMP:] = (
                avg_correlation[data_type][:ZERO_SHIFT_TIMESTAMP])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            data[data_type] * weights)

        clusters[data_type] = kmeans.cluster_centers_ / weights
        labels[data_type] = kmeans.labels_
    if return_labels:
        return clusters, labels
    else:
        return clusters

#%%
def rmse(data, clusters, labels, yhat, index):
    test_score = dict()
    for data_type in data:
        deltas = np.average(data[data_type] - clusters[data_type][labels],
                            axis=1)
        preds = yhat[data_type][labels] + deltas
        true_vals = data[data_type][:, index]
        test_score[data_type] = sqrt(
                mean_squared_error(preds, true_vals))
    return test_score

#%%
print(avg_correlation)


#%%
