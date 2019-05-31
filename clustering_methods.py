#%% [markdown]
# # Clustering Methods

#%% [markdown]
# ## Setup

#%%
import numpy as np
import pandas as pd
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
