#%% [markdown]
# # Baseline 1 - Persistent Model

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

#%%
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
# split into train and test sets
X = cpu_spatial_sample[0]
train_size = int(len(X) * 0.66)
train, test = X[:train_size], X[train_size:]
train_X, train_y = train[:-1], train[1:]
test_X, test_y = test[:-1], test[1:]
 
# persistence model
def model_persistence(x):
    return x
 
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = sqrt(mean_squared_error(test_y, predictions))
print('RMSE: %.3f' % test_score)
 
# plot predictions and expected results
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.axis([7990, 8000, 0, 1])
plt.show()

#%%