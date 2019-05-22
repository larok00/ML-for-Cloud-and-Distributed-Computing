#%% [markdown]
# # Baseline 2 - Expanding Window Model

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import mean, median

#%%
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
# split into train and test sets
X = CPU_SPATIAL_SAMPLE[0]
train_size = int(len(X) * 0.66)
train, test = X[:train_size], X[train_size:]
train_X, train_y = train[:-1], train[1:]
test_X, test_y = test[:-1], test[1:]
 
# walk-forward validation
history = [x for x in test_X]
predictions = list()
for i in range(len(test_y)):
    # make prediction
    yhat = mean(history)
    predictions.append(yhat)
    # observation
    history.append(test_y[i])

# report performance
rmse = sqrt(mean_squared_error(test_y, predictions))
print('RMSE: %.3f' % rmse)

plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()

#%%