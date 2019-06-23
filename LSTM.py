#%% [markdown]
# # LSTM

#%% [markdown]
# ## Setup

#%%
from clustering_methods import *
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import time

#%%
np.random.seed(19)
SAMPLE_SIZE = NO_OF_MACHINES

for data_type in DATA:
    SAMPLES[data_type] = DATA[data_type][:SAMPLE_SIZE]

#%% [markdown]
# ## Data Formatting

#%%
TRAIN, TEST, VALIDATION = dict(), dict(), dict()
for data_type in SAMPLES:
    sample = SAMPLES[data_type]
    TRAIN[data_type], TEST[data_type], VALIDATION[data_type] = (
        sample[: int(0.6 * len(sample))],
        sample[int(0.6 * len(sample)) : int(0.8 * len(sample))],
        sample[int(0.8 * len(sample)) : len(sample)])

#%%
CLUSTERS = average_all_machines(TRAIN)
print(CLUSTERS['CPU'].shape)

#%%
rng = pd.date_range(start='05/01/2011 19:00', periods=NO_OF_TIMESTAMPS,
                    freq='5min', tz='US/Eastern')

columns = list()
dataframes = list()
for data_type in CLUSTERS:
    columns.append(data_type)
    data = CLUSTERS[data_type].T
    dataframes.append(pd.DataFrame(data, index=rng))
df = pd.concat(dataframes, keys=columns, axis=1)
print(df.head())

#%%
# multivariate output stacked lstm example
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences.iloc[i:end_ix, :], sequences.iloc[end_ix, :]
        X.append(seq_x.values)
        y.append(seq_y.values)
    return np.array(X), np.array(y)

#%%
# choose a number of time steps
n_steps = 3 * 24 * 60 // 5
# convert into input/output
X, y = split_sequences(df.xs('CPU', axis=1), n_steps)

# the dataset knows the number of features, e.g. 1
n_features = X.shape[2]

#%%
start = time.process_time()
# define model
model = Sequential()
model.add(LSTM(5, activation='relu', return_sequences=True,
               input_shape=(n_steps, n_features)))
model.add(LSTM(2, activation='relu'))
model.add(Dense(n_features, activation='relu'))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=10, verbose=0)
elapsed_time = time.process_time() - start

#%%
# demonstrate prediction
print(elapsed_time)
n = 3
x_input = X[n]
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
print(y[n])
print(sqrt(mean_squared_error(yhat[0], y[n])))

#%%
