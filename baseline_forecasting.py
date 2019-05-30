#%% [markdown]
# # Baseline Forecasting

#%% [markdown]
# ## Setup

#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

np.random.seed(19)
DATA_TYPES = {'CPU', 'MEM'}

#%%
DATA = dict()

#%%
for data_type in DATA_TYPES:
    if data_type not in DATA:
        DATA[data_type] = np.load('google-cpu-full.npy')
        np.random.shuffle(DATA[data_type])

second_data_type = None
for first_data_type in DATA:
    if second_data_type is not None:
        assert DATA[first_data_type].shape == DATA[second_data_type].shape
    second_data_type = first_data_type

NO_OF_MACHINES = DATA[first_data_type].shape[0]
NO_OF_TIMESTAMPS = DATA[first_data_type].shape[1]
DAYS_TO_MINUTES = 24*60

SAMPLE_SIZE = 1250
SAMPLES = dict()
for data_type in DATA:
    SAMPLES[data_type] = DATA[data_type][:SAMPLE_SIZE]
    
#%% [markdown]
# ## Models
# Visit https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
# for help on models implemented.

#%%
def fit_model(model_instantiator):
    results = list()
    for data_type in SAMPLES:
        data_one = SAMPLES[data_type][0]
        # fit model
        model = model_instantiator(data_one)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(data_one), len(data_one))
        results.append(yhat)
    return results

#%% [markdown]
# ### AR

#%%
yhat = fit_model(AR)
print(yhat)

#%% [markdown]
# ### MA

#%%
yhat = fit_model(lambda data : ARMA(data, order=(0, 1)))
print(yhat)

#%% [markdown]
# ### ARMA

#%%
yhat = fit_model(lambda data : ARMA(data, order=(2, 1)))
print(yhat)

#%% [markdown]
# ### ARIMA

#%%
yhat = fit_model(lambda data : ARIMA(data, order=(1, 1, 1)))
print(yhat)

#%% [markdown]
# ### SARIMA

#%%
yhat = fit_model(lambda data : SARIMAX(data, order=(1, 1, 1),
                                         seasonal_order=(1, 1, 1, 1)))
print(yhat)

#%% [markdown]
# ### SES

#%%
yhat = fit_model(SimpleExpSmoothing)
print(yhat)

#%% [markdown]
# ### HWES

#%%
yhat = fit_model(ExponentialSmoothing)
print(yhat)

#%%
