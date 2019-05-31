#%% [markdown]
# # SARIMA

#%% [markdown]
# ## Setup

#%%
from clustering_methods import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

#%%
np.random.seed(19)
SAMPLE_SIZE = NO_OF_MACHINES

for data_type in DATA:
    SAMPLES[data_type] = DATA[data_type][:SAMPLE_SIZE]

#%% [markdown]
# ## Data Formatting

#%%
CLUSTERS = average_all_machines(SAMPLES)
print(CLUSTERS['CPU'].shape)

#%%
rng = pd.date_range(start='05/01/2011 19:00', periods=NO_OF_TIMESTAMPS,
                    freq='5min', tz='US/Eastern')

data = np.full((NO_OF_TIMESTAMPS, 0), np.nan)
columns = list()
for data_type in CLUSTERS:
    columns.append(data_type)
    data = np.append(data, CLUSTERS[data_type].T, axis=1)

df = pd.DataFrame(data, index=rng, columns=columns)
print(df.head())

#%%
train_size, validation_size, test_size = tuple(
    [ int(x * l) for x in (0.7, 0.2, 0.1) for l in [len(df)]])
train, validation, test = (
    df[:train_size],
    df[train_size : train_size + validation_size],
    df[train_size + validation_size : train_size + validation_size + test_size])
train_X, train_y = train[:-1], train[1:]
validation_X, validation_y = validation[:-1], validation[1:]
test_X, test_y = test[:-1], test[1:]
    
#%% [markdown]
# ## Models

#%% [markdown]
# Visit https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
# for help on models implemented.

#%%
def fit_model(model_instantiator, print_summary=False):
    results = list()
    for data_type in SAMPLES:
        # fit model
        model = model_instantiator(train[data_type])
        model_fit = model.fit()
        if print_summary:
            print(model_fit.summary())
        # make prediction
        pred = model_fit.predict(
            start=validation.index[0], end=validation.index[-1])
        test_score = sqrt(mean_squared_error(validation[data_type], pred))
        results.append(test_score)
        plt.plot(train[data_type])
        plt.plot(validation[data_type])
        plt.plot(pd.Series(pred, index=validation.index))
        plt.xticks(
            ticks=[plt.xticks()[0][i] for i in range(0,
                                                     len(plt.xticks()[0]), 2)])
        plt.show()
    return results

#%% [markdown]
# ### AR

#%%
results = fit_model(AR)
print(results)

#%% [markdown]
# ### MA

#%%
results = fit_model(lambda data : ARMA(data, order=(0, 1)))
print(results)

#%% [markdown]
# ### ARMA

#%%
results = fit_model(lambda data : ARMA(data, order=(2, 1)))
print(results)

#%% [markdown]
# ### ARIMA

#%%
results = fit_model(lambda data : ARIMA(data, order=(7, 0, 1)))
print(results)

#%% [markdown]
# ### SARIMA

#%%
results = fit_model(lambda data : SARIMAX(data, order=(1, 1, 1),
                                         seasonal_order=(1, 1, 1, 1)))
print(results)

#%% [markdown]
# ### SES

#%%
results = fit_model(SimpleExpSmoothing)
print(results)

#%% [markdown]
# ### HWES

#%%
results = fit_model(ExponentialSmoothing)
print(results)

#%%
