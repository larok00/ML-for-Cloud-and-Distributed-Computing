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
#grid search
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import time
 
# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test, n_validation):
    return (
        data[:-n_test-n_validation],
        data[-n_test-n_validation:-n_validation],
        data[-n_validation:])
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, n_validation, cfg):
    predictions = list()
    # split dataset
    train, test, validation = train_test_split(data, n_test, n_validation)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error
 
# score a model, return None on failure
def score_model(data, n_test, n_validation, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, n_validation, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, n_validation, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)
 
# grid search configs
def grid_search(data, cfg_list, n_test, n_validation, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, n_validation, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, n_validation, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = range(1) #range(6)
    d_params = range(1) #range(3)
    q_params = range(1) #range(6)
    t_params = ['n'] #,'c','t','ct']
    P_params = range(1) #[0, 1, 2]
    D_params = range(1) #[0, 1]
    Q_params = range(1) #[0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models
 
#%%
start = time.process_time()
# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
# data split
n_test = n_validation = len(data)//5
# model configs
cfg_list = sarima_configs()
# grid search
scores = grid_search(data, cfg_list, n_test, n_validation)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)
elapsed_time = time.process_time() - start
print('Time:', elapsed_time)


#%%
# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
 
# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)
 
# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models
 
# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# data split
n_test = 4
# model configs
cfg_list = sarima_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)


#%%
