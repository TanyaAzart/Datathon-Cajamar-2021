'''
This code has been developed for data exploration and preprocessing for Datathon Cajamar 2021, Spain. 

'''
# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

# check for incorrect data chaining
pd.options.mode.chained_assignment = 'raise'

# stipulate default format for graphs
from pylab import rcParams
rcParams['figure.figsize'] = 15,5

# DATA EXPLORATION AND PREPROCESSING

# import labeled and unlabeled datasets
data_train = pd.read_csv('Modelar_UH2021.txt', sep='|', low_memory=False)
data_test = pd.read_csv('Estimar_UH2021.txt', sep='|', low_memory=False)

# copy data sets for further changes
data = data_train.copy()
data_t = data_test.copy()

# check for missing values in labeled set
data.isna().any()

# check for missing values in unlabeled dataset
data_t.isna().any()

# fill in empty cells with closest previous prices
data['precio'].fillna(method='backfill', inplace=True)

# get description for data types in the labeled set
data.info()

# get description for data types in the test set
data_t.info()

# change data type for prices from 'string' to 'float'
data['precio'] = data['precio'].replace(',','.', regex=True).astype(float)
data_t['precio'] = data_t['precio'].replace(',','.', regex=True).astype(float)

# change data type for dates to 'timestamps' 
data['fecha']=pd.to_datetime(data_train['fecha'], format="%d/%m/%Y %H:%M:%S")
data['fecha'].dtype

'''
The format of dates now is YYYY-MM-DD. 
The range of dates in the labeled set is from June 1, 2015 till September 30, 2016 
'''

# select columns needed for further time series analysis and modeling
data = data.loc[:,['fecha','id','dia_atipico','campaña','visitas','precio','unidades_vendidas']]
data_t = data_t.loc[:,['fecha','id','dia_atipico','campaña','visitas','precio']]

# count duplicates in labeled dataset
data.duplicated(keep='first').sum()

# count duplicates in unlabeled data set
data_t.duplicated(keep='first').sum()

# eliminate duplicates
data = data.drop_duplicates()
data_t = data_t.drop_duplicates()

# count unique registers in labeled data set 
print("Number of unique registrations in the train dataset: "+ str(len(data)))

# count unique registers in unlabeled data set
print("Number of unique registrations in the test dataset: "+ str(len(data_t)))

# find out the number of unique stock codes in labeled data set
print(" Number of stock codes in the train dataset: " + str(len(data['id'].unique())))

# find out the number of unique stock codes in unlabeled data set
print(" Number of stock codes in the test dataset: " + str(len(data_t['id'].unique())))

''' 
Since time series should have only one observation for each day, check for duplicated dates
for each stock code 
'''
duplicates = []

for each in data_t['id'].unique():
    df = data[data['id']==each]
    if not df['fecha'].is_unique:
        duplicates.append(each)

print('Number of duplicates in labeled dataset: '+ str(len(duplicates)))

# collect all lines with duplicated dates in a dataframe for further investigation
err_entries = pd.DataFrame()

for each in duplicates:
    
    df = data[data['id']==each].copy()
    fechas_dup = pd.Series(df['fecha'])[pd.Series(df['fecha']).duplicated()].values  
    
    for fecha in fechas_dup:
        df1 = df[df['fecha']==fecha].copy()
        err_entries = pd.concat([err_entries, df1])  

# view some duplicated entries
err_entries.head()

# view all problematic lines for a stock code, for example 327526
err_entries[err_entries['id']==327526]

'''
The duplicated entries have all fields identical except 'campaña', 
the lines with 'campaña'=0 are to be deleted and with 'campaña'=1 kept,
since the sales numbers are quite high - most likely these are days of promotion
'''

# create a dataframe for the stock code 327526 time series for further investigation
df_ = pd.DataFrame(index=pd.date_range('2015-06-01', periods=488, freq='D'))

df = data[data['id']==327526].copy()

# mark duplicated lines (with two values for 'campaña' field for the same date)
fechas_dup = pd.Series(df['fecha'])[pd.Series(df['fecha']).duplicated()].values  
    
for fecha in fechas_dup:
    index = df[(df['fecha'] ==fecha ) & (df['campaña']==0)].index
    df.loc[index, 'delete']=1

# eliminate lines with 'campaña'=0           
index_names = df[df['delete'] == 1 ].index 
df.drop(index_names, inplace = True)


# change index to timestamp
df.index=df.loc[:,'fecha']
df.index = pd.to_datetime(df.index)

# for each day of  promotion ('campaña'=1), change 'dia_atipico' to zero to keep the fields independent 
for i in range(len(df)):
    if (df.loc[df.index[i],'campaña']==1):
        df.loc[df.index[i],['dia_atipico']]=0 
        

# drop unnecessary columns
df = pd.DataFrame(df[['unidades_vendidas', 'visitas', 'campaña','dia_atipico','precio']])
      
# add the time series to its dataframe and fill in the gaps corresponding to dates
# missing in the original dataset        
ts = pd.concat([df_,df], axis=1)
ts.fillna(value=0, inplace=True)


# check for uniqueness of dates in the time series
ts.index.is_unique

# view and check that for days of promotion -> if 'campaña'= 1, 'dia_atipico'=0 
ts[ts['campaña']!=0]

# TIME SERIES ANALYSIS

# plot time series for stock code 327526
ts['unidades_vendidas'].plot()

# limit time series to the last year of observations
ts=ts.iloc[-365:,:]
ts['unidades_vendidas'].plot()

# plot the series indicating days of promotion and unusual sales
ax = ts['unidades_vendidas'].plot()
  
for day in ts[ts['campaña']==1].index:
              ax.axvline(x=day, color='red', alpha=0.5)  

for day in ts[ts['dia_atipico']==1].index:
              ax.axvline(x=day, color='green', alpha=0.5)  
        
for day in ts[ts['dia_atipico']==-1].index:
              ax.axvline(x=day, color='black', alpha=0.5)  

# decompose time series and plot components
ts_dec = seasonal_decompose(ts['unidades_vendidas'])
ts_dec.plot();

# check for autocorrelation in the series: plot ACF
plot_acf(ts['unidades_vendidas'], lags=100);

# check for autocorrelation in the series: plot PACF
plot_pacf(ts['unidades_vendidas'], lags=100);

# check for stationarity -> Augmented Dickey-Fuller test
adfuller(ts['unidades_vendidas'])

# SARIMAX MODELING

'''
Since there is autocorrelation in the series, SARIMAX metodology can be a good option for modeling and forecasting.
Different exogenous variables have been tried, including 'visitas' and 'precio' but according to AIC the best model is the one with two exogenuos variables: 'campaña' and 'dia_atipico'
The code below is for this case.
'''

# find the best model with two exogenous variables with autoarima module
res = auto_arima(ts['unidades_vendidas'],exogenous=ts[['campaña','dia_atipico']], m=7, 
trace=True, suppress_warnings=True, enforce_invertibility=False)

order = res.order
seasonal_order = res.seasonal_order

# view the best model parameters
sar = SARIMAX(ts['unidades_vendidas'], 
          exogenous=ts[['campaña','dia_atipico']],    
                                order=order,
             seasonal_order = seasonal_order, ).fit()
sar.summary()

# plot residuals diagnostics
resid = sar.resid[sar.loglikelihood_burn:]
sar.plot_diagnostics(lags=50,figsize = (20,10),);

# divide labeled data in train and validation sets
train = ts[:-90]
valid = ts[-90:]

# fit the model using train set
sar = SARIMAX(train['unidades_vendidas'], 
              exogenous=ts[['campaña','dia_atipico']],
                                order=order,
             seasonal_order=seasonal_order).fit()
sar.summary()

# make and plot predictions for validation dataset 
start=len(train)
end=len(train)+len(valid)-1

predictions = sar.predict(start=start, end=end).rename('SARIMAX')
valid['unidades_vendidas'].plot(legend=True)
predictions.plot(legend=True, title="SARIMAX");

# calculate prediction error
mae = mean_absolute_error(valid['unidades_vendidas'], predictions)
print('MAE, SARIMAX: '+str(mae))

# PROPHET MODELING

'''
The best model has been obtained adding three regressors: 'visitas', 'campaña', 'dia_atipico'.
The code below is for this case.
'''

# add to time series two columns required by Prophet library
ts['ds']=ts.index
ts['y']=ts.loc[:,'unidades_vendidas']

# update train and validation sets
train = ts[:-90]
valid = ts[-90:]

# make a model with three additional regressors 
m = Prophet(changepoint_prior_scale=0.05)
m.add_regressor('campaña')
m.add_regressor('dia_atipico')
m.add_regressor('visitas')
m.fit(train)

future = m.make_future_dataframe(periods=len(valid), freq='D')
future['campaña']=ts['campaña'].to_list()
future['dia_atipico']=ts['dia_atipico'].to_list()
future['visitas']=ts['visitas'].to_list()

forecast = m.predict(future)

# plot the time series components
fig = m.plot_components(forecast)

# plot time series produced by the model together with confidence intervals and tendency line
fig = m.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# change negative predicted values to zeros
predictions = forecast.iloc[-90:][['ds','yhat']]
predictions['yhat']=predictions['yhat'].astype(int)
predictions.index = predictions['ds']

for i in range(len(predictions)):
    if predictions.iloc[i,1] < 0:
        predictions.iloc[i,1] = 0

# plot the series in the validation period: predictions vs actuals
xlim=[predictions.index[0], predictions.index[-1]]
ax = forecast.plot(x='ds',y='yhat', label='Predictions', legend=True)
valid.plot(x='ds',y='y', title='Predictions vs Actuals',xlim=xlim, label='Actuals', legend=True, ax=ax);

# calculate prediction error
mae = mean_absolute_error(valid['y'], predictions['yhat'])

print('MAE, Prophet: '+str(mae))

'''
The error produced by Prophet is significantly lower than SARIMAX error.
Besides, modeling with Prophet is less time consuming.
That's why the forecast is to be done using Prophet methodology.
'''