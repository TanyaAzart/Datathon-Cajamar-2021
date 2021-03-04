'''
This code uses previously found SARIMAX models to forecast future sales for stock items in the test set
'''

# import libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# check for incorrect data chaining
pd.options.mode.chained_assignment = 'raise'

# import labeled and unlabeled data sets
data_train = pd.read_csv('Modelar_UH2021.txt', sep='|', low_memory=False)
data_test = pd.read_csv('Estimar_UH2021.txt', sep='|', low_memory=False)

# select columns needed for further time series modelling
data = data_train.loc[:,['fecha','id','dia_atipico','campaña','unidades_vendidas']]
data_t = data_test.loc[:,['fecha','id','dia_atipico','campaña']]

# eliminate duplicates
data = data.drop_duplicates()
data_t = data_t.drop_duplicates()

# change dates data type to 'timestamps' 
data['fecha']=pd.to_datetime(data['fecha'], format="%d/%m/%Y %H:%M:%S")

# unique stock codes in the test dataset
codes_test = data_test['id'].unique()

'''
The following chunk of code is to be executed only if it hasn't been executed earlier!
It creates a folder with data files for individual stock codes.
'''
# create and save separate files for each stock code to be predicted
for each in codes_test:
    df = data[data['id']==each].copy()
    path = './files/' +str(each)+'.csv'
    df.to_csv(path, index=False)

'''
END of chunk
'''

# function for getting time series for individual stock code
def get_ts (stock_code):

    # read previously saved data for an individual stock code
    path = './files/' +str(stock_code)+'.csv'   
    df = pd.read_csv(path)
    
    # read data for the stock code in the test set
    df_t = data_t[data_t['id']==stock_code].copy()
    
    
    # create time series dataframes for the stock code
    df_ = pd.DataFrame(index=pd.date_range('2015-06-01', periods=488, freq='D'))
    df_t_ = pd.DataFrame(index=pd.date_range('2016-10-01', periods=92, freq='D'))

    # mark duplicates to eliminate
    df.loc[:,'delete']=0
    df_t.loc[:,'delete']=0
    
    fechas_dup = pd.Series(df['fecha'])[pd.Series(df['fecha']).duplicated()].values
    for fecha in fechas_dup:
        index = df[(df['fecha'] ==fecha ) & (df['campaña']==0)].index
        df.loc[index, 'delete']=1

    fechas_dup_t = pd.Series(df_t['fecha'])[pd.Series(df_t['fecha']).duplicated()].values
    for fecha in fechas_dup_t:
            index = df_t[(df_t['fecha'] ==fecha ) & (df_t['campaña']==0)].index
            df_t.loc[index, 'delete']=1

    # eliminate duplicates
    index_names = df[df['delete'] == 1 ].index 
    df.drop(index_names, inplace = True)
    
    index_names = df_t[df_t['delete'] == 1 ].index 
    df_t.drop(index_names, inplace = True)

    # change index to timestamp
    df.index=df.loc[:,'fecha']
    df.index = pd.to_datetime(df.index)

    df_t.index=df_t.loc[:,'fecha']
    df_t.index = pd.to_datetime(df_t.index)    

    # for each day of promotion ('campaña'=1), change'dia_atipico' to zero to have these codes independent 
    for i in range(len(df)):
        if (df.loc[df.index[i],'campaña']==1):
            df.loc[df.index[i],['dia_atipico']]=0   

    for i in range(len(df_t)):
        if (df_t.loc[df_t.index[i],'campaña']==1):
            df_t.loc[df_t.index[i],['dia_atipico']]=0     
          
            
    # eliminate columns which are no longer required
    df = pd.DataFrame(df[['unidades_vendidas','campaña','dia_atipico']])
    df_t = pd.DataFrame(df_t[['campaña','dia_atipico']])

    # add time series to their dataframes, fill in the gaps with zeroes
    ts = pd.concat([df_,df], axis=1)
    ts = ts.fillna(value=0)
    
    ts_t = pd.concat([df_t_,df_t], axis=1)
    ts_t = ts_t.fillna(value=0)
    
    # limit time series to the last year of observations
    ts=ts.iloc[-365:,:]

    return (ts, ts_t)

# function for forecasting
def forecast (stock_code, ts_train, ts_test, order, seasonal_order):

    # fit the model using labeled time series
    model = SARIMAX(ts_train['unidades_vendidas'], exog=ts_train[['campaña', 'dia_atipico']],
                    order=order, seasonal_order = seasonal_order).fit()
    
    # make predictions for test time series
    start=len(ts_train)
    end=len(ts_train)+len(ts_test)-1
    predictions = model.predict(start=start, end=end, exog=ts_test[['campaña', 'dia_atipico']]).astype(int)
    
    
    # change negative forecast values to zero
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
    
   
    # add predicted values to dataframe
    df_pred = pd.DataFrame(index=ts_test.index)
    df_pred.loc[:,'id']=stock_code
    df_pred.loc[:,'unidades_vendidas'] = predictions.to_list()
    
    return (df_pred)

# FORECAST FOR TEST PERIOD

# read parameters of the models found previously
parameters =  pd.read_csv('sarima_res.csv')

# create a dictionary with parameters
params ={}

for i in range(len(parameters)):
    id = parameters.iloc[i,0]
    order = parameters.iloc[i,1]
    seasonal_order = parameters.iloc[i,2]
    params[id]=[order, seasonal_order]


# make a dataframe for collecting predictions
sarima_pred = pd.DataFrame(columns=['id','unidades_vendidas'])

for each in codes_test:

    # read model parameters for stock code
    order = eval(params[each][0])
    seasonal_order = eval(params[each][1])
    
    # get train and test timeseries for the stock code
    ts, ts_t = get_ts(each)

    # get forecast for the stock code
    forecast_df = forecast(each,ts, ts_t, order, seasonal_order)

    # add forecast to dataframe 
    sarima_pred = pd.concat([sarima_pred, forecast_df])

    # print progress line
    print('Item number: '+str(len(sarima_pred)) + ', Item code: '+ str(each))

# save forecast dataframe as file
sarima_pred.ro_csv('sarima_forecast.csv', index=False)