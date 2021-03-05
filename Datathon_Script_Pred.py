'''
This code produces Prophet models for each stock item in the test set and forecasts for the test period.
'''

# import libraries
import pandas as pd
import numpy as np
from fbprophet import Prophet

# read data sets 
data_past = pd.read_csv('Modelar_UH2021.txt', sep='|', low_memory=False)
data_test = pd.read_csv('Estimar_UH2021.txt', sep='|', low_memory=False)

# select columns needed for further time series modelling
data = data_past.loc[:,['fecha','id','visitas','dia_atipico','campaña','unidades_vendidas']]
data_t = data_test.loc[:,['fecha','id','visitas','dia_atipico','campaña']]

# eliminate duplicates
data = data.drop_duplicates()
data_t = data_t.drop_duplicates()

# change dates data type to 'timestamps' 
data['fecha']=pd.to_datetime(data['fecha'], format="%d/%m/%Y %H:%M:%S")

# make list of the stock codes to forecast
codes_test = data_t['id'].unique()

# function for getting time series in appropriate format
# start_date : the earliest date in time series/past data
# periods: number of days in time series/past data + numbers of days to be forecasted
def get_ts (stock_code, start_date, periods):
    
    # read past/labeled data for an individual stock code
    df = data.loc[data['id']==stock_code].copy()
    
    # read unlabeled data for the stock code in the test set
    df_t = data_t[data_t['id']==stock_code].copy()
    
    # create a dataframe for the time series
    df_ = pd.DataFrame(index=pd.date_range(start_date, periods, freq='D'))

    # mark duplcated entries (two'campaña' values for the same day) to eliminate in both sets
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
            
    # eliminate columns which are not required
    df = pd.DataFrame(df[['unidades_vendidas','visitas','campaña','dia_atipico']])

    # add labeled time series to its dataframe, fill in the gaps with zeroes
    ts = pd.concat([df_,df], axis=1)
    ts = ts.fillna(value=0)
    
    # add columns required by Prophet
    ts['ds']=ts.index
    ts['y']=ts.loc[:,'unidades_vendidas']
    
    
    return (ts, df_t)

    
def forecast_prophet(stock_code,ts, df_t, scale=0.05):

    # create time series for regressors
    columns = ['dia_atipico','campaña', 'visitas']
    ts_t = pd.concat([ts[columns], df_t[columns]])
    ts_t.fillna(value=0, inplace=True)
    

    # make and fit model with three additional regressors
    m = Prophet(changepoint_prior_scale=scale)
    m.add_regressor('campaña')
    m.add_regressor('dia_atipico')
    m.add_regressor('visitas')
    m.fit(ts)

    # make predictions
    future = m.make_future_dataframe(periods=len(df_t), freq='D')
    future.loc[:,'campaña']=ts_t.loc[:,'campaña'].to_list()
    future.loc[:,'dia_atipico']=ts_t.loc[:,'dia_atipico'].to_list()
    future.loc[:,'visitas']=ts_t.loc[:,'visitas'].to_list()
    forecast = m.predict(future)
       
    # extract predictions for test period
    predictions = forecast.iloc[-len(df_t):][['ds','yhat']]
    predictions['yhat']=predictions['yhat'].astype(int)
    
    
    # change negative values to zeros
    for i in range(len(predictions)):
        if predictions.iloc[i,1] < 0:
            predictions.iloc[i,1] = 0
    
    # create dataframe for prediction lines
    pred_lines = pd.DataFrame(columns=['fecha','id','unidades_vendidas'])
    pred_lines['unidades_vendidas']=predictions['yhat']
    pred_lines['fecha']=predictions['ds']
    pred_lines['id']=stock_code  
        
    return (pred_lines)

#### FORECASTING WITH PROPHET FOR TEST PERIOD


def main (start_date='2015-06-01', periods_train=396, periods_test=92, scale=0.05):   
    
    # Set dictionary for keeping forecasts
    pred_prophet = pd.DataFrame()
    
    for each in codes_test:

        ts, df_t = get_ts(each, start_date=start_date, periods=periods_train + periods_test)

        pred_lines = forecast_prophet(each, ts, df_t, scale)
       
        # add predictions to the dataframe
        pred_prophet= pd.concat([pred_prophet, pred_lines])

        # print progress line
        print('Item number: '+str(np.where(codes_test==each)[0][0]+1) + ', Item code: '+ str(each))

    # save forecast to file
    pred_prophet.to_csv('prophet_forecast.csv', index=False)

main()