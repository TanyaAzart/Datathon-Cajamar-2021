'''
This code produces Prophet models for each stock item in the test set, makes predictions for validation period and calculates predictions errors.
It also makes predictions for the test period.
'''

# import libararies
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet

# check for incorrect data chaining
pd.options.mode.chained_assignment = 'raise'

# read data sets 
data_test = pd.read_csv('Estimar_UH2021.txt', sep='|', low_memory=False)
data_train = pd.read_csv('Modelar_UH2021.txt', sep='|', low_memory=False)

# select columns needed for further time series modelling
data = data_train.loc[:,['fecha','id','visitas','dia_atipico','campaña','unidades_vendidas']]
data_t = data_test.loc[:,['fecha','id','visitas','dia_atipico','campaña']]

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

# function for data preprocessing
def get_ts (stock_code):
    
    # read previously saved labeled data for an individual stock code
    path = './files/' +str(stock_code)+'.csv'   
    df = pd.read_csv(path)
    
    # read unlabeled data for the stock code in the test set
    df_t = data_t[data_t['id']==stock_code].copy()
    
    
    # create a time series dataframe for the stock code
    df_ = pd.DataFrame(index=pd.date_range('2015-06-01', periods=488, freq='D'))

    # mark duplcates to eliminate in both sets
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
            
            
    # eliminate the columns which are not required
    df = pd.DataFrame(df[['unidades_vendidas','visitas','campaña','dia_atipico']])

    # add labeled time series to its dataframe, fill in the gaps with zeroes
    ts = pd.concat([df_,df], axis=1)
    ts = ts.fillna(value=0)
    
    return (ts, df_t)

def modeling_prophet(ts, scale=0.05):
    # add columns required by Prophet
    ts['ds']=ts.index
    ts['y']=ts.loc[:,'unidades_vendidas']
    
    # divide data in train and validation sets
    train = ts.iloc[:-90,:]
    test = ts.iloc[-90:,:]

    # make and fit a model with three additional regressors
    m = Prophet(changepoint_prior_scale=scale)
    m.add_regressor('campaña')
    m.add_regressor('dia_atipico')
    m.add_regressor('visitas')
    m.fit(train)

    # make predictions for validation period
    future = m.make_future_dataframe(periods=len(test), freq='D')
    future.loc[:,'campaña']=ts.loc[:,'campaña'].to_list()
    future.loc[:,'dia_atipico']=ts.loc[:,'dia_atipico'].to_list()
    future.loc[:,'visitas']=ts.loc[:,'visitas'].to_list()
    forecast = m.predict(future)
       
    # extract predictions for validation period
    predictions = forecast.iloc[-len(test):]['yhat'].astype(int)
    
    # eliminate negative values from predictions
    for i in range(predictions.index.start, predictions.index.stop):
        if predictions.loc[i] < 0:
            predictions.loc[i] = 0
    
    # calculate prediction error
    mae = mean_absolute_error(test['unidades_vendidas'], predictions)
    
    # calculate the proportion MAE/STD
    coef = mae / (test['unidades_vendidas'].std()+0.001)
    
    return (mae, coef)
    
def forecast_prophet(stock_code,ts, df_t, scale=0.05):

    # create a time series for regressors
    columns = ['dia_atipico','campaña', 'visitas']
    ts_t = pd.concat([ts[columns], df_t[columns]])
    ts_t.fillna(value=0, inplace=True)
    

    # make and fit a model with three additional regressors
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
    predictions.index = predictions['ds']    
    
    # change negative values to zeros
    for i in range(len(predictions)):
        if predictions.iloc[i,1] < 0:
            predictions.iloc[i,1] = 0
    
    # create dataframe for predictions
    pred_lines = pd.DataFrame(columns=['id','unidades_vendidas'])
    pred_lines['unidades_vendidas']=predictions['yhat']
    pred_lines['id']=stock_code    
    
    return (pred_lines)

#### MODELING WITH PROPHET AND KEEPING RESULTS

# Set dictionary for keeping model parameters
results_prophet = {}

for each in codes_test:

    ts = get_ts(each)[0]

    mae, coef = modeling_prophet(ts, scale=0.05)

    # add results to the dictionary
    results_prophet[each]=(mae, coef)
    
    # print the progress line
    print('Item number: '+str(len(results_prophet)) + ', Item code: '+ str(each)+ ', MAE: '+str(mae)+', Coef: '+ str(coef))

# save modeling results to dataframe
prophet_res=pd.DataFrame(columns=['id','mae','coef'])

id=[]
mae=[]
coef=[]

for each in results_prophet:
    id.append(each)
    mae.append(results_prophet[each][0])
    coef.append(results_prophet[each][1])
    
prophet_res['id']=id
prophet_res['mae']=mae
prophet_res['coef']=coef

# save dataframe to file
prophet_res.to_csv('prophet_results.csv', index=False) 

#### FORECASTING WITH PROPHET FOR TEST PERIOD

# Set dictionary for keeping forecasts
pred_prophet = pd.DataFrame(columns=['id', 'unidades_vendidas'])

for each in codes_test:

    ts, df_t = get_ts(each)

    pred_lines = forecast_prophet(each, ts, df_t, scale=0.05)

    # add predictions to the dataframe
    pred_prophet= pd.concat([pred_prophet, pred_lines])

    # print progress line
    print('Item number: '+str(len(pred_prophet)) + ', Item code: '+ str(each))

# save forecast to file
pred_prophet.to_csv('prophet_pred.csv', index=False)