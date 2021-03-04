'''
This code produces SARIMAX models for each stock item in the test set, makes predictions for a validation period and calculates predictions errors
'''

# import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# check for incorrect data chaining
pd.options.mode.chained_assignment = 'raise'

# import labeled and unlabeled data sets
data_train = pd.read_csv('Modelar_UH2021.txt', sep='|', low_memory=False)
data_test = pd.read_csv('Estimar_UH2021.txt', sep='|', low_memory=False)

# select columns needed for further time series modelling
data = data_train.loc[:,['fecha','id','dia_atipico','campaña','unidades_vendidas']]

# eliminate duplicates
data = data.drop_duplicates()

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
def prep_data (stock_code):
    
    # read previously saved file for an individual stock code
    path = './files/' +str(each)+'.csv'   
    df = pd.read_csv(path)
    
    # create dataframe for the stock code's time series
    df_ = pd.DataFrame(index=pd.date_range('2015-06-01', periods=488, freq='D'))

    # mark duplcates to eliminate
    df.loc[:,'delete']=0
        
    fechas_dup = pd.Series(df['fecha'])[pd.Series(df['fecha']).duplicated()].values
    for fecha in fechas_dup:
        index = df[(df['fecha'] ==fecha ) & (df['campaña']==0)].index
        df.loc[index, 'delete']=1
    
    # eliminate duplicates
    index_names = df[df['delete'] == 1 ].index 
    df.drop(index_names, inplace = True)
    
    # change the index
    df.index=df.loc[:,'fecha']
    df.index = pd.to_datetime(df.index)

    # for each day of promotion ('campaña'=1), change 'dia_atipico' to zero to have these codes independent 
    for i in range(len(df)):
        if (df.loc[df.index[i],'campaña']==1):
            df.loc[df.index[i],['dia_atipico']]=0   

    # eliminate the columns which are not longer required
    df = pd.DataFrame(df[['unidades_vendidas', 'campaña','dia_atipico']])

    # add time series to its dataframe, fill in the gaps with zeroes
    ts = pd.concat([df_,df], axis=1)
    ts = ts.fillna(value=0)
    
    # limit the series to the last year of observations
    ts=ts.iloc[-365:,:]

    return ts


# function for modeling with SARIMAX
def modeling (ts):

    # divide data in train and validation sets
    train = ts.iloc[:-90,:]
    test = ts.iloc[-90:,:]

    # find best model for the series with 'autoarima'
    res = auto_arima(ts['unidades_vendidas'], exogenous=ts[['campaña', 'dia_atipico']], m=7,
                     suppress_warnings=True, enforce_invertibility=False)
    
    order = res.order
    seasonal_order = res.seasonal_order
    
    
    # fit the model using train set
    model = SARIMAX(train['unidades_vendidas'], exog=train[['campaña', 'dia_atipico']],
                    order=order,
                    seasonal_order = seasonal_order).fit()
    
    # make predictions for validation set
    start=len(train)
    end=len(train)+len(test)-1
    predictions = model.predict(start=start, end=end, exog=test[['campaña', 'dia_atipico']])
    
    
    # change negative values to zero
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
    
    # calculate prediction error
    mae = mean_absolute_error(test['unidades_vendidas'], predictions)
    
    # calculate proportion MAE/STD
    coef = mae / (test['unidades_vendidas'].std()+0.001)

    return (order, seasonal_order, mae, coef)

# FIND BEST MODELS AND SAVE PARAMETERS

# Set a dictionary for keeping found models' parameters
results_sarima = {}

# complete dictionary with found model's parameters for all stock codes
for each in codes_test:
    
    ts = prep_data(each)
    results_sarima[each] = modeling(ts)
    mae = results_sarima[each][2]
    coef =results_sarima[each][3]

    # print the progress line
    print('Item number: '+str(len(results_sarima)) + ', Item code: '+ str(each)+ ', MAE: '+str(mae)+', Coef: '+ str(coef))
    
# save models as dataframe
sarima_res=pd.DataFrame(columns=['id','order','seasonal_order','mae','coef'])

id=[]
order=[]
seasonal_order=[]
mae=[]
coef=[]

for each in results_sarima:
    id.append(each)
    order.append(results_sarima[each][0])
    seasonal_order.append(results_sarima[each][1])
    mae.append(results_sarima[each][2])
    coef.append(results_sarima[each][3])
    
sarima_res['id']=id
sarima_res['order']=order
sarima_res['seasonal_order']=seasonal_order
sarima_res['mae']=mae
sarima_res['coef']=coef

# save dataframe as a file
sarima_res.to_csv('sarima_models.csv', index=False) 



