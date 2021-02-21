#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import dask
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
# from dask_ml.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score, accuracy_score
# import dask_xgboost
from dask.distributed import Client
from dask.diagnostics import ProgressBar

# import matplotlib.pyplot as plt
#from dask_ml.preprocessing import DummyEncoder
import numpy as np
np.random.seed(seed = 101)
import xgboost as xgb; print('XGBoost Version:', xgb.__version__)
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import SGDRegressor
# from scipy.stats import uniform, loguniform
# from sklearn.ensemble import RandomForestRegressor
import joblib


# In[2]:


print('Dask Version:',dask.__version__)
jobid=os.getenv('SLURM_JOBID')
client= Client(scheduler_file='scheduler_%s.json' %jobid)

print(f'Job_id:{jobid}')


# In[3]:


print (client)
client


# In[4]:


big_df = pd.DataFrame()
# all_files = [x for x in os.listdir("synapse_DREAM_challenge/CELL_LINES/") if ".csv" in x and x != "subchallenge_1_template_data.csv"  ]
all_files = [x for x in os.listdir("data") if ".csv" in x and x != "subchallenge_1_template_data.csv"  ]


target_cell_lines = ['AU565', 'EFM19', 'HCC2218', 'LY2', 'MACLS2', 'MDAMB436']


target_genes = ['p.ERK', 'p.Akt.Ser473.', 'p.S6', 'p.HER2', 'p.PLCg2']


for m in all_files:
    #print (m)
    df_temp = dd.read_csv("data/%s"%m)
    big_df = dd.concat([big_df,df_temp],axis=0,sort=True)


train_df = big_df[~big_df["cell_line"].isin(target_cell_lines)]

X = train_df.drop(target_genes,axis=1)

y = train_df[target_genes]

my_target_ditionary = {'ERK':'p.ERK', 'AKT':'p.Akt.Ser473.', 'S6':'p.S6', 'HER':'p.HER2', 'PLCG2':'p.PLCg2'}
    
# In[4]:


print ("categorize")

X = X.categorize(columns=["treatment"])

print ("dummies")

my_dummies = dd.get_dummies(X["treatment"])


X= X.drop(['treatment', 'cell_line', 'time', 'cellID', 'fileID'],axis=1)

X = X.persist()


# In[5]:


print("fill na with mean")
# y = y.fillna(0)

for m in y.columns :
    y[m]= y[m].fillna(y[m].mean() )
    


# In[6]:


print("Seach for nan , it should return 0")
for m in y.columns :
    a= len(y[y[m].isna()== True])
    print(m,a)


# In[7]:


y = y.persist()


# In[8]:



print("Scaler")
scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)

# In[7]:

X_train, X_test, y_train, y_test = train_test_split(X_scaled , 
                                                        y["p.ERK"], 
                                                        test_size=0.33, 
                                                        random_state=101,shuffle=True)


# In[9]:


model = xgb.XGBRegressor(objective='reg:squarederror',verbose=2,  n_jobs=-1,nthread=-1,seed=101)


# In[10]:


with joblib.parallel_backend('dask'):
    model.fit(X_train, y_train)


# In[11]:


print ("base model")
print (model)


# In[13]:


y_hat = model.predict( X_test)
y_hat


# In[15]:


print("base model")
r=r2_score(y_test.compute(), y_hat)
mae=mean_absolute_error(y_test.compute(), y_hat)
mse=mean_squared_error(y_test.compute(), y_hat)
print ("R^2:",r)
print ("MAE:",mae)
print ("MSE:",mse)


# In[17]:


model2 = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=1000, nthread=-1, seed=101,n_jobs=-1,verbosity=2
                         ,scale_pos_weight=1,learning_rate =0.1,max_depth=5,gamma=0,min_child_weight=1,subsample=0.8, colsample_bytree=0.8 
                        )

with joblib.parallel_backend('dask'):
    model2.fit(X_train, y_train)


# In[18]:


y_hat2 = model2.predict(X_test)
y_hat2


# In[26]:


type(y_hat2)


# In[28]:


print ("Second model")
r_2=r2_score(y_test.compute(), y_hat2)
mae_2=mean_absolute_error(y_test.compute(), y_hat2)
mse_2=mean_squared_error(y_test.compute(), y_hat2)
print ("R^2:",r_2)
print ("MAE:",mae_2)
print ("MSE:",mse_2)


# In[29]:


def evaluate( predictions, test_labels):
    
    errors = abs(predictions - test_labels.compute())
    mape = 100 * np.mean(errors / test_labels.compute())
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[34]:


base_accuray  = evaluate(y_hat,y_test)


# In[35]:


boosted = evaluate(y_hat2,y_test)


# In[37]:


print('Improvement of {:0.2f}%.'.format( 100 * (boosted - base_accuray) / base_accuray))


# In[ ]:




