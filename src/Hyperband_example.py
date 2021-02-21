#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load hyperparam_subCh1_Akt.py
#!/usr/bin/env python

# In[1]:


import sys, os
import pandas as pd
import dask
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.model_selection import HyperbandSearchCV ,RandomizedSearchCV 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
import dask_xgboost
from dask.distributed import Client

# import matplotlib.pyplot as plt
#from dask_ml.preprocessing import DummyEncoder
import numpy as np
np.random.seed(seed = 101)
# import xgboost as xgb; print('XGBoost Version:', xgb.__version__)
# from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from scipy.stats import uniform, loguniform


# In[2]:


print('Dask Version:',dask.__version__)
jobid=os.getenv('SLURM_JOBID')
client= Client(scheduler_file='scheduler_%s.json' %jobid)

print(f'Job_id:{jobid}')


# In[3]:


print (client)
client


# In[4]:


#### MLP params 
# params = {
#     "hidden_layer_sizes": [
#         (24, ),
#         (12, 12),
#         (6, 6, 6, 6),
#         (4, 4, 4, 4, 4, 4),
#         (12, 6, 3, 3),
#     ],
#     "activation": ["relu", "logistic", "tanh"],
#     "alpha": np.logspace(-6, -3, num=1000),  # cnts
#     "batch_size": [16, 32, 64, 128, 256, 512],
# }

#### SGD params
params = {
    "l1_ratio": uniform(0, 1),
    "alpha": loguniform(1e-5, 1e-1),
    "penalty": ["l2", "l1", "elasticnet"],
    "learning_rate": ["invscaling", "adaptive","optimal"],
    "power_t": uniform(0, 1),
    "average": [True, False],
    "loss":["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    "epsilon": loguniform(1e-5, 1e-1),
}

   
    
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


print ("train base model")
import joblib
model = SGDRegressor(verbose=2)
with joblib.parallel_backend('dask'):
    model.fit(X_train.compute(), y_train.compute())


# In[10]:


Hb_model = SGDRegressor()


# In[11]:


#In practice, HyperbandSearchCV is most useful for longer searches
n_examples = 15 * len(X_train)
n_params = 15

max_iter = n_params  # number of times partial_fit will be called
chunks = n_examples // n_params  # number of examples each call sees

print(f"max_iter:{max_iter},chunks:{chunks} ")


# In[12]:


ml_hyperband = HyperbandSearchCV(
    estimator=Hb_model,
    parameters=params,
    max_iter=max_iter,
    patience=True,
    random_state=101
    
)

ml_hyperband.metadata["partial_fit_calls"]


# In[13]:


print (" hyperband search")
ml_hyperband


# In[14]:


## move dask dataframe to array
X_train = X_train.to_dask_array(lengths=True)
y_train = y_train.to_dask_array(lengths=True)


# In[15]:


# rechunk the array for the partial fit
X_train = X_train.rechunk((chunks, -1))
y_train = y_train.rechunk(chunks)


# In[16]:


print ("Starting hyperband search")
ml_hyperband.fit(X=X_train,y=y_train)


# In[17]:


print (f"best train score :{ml_hyperband.best_score_}")
print (f"best estimator : \n{ml_hyperband.best_estimator_}")
print (f"Score on test set : {ml_hyperband.score(X_test, y_test)}")


# In[18]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features.compute())
    errors = abs(predictions - test_labels.compute())
    mape = 100 * np.mean(errors / test_labels.compute())
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[19]:


base_accuracy = evaluate(model, X_test, y_test)
predictions = model.predict(X_test)


print ("base model")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))


# In[20]:


# In[ ]:


best_random = ml_hyperband.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test )

predictions = best_random.predict(X_test)
print ("best model hyeperband ")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))

# In[ ]:


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# In[ ]:





# In[ ]:





# In[ ]:




