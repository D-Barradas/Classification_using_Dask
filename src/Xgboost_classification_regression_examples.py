#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys, os
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.model_selection import RandomizedSearchCV 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import roc_curve,r2_score,classification_report

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
np.random.seed(seed = 101)
import xgboost as xgb; print('XGBoost Version:', xgb.__version__)
import dask_xgboost
print('Dask Version:',dask.__version__)
jobid=os.getenv('SLURM_JOBID')
client= Client(scheduler_file='scheduler_%s.json' %jobid)

print(f'Job_id:{jobid}')


# In[3]:


client


# In[4]:


from dask_ml.datasets import make_regression

X, y = make_regression(n_samples=4000000, n_features=32,
                           chunks=1000, n_informative=10,
                           random_state=101)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[6]:


params = {'objective': 'reg:squarederror','n_estimators':100000,
          'max_depth': 4, 'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 0.5}

bst = dask_xgboost.train(client, params, X_train, y_train, num_boost_round=100)


# In[7]:


y_hat = dask_xgboost.predict(client, bst, X_test).persist()
y_hat


# In[8]:


r=r2_score(y_test.compute(), y_hat.compute())
mae=mean_absolute_error(y_test.compute(), y_hat.compute())
mse=mean_squared_error(y_test.compute(), y_hat.compute())
print ("R^2:",r)
print ("MAE:",mae)
print ("MSE:",mse)


# In[9]:


from dask_ml.datasets import make_classification

X, y = make_classification(n_samples=4000000, n_features=32,
                           chunks=1000, n_informative=6,
                           random_state=101)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[11]:


params = {'objective': 'binary:logistic',
          'max_depth': 4, 'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 0.5}

bst = dask_xgboost.train(client, params, X_train, y_train, num_boost_round=10)


# In[21]:


y_hat = dask_xgboost.predict(client, bst, X_test).persist()
y_hat


# In[12]:


ax = xgb.plot_importance(bst, height=0.8, max_num_features=9)
ax.grid(False, axis="y")
ax.set_title('Estimated feature importance')
# plt.show()
plt.savefig("figures/Feature_importance_test.png",format="png")


# In[22]:


y_test, y_hat = dask.compute(y_test, y_hat)
fpr, tpr, _ = roc_curve(y_test, y_hat)


# In[23]:


from sklearn.metrics import auc

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=3,
        label='ROC Curve (area = {:.2f})'.format(auc(fpr, tpr)))
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    title="ROC Curve",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)
ax.legend();
# plt.show()
plt.savefig("figures/ROC_curve_test.png",format="png")

