import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import env
import eval_model
import wrangle3


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



import gmaps
import gmaps.datasets

gmaps.configure(api_key="AIzaSyDlW6BYId6BmIp-mmA_lY_xNiQOKabd-2Q")


#Modeling Tools
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from statsmodels.formula.api import ols

from datetime import date
from scipy import stats


## Evaluation tools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


###########################
def plot_prices_clusters():
    #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    # Plot prices for plot clusters
    fig, axes = plt.subplots(3,2, sharex=False, figsize=(20, 25))
    fig.suptitle('Home prices by Price Cluster')
    p0=X_train[X_train['price_cluster']=='a']
    p1=X_train[X_train['price_cluster']=='b']
    p2=X_train[X_train['price_cluster']=='c']
    p3=X_train[X_train['price_cluster']=='d']
    p4=X_train[X_train['price_cluster']=='e']
#axes[0].set_title('All clusters together')
#axes[1].set_title('Cluster a')
#axes[2].set_title('Cluster b')
#axes[3].set_title('Cluster c')
#axes[4].set_title('Cluster d')
#axes[5].set_title('Cluster e')
    sns.histplot(data=p0, x='taxvaluedollarcnt', alpha=0.5, color='red', ax=axes[0,0])
    sns.histplot(data=p1, x='taxvaluedollarcnt', alpha=0.5, color='orange', ax=axes[0,0])
    sns.histplot(data=p2, x='taxvaluedollarcnt', alpha=0.5, color='yellow', ax=axes[0,0])
    sns.histplot(data=p3, x='taxvaluedollarcnt', alpha=0.5, color='green', ax=axes[0,0])
    sns.histplot(data=p4, x='taxvaluedollarcnt', alpha=0.5, color='blue', ax=axes[0,0])
    axes[0,0].ticklabel_format(style='plain')
    axes[0,0].set_title('Price clusters superimposed')
    axes[0,1].ticklabel_format(style='plain')
    axes[0,1].set_title('Cluster a')
    axes[1,0].ticklabel_format(style='plain')
    axes[1,0].set_title('Cluster b')
    axes[1,1].ticklabel_format(style='plain')
    axes[1,1].set_title('Cluster c')
    axes[2,0].ticklabel_format(style='plain')
    axes[2,0].set_title('Cluster d')
    axes[2,1].ticklabel_format(style='plain')
    axes[2,1].set_title('Cluster e')

    sns.histplot(data=p0, x='taxvaluedollarcnt', alpha=0.5, color='red', ax=axes[0,1])
    sns.histplot(data=p1, x='taxvaluedollarcnt', alpha=0.5, color='orange', ax=axes[1,0])
    sns.histplot(data=p2, x='taxvaluedollarcnt', alpha=0.5, color='yellow', ax=axes[1,1])
    sns.histplot(data=p3, x='taxvaluedollarcnt', alpha=0.5, color='green', ax=axes[2,0])
    sns.histplot(data=p4, x='taxvaluedollarcnt', alpha=0.5, color='blue', ax=axes[2,1])

    plt.ticklabel_format(style='plain')

    plt.show()

###########################
# def funciton_city():
#     pct_change=round(((ols_RMSE-baseline_RMSE)/baseline_RMSE)*100, 2)
# #rmse_validate = round(sqrt(mean_squared_error(validate_eval.actual, validate_eval.ols_yhat)))
#     baseline_r2 = (r2_score(ols_eval.actual, ols_eval.baseline_yhat), 2)
#     ols_train_r2 = (r2_score(ols_eval.actual, ols_eval.ols_yhat), 2)
# #ols_validate_r2 = round(r2_score(validate_eval.actual, validate_eval.ols_yhat), 2)

# #Output Findings

#     print(f'My model has value: {ols_RMSE < baseline_RMSE}')
#     print()
#     print(f'Baseline RMSE: {baseline_RMSE}')
#     print(f'My model train RMSE: {ols_RMSE}')
# #print(f'My model validate RMSE: {rmse_validate}')
#     print(f'RMSE difference baseline to model: {baseline_RMSE- ols_RMSE}')
# #print(f'RMSE difference train to validate: {ols_RMSE- rmse_validate}')
#     print(f'RMSE improvement: {pct_change}%')
#     print()
#     print(f'Baseline R2: {baseline_r2}')
#     print(f'Model train  R2: {ols_train_r2}')
# #print(f'Model Validate R2: {ols_validate_r2}')
