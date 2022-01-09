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

##
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

train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()


# Set up dataframes for predictions
train_predictions = pd.DataFrame(y_train.copy())
validate_predictions = pd.DataFrame(y_validate.copy())
#Rename column to actual
train_predictions.rename(columns={'logerror': 'actual'}, inplace=True)
validate_predictions.rename(columns={'logerror': 'actual'}, inplace=True)
#Add model 1 predictions

# Create Model
model1 = ols(formula='logerror ~ area_cluster_la_newer + area_cluster_la_older + area_cluster_northwest_costal + area_cluster_palmdale_landcaster + area_cluster_santa_clarita + area_cluster_se_coast + size_cluster_1250_to_1650 + size_cluster_1300_to_2000 + size_cluster_1500_to_1900 + size_cluster_1500_to_2800 + size_cluster_2300_to_4400 + size_cluster_2900_to_4000 + price_cluster_69000_to_210000 + price_cluster_144000_to_355000 + price_cluster_34000_to_110000 + price_cluster_420000_to_870000 + price_cluster_45000_to_173000 + price_cluster_69000_to_210000', data=train).fit()
# Create Model
model2 = ols(formula='logerror ~ area_cluster_la_newer + area_cluster_la_older + area_cluster_northwest_costal + area_cluster_palmdale_landcaster + area_cluster_santa_clarita + area_cluster_se_coast + bathroomcnt + bedroomcnt + calculatedfinishedsquarefeet + acres + age', data=train).fit()
# Create Model
model3 = ols(formula='logerror ~ area_cluster_la_newer + area_cluster_la_older + area_cluster_northwest_costal + area_cluster_palmdale_landcaster + area_cluster_santa_clarita + area_cluster_se_coast + taxamount + taxvaluedollarcnt + structuretaxvaluedollarcnt  + landtaxvaluedollarcnt + acres + age', data=train).fit()
# Create Model
model4 = ols(formula='logerror ~ taxvaluedollarcnt + structuretaxvaluedollarcnt + taxvaluedollarcnt + landtaxvaluedollarcnt + acres + age', data=train).fit()
# Create Model
model5 = ols(formula='logerror ~ area_cluster_la_newer + area_cluster_la_older + area_cluster_northwest_costal + area_cluster_palmdale_landcaster + area_cluster_santa_clarita + area_cluster_se_coast + size_cluster_1250_to_1650 + size_cluster_1300_to_2000 + size_cluster_1500_to_1900 + size_cluster_1500_to_2800 + size_cluster_2300_to_4400 + size_cluster_2900_to_4000 + size_cluster_900_to_1200 + price_cluster_144000_to_355000 + price_cluster_34000_to_110000 + price_cluster_420000_to_870000 + price_cluster_45000_to_173000 + price_cluster_69000_to_210000 + taxvaluedollarcnt + structuretaxvaluedollarcnt  + landtaxvaluedollarcnt + taxamount', data=train).fit()
# Create OLS Model using encoded clusters
model6 = ols(formula='logerror ~ area_cluster_la_newer + area_cluster_la_older + area_cluster_northwest_costal + area_cluster_palmdale_landcaster + area_cluster_santa_clarita + area_cluster_se_coast + size_cluster_1250_to_1650 + size_cluster_1300_to_2000 + size_cluster_1500_to_1900 + size_cluster_1500_to_2800 + size_cluster_2300_to_4400 + size_cluster_2900_to_4000 + price_cluster_69000_to_210000 + price_cluster_144000_to_355000 + price_cluster_34000_to_110000 + price_cluster_420000_to_870000 + price_cluster_45000_to_173000 + price_cluster_69000_to_210000 + tax_cluster_1000_to_3000 + tax_cluster_16000_to_22000 + tax_cluster_30000_to_40000 + tax_cluster_5000_to_6000 + tax_cluster_8500_to_12000 ', data=train).fit()
#######
train_predictions['baseline_yhat']=train_predictions['actual'].mean ()
validate_predictions['baseline_yhat']=validate_predictions['actual'].mean()
train_predictions['baseline_residuals']=train_predictions.baseline_yhat-train_predictions.actual
validate_predictions['baseline_residuals']=validate_predictions.baseline_yhat-validate_predictions.actual
train_predictions['model1_yhat']=model1.predict(X_train)
validate_predictions['model1_yhat']=model1.predict(X_validate)
train_predictions['model1_residuals']=train_predictions.model1_yhat-train_predictions.actual
validate_predictions['model1_residuals']=validate_predictions.model1_yhat-validate_predictions.actual
baseline_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.baseline_yhat)))
validate_baseline_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.baseline_yhat)))
train_model1_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.model1_yhat)))
validate_model1_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.model1_yhat)))
train_baseline_r2 = (r2_score(train_predictions.actual,train_predictions.baseline_yhat))
validate_baseline_r2 = (r2_score(validate_predictions.actual,validate_predictions.baseline_yhat))
train_model1_r2 = (r2_score(train_predictions.actual,train_predictions.model1_yhat))
validate_model1_r2 = (r2_score(validate_predictions.actual,validate_predictions.model1_yhat))
##########
#Model #1
def model_1():
    # Make predictions
    train_predictions['model1_yhat'] = model1.predict(X_train)

    validate_predictions['model1_yhat'] = model1.predict(X_validate)

    train_predictions['model1_residuals']=train_predictions.model1_yhat-train_predictions.actual

    validate_predictions['model1_residuals']=validate_predictions.model1_yhat-validate_predictions.actual

    train_model1_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.model1_yhat)))
    validate_model1_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.model1_yhat)))

    train_model1_r2 = (r2_score(train_predictions.actual,train_predictions.model1_yhat))

    validate_model1_r2 = (r2_score(validate_predictions.actual,validate_predictions.model1_yhat))

    print(f'train_rmse: {train_model1_RMSE}')
    print(f'train_r2: {train_model1_r2}')

    print(f'validate_rmse: {validate_model1_RMSE}')
    print(f'validate_model1_r2: {validate_model1_r2}')

##########
#Model #2
def model_2():
    # Make predictions
    train_predictions['model2_yhat'] = model2.predict(X_train)

    validate_predictions['model2_yhat'] = model2.predict(X_validate)

    train_predictions['model2_residuals']=train_predictions.model2_yhat-train_predictions.actual

    validate_predictions['model2_residuals']=validate_predictions.model2_yhat-validate_predictions.actual

    train_model2_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.model2_yhat)))
    validate_model2_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.model2_yhat)))

    train_model2_r2 = (r2_score(train_predictions.actual,train_predictions.model2_yhat))

    validate_model2_r2 = (r2_score(validate_predictions.actual,validate_predictions.model2_yhat))

    print(f'train_rmse: {train_model2_RMSE}')
    print(f'train_r2: {train_model2_r2}')

    print(f'validate_rmse: {validate_model2_RMSE}')
    print(f'validate_model2_r2: {validate_model2_r2}')

##########
#Model #3
def model_3():
    # Make predictions
    train_predictions['model3_yhat'] = model3.predict(X_train)

    validate_predictions['model3_yhat'] = model3.predict(X_validate)

    train_predictions['model3_residuals']=train_predictions.model3_yhat-train_predictions.actual

    validate_predictions['model3_residuals']=validate_predictions.model3_yhat-validate_predictions.actual

    train_model3_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.model3_yhat)))
    validate_model3_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.model3_yhat)))

    train_model3_r2 = (r2_score(train_predictions.actual,train_predictions.model3_yhat))

    validate_model3_r2 = (r2_score(validate_predictions.actual,validate_predictions.model3_yhat))

    print(f'train_rmse: {train_model3_RMSE}')
    print(f'train_r2: {train_model3_r2}')

    print(f'validate_rmse: {validate_model3_RMSE}')
    print(f'validate_model3_r2: {validate_model3_r2}')

##########
#Model #4
def model_4():
    # Make predictions
    train_predictions['model4_yhat'] = model4.predict(X_train)

    validate_predictions['model4_yhat'] = model4.predict(X_validate)

    train_predictions['model4_residuals']=train_predictions.model4_yhat-train_predictions.actual

    validate_predictions['model4_residuals']=validate_predictions.model4_yhat-validate_predictions.actual

    train_model4_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.model4_yhat)))
    validate_model4_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.model4_yhat)))

    train_model4_r2 = (r2_score(train_predictions.actual,train_predictions.model4_yhat))

    validate_model4_r2 = (r2_score(validate_predictions.actual,validate_predictions.model4_yhat))

    print(f'train_rmse: {train_model4_RMSE}')
    print(f'train_r2: {train_model4_r2}')

    print(f'validate_rmse: {validate_model4_RMSE}')
    print(f'validate_model4_r2: {validate_model4_r2}')

##########
#Model #5
def model_5():
    # Make predictions
    train_predictions['model5_yhat'] = model5.predict(X_train)

    validate_predictions['model5_yhat'] = model5.predict(X_validate)

    train_predictions['model5_residuals']=train_predictions.model5_yhat-train_predictions.actual

    validate_predictions['model5_residuals']=validate_predictions.model5_yhat-validate_predictions.actual

    train_model5_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.model5_yhat)))
    validate_model5_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.model5_yhat)))

    train_model5_r2 = (r2_score(train_predictions.actual,train_predictions.model5_yhat))

    validate_model5_r2 = (r2_score(validate_predictions.actual,validate_predictions.model5_yhat))

    print(f'train_rmse: {train_model5_RMSE}')
    print(f'train_r2: {train_model5_r2}')

    print(f'validate_rmse: {validate_model5_RMSE}')
    print(f'validate_model5_r2: {validate_model5_r2}')
##########
#Model #6
def model_6():
    # Make predictions
    train_predictions['model6_yhat'] = model6.predict(X_train)

    validate_predictions['model6_yhat'] = model6.predict(X_validate)

    train_predictions['model6_residuals']=train_predictions.model6_yhat-train_predictions.actual

    validate_predictions['model6_residuals']=validate_predictions.model6_yhat-validate_predictions.actual

    train_model6_RMSE=(sqrt(mean_squared_error(train_predictions.actual,train_predictions.model6_yhat)))
    validate_model6_RMSE=(sqrt(mean_squared_error(validate_predictions.actual,validate_predictions.model6_yhat)))

    train_model6_r2 = (r2_score(train_predictions.actual,train_predictions.model6_yhat))

    validate_model6_r2 = (r2_score(validate_predictions.actual,validate_predictions.model6_yhat))

    print(f'train_rmse: {train_model6_RMSE}')
    print(f'train_r2: {train_model6_r2}')

    print(f'validate_rmse: {validate_model6_RMSE}')
    print(f'validate_model6_r2: {validate_model6_r2}')