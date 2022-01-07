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
