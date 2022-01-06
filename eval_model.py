# Basic libraries
import pandas as pd
import numpy as np 

#Vizualization Tools
import matplotlib.pyplot as plt
import seaborn as sns

#Modeling Tools
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

from datetime import date
from scipy import stats


## Evaluation tools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

############# FUNCTION FOR VIZUALIZATIONS #######

def correlation_exploration(df, x_string, y_string):
    r, p = stats.pearsonr(df[x_string], df[y_string])
    ax= sns.regplot(x=x_string, y=y_string, data=df, line_kws={"color": "red"})
    plt.figure(figsize=(16,8))
    plt.title(f"{x_string}'s Relationship with {y_string}")
    print(f'The p-value is: {p:4f}. There is {round(p,3)}% chance that we see these results by chance.')
    print(f'r = {round(r, 2)}')