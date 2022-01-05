
# Basic libraries
from env import host, user, password  # Database credentials
import wrangle
import pandas as pd
import numpy as np
import numpy as np

# Vizualization Tools
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling Tools
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

from datetime import date

import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings("ignore")

# Custim functions

# URL from ENV


def get_db_url(database):
    '''
    Gets appropriate url to pull data from credentials stored in env file
    '''
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

# Query


def get_zillow():
    query = '''
    SELECT prop.*, 
        pred.logerror, 
        pred.transactiondate, 
        air.airconditioningdesc, 
        arch.architecturalstyledesc, 
        build.buildingclassdesc, 
        heat.heatingorsystemdesc, 
        landuse.propertylandusedesc, 
        story.storydesc, 
        construct.typeconstructiondesc 

    FROM   properties_2017 prop  
        INNER JOIN (SELECT parcelid,
                            logerror,
                            Max(transactiondate) transactiondate 
                    FROM   predictions_2017 
                    GROUP  BY parcelid, logerror) pred
                USING (parcelid) 
        LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
        LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
        LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
        LEFT JOIN storytype story USING (storytypeid) 
        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
    WHERE  prop.latitude IS NOT NULL 
        AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31' 
    '''

    df = pd.read_sql(query, get_db_url('zillow'), index_col='id')
    return df

    # Pull values from SQL


def single_use(df):
    '''
    Ensures we are only looking at single use properties with at least one bedroom and >= 350 sf.
    '''
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt <= 1) | df.unitcnt.isnull())
            & (df.calculatedfinishedsquarefeet > 350)]
    return df


def add_county(df):
    '''
    Add column for counties
    '''
    df['county'] = df['fips'].replace({6037:'LA',6059: 'Orange',6111:'Ventura'})
    # import numpy as np
    # df['county'] = np.where(df.fips == 6037, 'Los_Angeles')
    # df['county'] = np.where(df.fips == 6059, 'Orange')
    # df['county'] = np.where(df.fips == 6111, 'Ventura')
    
    return df


def handle_missing_values(df, prop_required_column=.5, prop_required_row=.70):
    threshold = int(round(prop_required_column*len(df.index), 0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns), 0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def remove_columns(df, cols_to_remove):
    '''
    Pass a list od columns to remove
    '''
    df = df.drop(columns=cols_to_remove)
    return df


<<<<<<< HEAD
# def clean(df):
#     # replace nulls with median values for select columns
#     df.lotsizesquarefeet.fillna(7313, inplace=True)
#     # Columns to look for outliers
#     df = df[df.taxvaluedollarcnt < 5_000_000]
#     df[df.calculatedfinishedsquarefeet < 8000]
#     # Just to be sure we caught all nulls, drop them here
#     df = df.dropna()
#     return df
=======
def clean(df):
    # replace nulls with median values for select columns
    #df.lotsizesquarefeet.fillna(7313, inplace=True)
    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    #df[df.calculatedfinishedsquarefeet < 8000]
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    return df
>>>>>>> d5a7ad98f3be7602ac801454cbcf057932388b5f


def remove_outliers(df, col_list, k=1.5):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def split_my_data(df, pct=0.10):
    '''
    This splits a dataframe into train, validate, and test sets. 
    df = dataframe to split
    pct = size of the test set, 1/2 of size of the validate set
    Returns three dataframes (train, validate, test)
    '''
    train_validate, test = train_test_split(
        df, test_size=pct, random_state=123)
    train, validate = train_test_split(
        train_validate, test_size=pct*2, random_state=123)
    return train, validate, test


def min_max_scaler(train, validate, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    validate[num_vars] = scaler.transform(validate[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, validate, test


def add_baseline(train, validate, test):
    '''
    Assigns mean error as baseline prediction
    '''
    baseline = train.logerror.mean()
    train['baseline'] = baseline
    validate['baseline'] = baseline
    test['baseline'] = baseline
    return train, validate, test


def split_xy(train, validate, test):
    '''
    Splits dataframe into train, validate, and test data frames
    '''
    X_train = train.drop(columns='logerror')
    y_train = train.logerror

    X_validate = validate.drop(columns='logerror')
    y_validate = validate.logerror

    X_test = test.drop(columns='logerror')
    y_test = test.logerror

    return train, X_train, y_train, X_validate, y_validate, X_test, y_test


def scale(X_train, X_validate, X_test, train, validate, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(X_train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    train[num_vars] = scaler.fit_transform(X_train[num_vars])
    validate[num_vars] = scaler.transform(X_validate[num_vars])
    test[num_vars] = scaler.transform(X_test[num_vars])
    return X_train, X_validate, X_test


def wrangle():
    df = pd.read_csv('unedited_zillow.csv')
    df = single_use(df)
    df = add_county(df)
    df = handle_missing_values(df)
<<<<<<< HEAD
=======
    #df = clean(df)
>>>>>>> d5a7ad98f3be7602ac801454cbcf057932388b5f
    df = df.dropna()
    train, validate, test = split_my_data(df)
    train, validate, test = add_baseline(train, validate, test)
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = split_xy(
        train, validate, test)
    X_train, X_validate, X_test = scale(
        X_train, X_validate, X_test, train, validate, test)
    return train, X_train, y_train, X_validate, y_validate, X_test, y_test
