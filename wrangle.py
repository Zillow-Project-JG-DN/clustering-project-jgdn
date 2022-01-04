####### IMPORT LIBRARIES ############


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import env


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

################ PULL DATA FROM DB ############## 


def get_db_url(database):
    '''
    Gets appropriate url to pull data from credentials stored in env file
    '''
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url


# acquire zillow data using the query
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



#### INITIAL MVP DROP THEM ALL MISSING VALUES FUNCTION #########

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


######## DROP COLUMNS

def remove_columns(df, cols_to_remove):  
    '''
    Pass a list od columns to remove
    '''
    df = df.drop(columns=cols_to_remove)
    return df

def import_csv():
    '''
    Brings in previous stored spreadsheet
    '''
    df = pd.read_csv('zillow.csv')
    
def single_use(df):
    '''
    Ensures we are only looking at single use properties with at least one bedroom and >= 350 sf.
    '''
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]
    return df

def add_county(df):
    '''
    Add column for counties
    '''
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    # drop columns not needed
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 
        'censustractandblock', 'propertylandusedesc','heatingorsystemdesc','unitcnt'
                            ,'buildingqualitytypeid'])
    
def clean(df):    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    return df

def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test

def outlier_function(df, cols, k):
	#function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

def wrangle():
    df = get_data_from_sql()