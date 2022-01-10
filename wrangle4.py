
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
from sklearn.cluster import KMeans


from datetime import date

import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

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
    select prop.parcelid
        , pred.logerror
        , bathroomcnt
        , bedroomcnt
        , calculatedfinishedsquarefeet
        , fips
        , latitude
        , longitude
        , lotsizesquarefeet
        , regionidcity
        , regionidcounty
        , regionidzip
        , yearbuilt
        , structuretaxvaluedollarcnt
        , taxvaluedollarcnt
        , landtaxvaluedollarcnt
        , taxamount
    from properties_2017 prop
    inner join predictions_2017 pred on prop.parcelid = pred.parcelid
    where propertylandusetypeid = 261;
    '''
    return pd.read_sql(query, get_db_url('zillow'))

    #df = pd.read_sql(query, get_db_url('zillow'), index_col='id')
    # return df

    # Pull values from SQL


# Cache


def single_use(df):
    '''
    Ensures we are only looking at single use properties with at least one bedroom and >= 350 sf.
    '''
    #single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    #df = df[df.propertylandusetypeid.isin(single_use)]
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)  # & ((df.unitcnt <= 1) | df.unitcnt.isnull())
            & (df.calculatedfinishedsquarefeet > 350)]
    return df


def add_county(df):
    '''
    Add column for counties
    '''
    import numpy as np
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                            np.where(df.fips == 6059, 'Orange',
                                     'Ventura'))
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


def clean(df):
    # replace nulls with median values for select columns
    #df.lotsizesquarefeet.fillna(7313, inplace=True)
    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    #df[df.calculatedfinishedsquarefeet < 8000]
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    return df


def remove_outliers1(df, col_list, k=1.5):
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


def scale(X_train, X_validate, X_test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    scaled_vars = ['latitude', 'longitude', 'bathroomcnt', 'taxrate', 'bedroomcnt',
                   'lotsizesquarefeet', 'age', 'acres',  'bath_bed_ratio', 'calculatedfinishedsquarefeet']
    scaled_column_names = ['scaled_' + i for i in scaled_vars]
    #num_vars = list(X_train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    X_train[scaled_column_names] = scaler.fit_transform(X_train[scaled_vars])
    X_validate[scaled_column_names] = scaler.transform(X_validate[scaled_vars])
    X_test[scaled_column_names] = scaler.transform(X_test[scaled_vars])
    return X_train, X_validate, X_test


def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age,
                           bins=[0, 5, 10, 20, 30, 40, 50, 60, 70,
                                 80, 90, 100, 110, 120, 130, 140],
                           labels=[0, .066, .133, .20, .266, .333, .40, .466, .533,
                                   .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins=[0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200],
                             labels=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet,
                            bins=[0, 800, 1000, 1250, 1500, 2000,
                                  2500, 3000, 4000, 7000, 12000],
                            labels=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                            )

    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt / \
        df.calculatedfinishedsquarefeet

    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft,
                                             bins=[0, 25, 50, 75, 100, 150,
                                                   200, 300, 500, 1000, 1500],
                                             labels=[
                                                 0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                             )

    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins=[0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels=[
                                           0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                       )

    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})

    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    # 12447 is the ID for city of LA.
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    return df


def remove_outliers(df):
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''

    return df[((df.bathroomcnt <= 7) & (df.bedroomcnt <= 7) &
               (df.regionidzip < 100000) &
               (df.bathroomcnt > 0) &
               (df.bedroomcnt > 0) &
               (df.acres < 20) &
               (df.calculatedfinishedsquarefeet < 10000) &
               (df.taxrate < 10)
               )]


def create_clusters(X_train, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state=13)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X_train[cluster_vars])

    return kmeans


def get_centroids(kmeans, cluster_vars, cluster_name):
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_,
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df


def wrangle():
    df = pd.read_csv('zillow_wrangle3.csv')
    df = single_use(df)
    df = clean(df)
    df = add_county(df)
    df = handle_missing_values(df)
    df = create_features(df)
    df = remove_outliers(df)
    df.latitude = df.latitude/1000000
    df.longitude = df.longitude/1000000
    train, validate, test = split_my_data(df)
    train['logerror_bins'] = pd.cut(
        train.logerror, [-5, -.2, -.05, .05, .2, 4])
    train, validate, test = add_baseline(train, validate, test)
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = split_xy(
        train, validate, test)
    X_train, X_validate, X_test = scale(
        X_train, X_validate, X_test)
    k = 6
    cluster_vars = ['scaled_latitude', 'scaled_longitude', 'age_bin']
    cluster_name = 'area_cluster'
    kmeans = create_clusters(X_train, k, cluster_vars)
    centroid_df = get_centroids(kmeans, cluster_vars, cluster_name)
    X_train['area_cluster'] = kmeans.predict(X_train[cluster_vars])
    X_validate['area_cluster'] = kmeans.predict(X_validate[cluster_vars])
    X_test['area_cluster'] = kmeans.predict(X_test[cluster_vars])
    k = 7
    cluster_name = 'size_cluster'
    cluster_vars = ['scaled_bathroomcnt',
                    'sqft_bin', 'acres_bin', 'bath_bed_ratio']
    # fit kmeans
    kmeans = create_clusters(X_train, k, cluster_vars)
    # get centroid values per variable per cluster
    centroid_df = get_centroids(kmeans, cluster_vars, cluster_name)
    X_train['size_cluster'] = kmeans.predict(X_train[cluster_vars])
    X_validate['size_cluster'] = kmeans.predict(X_validate[cluster_vars])
    X_test['size_cluster'] = kmeans.predict(X_test[cluster_vars])

    k = 5
    cluster_name = 'price_cluster'
    cluster_vars = ['taxrate',
                    'structure_dollar_sqft_bin', 'lot_dollar_sqft_bin']

    # fit kmeans
    kmeans = create_clusters(X_train, k, cluster_vars)

    # get centroid values per variable per cluster
    centroid_df = get_centroids(kmeans, cluster_vars, cluster_name)
    X_train['price_cluster'] = kmeans.predict(X_train[cluster_vars])
    X_validate['price_cluster'] = kmeans.predict(X_validate[cluster_vars])
    X_test['price_cluster'] = kmeans.predict(X_test[cluster_vars])

    # create tax clusters
    # DId not use cluster because created on undscaled values, invalid
    k = 5
    cluster_name = 'tax_cluster'
    cluster_vars = ['taxamount', 'taxvaluedollarcnt',
                    'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt']

    # fit kmeans
    kmeans = create_clusters(X_train, k, cluster_vars)
    kmeans = create_clusters(X_validate, k, cluster_vars)

    X_train['tax_cluster'] = kmeans.predict(X_train[cluster_vars])
    X_validate['tax_cluster'] = kmeans.predict(X_validate[cluster_vars])
    X_test['tax_cluster'] = kmeans.predict(X_test[cluster_vars])

    # get centroid values per variable per cluster
    centroid_df = get_centroids(kmeans, cluster_vars, cluster_name)

    # Add english names to clusters

    X_train['area_cluster'] = X_train.area_cluster.map({
        0: "santa_clarita",
        1: "se_coast",
        2: "palmdale_landcaster",
        3: "la_older",
        4: "la_newer",
        5: "northwest_costal"
    })

    X_train['price_cluster'] = X_train.price_cluster.map({
        0: "420000_to_870000",
        1: "45000_to_173000",
        2: "69000_to_210000",
        3: "144000_to_355000",
        4: "34000_to_110000",
    })
    X_train['size_cluster'] = X_train.size_cluster.map({
        0: "1300_to_2000",
        1: "1250_to_1650",
        2: "1500_to_1900",
        3: "2900_to_4000",
        4: "2300_to_4400",
        5: "1500_to_2800",
        6: "900_to_1200",
    })

    X_train['tax_cluster'] = X_train.tax_cluster.map({
        0: "1000_to_3000",
        1: "30000_to_40000",
        2: "8500_to_12000",
        3: "16000_to_22000",
        4: "5000_to_6000",
    })

    X_validate['tax_cluster'] = X_validate.tax_cluster.map({
        0: "1000_to_3000",
        1: "30000_to_40000",
        2: "8500_to_12000",
        3: "16000_to_22000",
        4: "5000_to_6000",
    })

    X_validate['area_cluster'] = X_validate.area_cluster.map({
        0: "santa_clarita",
        1: "se_coast",
        2: "palmdale_landcaster",
        3: "la_older",
        4: "la_newer",
        5: "northwest_costal"
    })

    X_validate['price_cluster'] = X_validate.price_cluster.map({
        0: "420000_to_870000",
        1: "45000_to_173000",
        2: "69000_to_210000",
        3: "144000_to_355000",
        4: "34000_to_110000",
    })
    X_validate['size_cluster'] = X_validate.size_cluster.map({
        0: "1300_to_2000",
        1: "1250_to_1650",
        2: "1500_to_1900",
        3: "2900_to_4000",
        4: "2300_to_4400",
        5: "1500_to_2800",
        6: "900_to_1200",
    })

    X_test['area_cluster'] = X_test.area_cluster.map({
        0: "santa_clarita",
        1: "se_coast",
        2: "palmdale_landcaster",
        3: "la_older",
        4: "la_newer",
        5: "northwest_costal"
    })

    X_test['price_cluster'] = X_test.price_cluster.map({
        0: "420000_to_870000",
        1: "45000_to_173000",
        2: "69000_to_210000",
        3: "144000_to_355000",
        4: "34000_to_110000",
    })
    X_test['size_cluster'] = X_test.size_cluster.map({
        0: "1300_to_2000",
        1: "1250_to_1650",
        2: "1500_to_1900",
        3: "2900_to_4000",
        4: "2300_to_4400",
        5: "1500_to_2800",
        6: "900_to_1200",
    })
    dummy_df = pd.get_dummies(
        X_train[['area_cluster', 'size_cluster', 'price_cluster', 'tax_cluster']], drop_first=False)

    X_train = pd.concat([X_train, dummy_df], axis=1)

    dummy_df2 = pd.get_dummies(
        X_validate[['area_cluster', 'size_cluster', 'price_cluster', 'tax_cluster']], drop_first=False)

    X_validate = pd.concat([X_validate, dummy_df2], axis=1)

    dummy_df3 = pd.get_dummies(
        X_test[['area_cluster', 'size_cluster', 'price_cluster', 'tax_cluster']], drop_first=False)

    X_test = pd.concat([X_test, dummy_df3], axis=1)

    train2 = X_train
    train2['logerror'] = y_train
    train = train2
    return train, X_train, y_train, X_validate, y_validate, X_test, y_test
