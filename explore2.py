<<<<<<< HEAD
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy import stats
from datetime import date
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
import gmaps.datasets
import gmaps
=======
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
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


<<<<<<< HEAD
gmaps.configure(api_key="AIzaSyDlW6BYId6BmIp-mmA_lY_xNiQOKabd-2Q")


# Modeling Tools


# Evaluation tools


def find_k(X_train, cluster_vars, k_range):
    '''plots k for intertia, pct_delta, & delta - aid in finding K target for
    clustered features applied to function'''

    # enter clusters/features, cluster_name, & range
=======

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


def find_k(X_train, cluster_vars, k_range):
    '''plots k for intertia, pct_delta, & delta - aid in finding K target for 
    clustered features applied to function'''

    #enter clusters/features, cluster_name, & range    
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

<<<<<<< HEAD
        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X.
        kmeans.fit(X_train[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_)

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1], 0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1)
                 for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1],
                                         sse=sse[0:-1],
                                         delta=delta,
                                         pct_delta=pct_delta))
=======
        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(X_train[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
    sse=sse[0:-1], 
    delta=delta, 
    pct_delta=pct_delta))
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
<<<<<<< HEAD
    plt.title(
        'The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
=======
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
<<<<<<< HEAD
    plt.title(
        'For which k values are we seeing increased changes (absolute) in SSE?')
=======
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
    plt.show()

    return k_comparisons_df

################################
<<<<<<< HEAD


def plot_size_clusters():
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle3.wrangle()
    #
    s0 = X_train[X_train['size_cluster'] == '1300_to_2000']
    s1 = X_train[X_train['size_cluster'] == '1250_to_1650']
    s2 = X_train[X_train['size_cluster'] == '1500_to_1900']
    s3 = X_train[X_train['size_cluster'] == '2900_to_4000']
    s4 = X_train[X_train['size_cluster'] == '2300_to_4400']
    s5 = X_train[X_train['size_cluster'] == '1500_to_2800']
    s6 = X_train[X_train['size_cluster'] == '900_to_1200']

    # Plot size clusters
    fig, axes = plt.subplots(4, 2, sharex=False, figsize=(20, 25))
    fig.suptitle('Home prices by Price Cluster')
    # axes[0].set_title('All clusters together')
    # axes[1].set_title('Cluster a')
    # axes[2].set_title('Cluster b')
    # axes[3].set_title('Cluster c')
    # axes[4].set_title('Cluster d')
    # axes[5].set_title('Cluster e')
    sns.histplot(data=s0, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='red', ax=axes[0, 0])
    sns.histplot(data=s1, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='orange', ax=axes[0, 0])
    sns.histplot(data=s2, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='yellow', ax=axes[0, 0])
    sns.histplot(data=s3, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='green', ax=axes[0, 0])
    sns.histplot(data=s4, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='blue', ax=axes[0, 0])
    sns.histplot(data=s5, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='purple', ax=axes[0, 0])
    sns.histplot(data=s6, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='pink', ax=axes[0, 0])
    axes[0, 0].ticklabel_format(style='plain')
    axes[0, 0].set_title('Size clusters superimposed')
    axes[0, 1].ticklabel_format(style='plain')
    axes[0, 1].set_title('Cluster: 1300_to_2000')

    axes[1, 0].ticklabel_format(style='plain')
    axes[1, 0].set_title('Cluster: 1250_to_1650')
    axes[1, 1].ticklabel_format(style='plain')
    axes[1, 1].set_title('Cluster: 1500_to_1900')
    axes[2, 0].ticklabel_format(style='plain')
    axes[2, 0].set_title('Cluster: 2900_to_4000')
    axes[2, 1].ticklabel_format(style='plain')
    axes[2, 1].set_title('Cluster: 2300_to_4400')
    axes[3, 0].ticklabel_format(style='plain')
    axes[3, 0].set_title('Cluster: 1500_to_2800')
    axes[3, 1].ticklabel_format(style='plain')
    axes[3, 1].set_title('Cluster: 900_to_1200')

    sns.histplot(data=s0, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='red', ax=axes[0, 1])
    sns.histplot(data=s1, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='orange', ax=axes[1, 0])
    sns.histplot(data=s2, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='yellow', ax=axes[1, 1])
    sns.histplot(data=s3, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='green', ax=axes[2, 0])
    sns.histplot(data=s4, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='blue', ax=axes[2, 1])
    sns.histplot(data=s5, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='purple', ax=axes[3, 0])
    sns.histplot(data=s6, x='calculatedfinishedsquarefeet',
                 alpha=0.5, color='pink', ax=axes[3, 1])
=======
def plot_size_clusters():
        #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    #
    s0=X_train[X_train['size_cluster']=='1300_to_2000']
    s1=X_train[X_train['size_cluster']=='1250_to_1650']
    s2=X_train[X_train['size_cluster']=='1500_to_1900']
    s3=X_train[X_train['size_cluster']=='2900_to_4000']
    s4=X_train[X_train['size_cluster']=='2300_to_4400']
    s5=X_train[X_train['size_cluster']=='1500_to_2800']
    s6=X_train[X_train['size_cluster']=='900_to_1200']

        #Plot size clusters
    fig, axes = plt.subplots(4,2, sharex=False, figsize=(20, 25))
    fig.suptitle('Home prices by Price Cluster')
    #axes[0].set_title('All clusters together')
    #axes[1].set_title('Cluster a')
    #axes[2].set_title('Cluster b')
    #axes[3].set_title('Cluster c')
    #axes[4].set_title('Cluster d')
    #axes[5].set_title('Cluster e')
    sns.histplot(data=s0, x='calculatedfinishedsquarefeet', alpha=0.5, color='red', ax=axes[0,0])
    sns.histplot(data=s1, x='calculatedfinishedsquarefeet', alpha=0.5, color='orange', ax=axes[0,0])
    sns.histplot(data=s2, x='calculatedfinishedsquarefeet', alpha=0.5, color='yellow', ax=axes[0,0])
    sns.histplot(data=s3, x='calculatedfinishedsquarefeet', alpha=0.5, color='green', ax=axes[0,0])
    sns.histplot(data=s4, x='calculatedfinishedsquarefeet', alpha=0.5, color='blue', ax=axes[0,0])
    sns.histplot(data=s5, x='calculatedfinishedsquarefeet', alpha=0.5, color='purple', ax=axes[0,0])
    sns.histplot(data=s6, x='calculatedfinishedsquarefeet', alpha=0.5, color='pink', ax=axes[0,0])
    axes[0,0].ticklabel_format(style='plain')
    axes[0,0].set_title('Size clusters superimposed')
    axes[0,1].ticklabel_format(style='plain')
    axes[0,1].set_title('Cluster: 1300_to_2000')

    axes[1,0].ticklabel_format(style='plain')
    axes[1,0].set_title('Cluster: 1250_to_1650')
    axes[1,1].ticklabel_format(style='plain')
    axes[1,1].set_title('Cluster: 1500_to_1900')
    axes[2,0].ticklabel_format(style='plain')
    axes[2,0].set_title('Cluster: 2900_to_4000')
    axes[2,1].ticklabel_format(style='plain')
    axes[2,1].set_title('Cluster: 2300_to_4400')
    axes[3,0].ticklabel_format(style='plain')
    axes[3,0].set_title('Cluster: 1500_to_2800')
    axes[3,1].ticklabel_format(style='plain')
    axes[3,1].set_title('Cluster: 900_to_1200')

    sns.histplot(data=s0, x='calculatedfinishedsquarefeet', alpha=0.5, color='red', ax=axes[0,1])
    sns.histplot(data=s1, x='calculatedfinishedsquarefeet', alpha=0.5, color='orange', ax=axes[1,0])
    sns.histplot(data=s2, x='calculatedfinishedsquarefeet', alpha=0.5, color='yellow', ax=axes[1,1])
    sns.histplot(data=s3, x='calculatedfinishedsquarefeet', alpha=0.5, color='green', ax=axes[2,0])
    sns.histplot(data=s4, x='calculatedfinishedsquarefeet', alpha=0.5, color='blue', ax=axes[2,1])
    sns.histplot(data=s5, x='calculatedfinishedsquarefeet', alpha=0.5, color='purple', ax=axes[3,0])
    sns.histplot(data=s6, x='calculatedfinishedsquarefeet', alpha=0.5, color='pink', ax=axes[3,1])
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c

    plt.ticklabel_format(style='plain')

    plt.show()


################################
def plot_prices_clusters():
    ''' '''
<<<<<<< HEAD
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle3.wrangle()
    # Plot prices for plot clusters
    fig, axes = plt.subplots(3, 2, sharex=False, figsize=(20, 25))
    fig.suptitle('Home prices by Price Cluster')
    p0 = X_train[X_train['price_cluster'] == '420000_to_870000']
    p1 = X_train[X_train['price_cluster'] == '45000_to_173000']
    p2 = X_train[X_train['price_cluster'] == '69000_to_210000']
    p3 = X_train[X_train['price_cluster'] == '144000_to_355000']
    p4 = X_train[X_train['price_cluster'] == '34000_to_110000']
# axes[0].set_title('All clusters together')
# axes[1].set_title('Cluster a')
# axes[2].set_title('Cluster b')
# axes[3].set_title('Cluster c')
# axes[4].set_title('Cluster d')
# axes[5].set_title('Cluster e')
    sns.histplot(data=p0, x='taxvaluedollarcnt',
                 alpha=0.5, color='red', ax=axes[0, 0])
    sns.histplot(data=p1, x='taxvaluedollarcnt',
                 alpha=0.5, color='orange', ax=axes[0, 0])
    sns.histplot(data=p2, x='taxvaluedollarcnt',
                 alpha=0.5, color='yellow', ax=axes[0, 0])
    sns.histplot(data=p3, x='taxvaluedollarcnt',
                 alpha=0.5, color='green', ax=axes[0, 0])
    sns.histplot(data=p4, x='taxvaluedollarcnt',
                 alpha=0.5, color='blue', ax=axes[0, 0])
    axes[0, 0].ticklabel_format(style='plain')
    axes[0, 0].set_title('Price clusters superimposed')
    axes[0, 1].ticklabel_format(style='plain')
    axes[0, 1].set_title('Cluster: 420000_to_870000')
    axes[1, 0].ticklabel_format(style='plain')
    axes[1, 0].set_title('Cluster: 45000_to_173000')
    axes[1, 1].ticklabel_format(style='plain')
    axes[1, 1].set_title('Cluster: 69000_to_210000')
    axes[2, 0].ticklabel_format(style='plain')
    axes[2, 0].set_title('Cluster: 144000_to_355000')
    axes[2, 1].ticklabel_format(style='plain')
    axes[2, 1].set_title('Cluster: 34000_to_110000')

    sns.histplot(data=p0, x='taxvaluedollarcnt',
                 alpha=0.5, color='red', ax=axes[0, 1])
    sns.histplot(data=p1, x='taxvaluedollarcnt',
                 alpha=0.5, color='orange', ax=axes[1, 0])
    sns.histplot(data=p2, x='taxvaluedollarcnt',
                 alpha=0.5, color='yellow', ax=axes[1, 1])
    sns.histplot(data=p3, x='taxvaluedollarcnt',
                 alpha=0.5, color='green', ax=axes[2, 0])
    sns.histplot(data=p4, x='taxvaluedollarcnt',
                 alpha=0.5, color='blue', ax=axes[2, 1])
=======
    #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    # Plot prices for plot clusters
    fig, axes = plt.subplots(3,2, sharex=False, figsize=(20, 25))
    fig.suptitle('Home prices by Price Cluster')
    p0=X_train[X_train['price_cluster']=='420000_to_870000']
    p1=X_train[X_train['price_cluster']=='45000_to_173000']
    p2=X_train[X_train['price_cluster']=='69000_to_210000']
    p3=X_train[X_train['price_cluster']=='144000_to_355000']
    p4=X_train[X_train['price_cluster']=='34000_to_110000']
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
    axes[0,1].set_title('Cluster: 420000_to_870000')
    axes[1,0].ticklabel_format(style='plain')
    axes[1,0].set_title('Cluster: 45000_to_173000')
    axes[1,1].ticklabel_format(style='plain')
    axes[1,1].set_title('Cluster: 69000_to_210000')
    axes[2,0].ticklabel_format(style='plain')
    axes[2,0].set_title('Cluster: 144000_to_355000')
    axes[2,1].ticklabel_format(style='plain')
    axes[2,1].set_title('Cluster: 34000_to_110000')

    sns.histplot(data=p0, x='taxvaluedollarcnt', alpha=0.5, color='red', ax=axes[0,1])
    sns.histplot(data=p1, x='taxvaluedollarcnt', alpha=0.5, color='orange', ax=axes[1,0])
    sns.histplot(data=p2, x='taxvaluedollarcnt', alpha=0.5, color='yellow', ax=axes[1,1])
    sns.histplot(data=p3, x='taxvaluedollarcnt', alpha=0.5, color='green', ax=axes[2,0])
    sns.histplot(data=p4, x='taxvaluedollarcnt', alpha=0.5, color='blue', ax=axes[2,1])
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c

    plt.ticklabel_format(style='plain')

    plt.show()
###########################
<<<<<<< HEAD


def plot_tax_cluster():
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle3.wrangle()
    # labels
    t0 = X_train[X_train['tax_cluster'] == '1000_to_3000']
    t1 = X_train[X_train['tax_cluster'] == '30000_to_40000']
    t2 = X_train[X_train['tax_cluster'] == '8500_to_12000']
    t3 = X_train[X_train['tax_cluster'] == '16000_to_22000']
    t4 = X_train[X_train['tax_cluster'] == '5000_to_6000']
    # t5 =X_train[X_train['tax_cluster']==5]
    # Plot tax for plot clusters
    fig, axes = plt.subplots(3, 2, sharex=False, figsize=(20, 25))
    fig.suptitle('Tax amount by Tax Cluster')
    # axes[0].set_title('All clusters together')
    # axes[1].set_title('Cluster a')
    # axes[2].set_title('Cluster b')
    # axes[3].set_title('Cluster c')
    # axes[4].set_title('Cluster d')
    # axes[5].set_title('Cluster e')
    sns.histplot(data=t0, x='taxamount', alpha=0.5, color='red', ax=axes[0, 0])
    sns.histplot(data=t1, x='taxamount', alpha=0.5,
                 color='orange', ax=axes[0, 0])
    sns.histplot(data=t2, x='taxamount', alpha=0.5,
                 color='yellow', ax=axes[0, 0])
    sns.histplot(data=t3, x='taxamount', alpha=0.5,
                 color='green', ax=axes[0, 0])
    sns.histplot(data=t4, x='taxamount', alpha=0.5,
                 color='blue', ax=axes[0, 0])
    axes[0, 0].ticklabel_format(style='plain')
    axes[0, 0].set_title('Tax clusters superimposed')
    axes[0, 1].ticklabel_format(style='plain')
    axes[0, 1].set_title('Cluster: 1000_to_3000')
    axes[1, 0].ticklabel_format(style='plain')
    axes[1, 0].set_title('Cluster 30000_to_40000')
    axes[1, 1].ticklabel_format(style='plain')
    axes[1, 1].set_title('Cluster 8500_to_12000')
    axes[2, 0].ticklabel_format(style='plain')
    axes[2, 0].set_title('Cluster 16000_to_22000')
    axes[2, 1].ticklabel_format(style='plain')
    axes[2, 1].set_title('Cluster 5000_to_6000')

    sns.histplot(data=t0, x='taxamount', alpha=0.5, color='red', ax=axes[0, 1])
    sns.histplot(data=t1, x='taxamount', alpha=0.5,
                 color='orange', ax=axes[1, 0])
    sns.histplot(data=t2, x='taxamount', alpha=0.5,
                 color='yellow', ax=axes[1, 1])
    sns.histplot(data=t3, x='taxamount', alpha=0.5,
                 color='green', ax=axes[2, 0])
    sns.histplot(data=t4, x='taxamount', alpha=0.5,
                 color='blue', ax=axes[2, 1])
=======
def plot_tax_cluster():
    #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    #labels
    t0 =X_train[X_train['tax_cluster']=='1000_to_3000']
    t1 =X_train[X_train['tax_cluster']=='30000_to_40000']
    t2 =X_train[X_train['tax_cluster']=='8500_to_12000']
    t3 =X_train[X_train['tax_cluster']=='16000_to_22000']
    t4 =X_train[X_train['tax_cluster']=='5000_to_6000']
    #t5 =X_train[X_train['tax_cluster']==5]
    # Plot tax for plot clusters
    fig, axes = plt.subplots(3,2, sharex=False, figsize=(20, 25))
    fig.suptitle('Tax amount by Tax Cluster')
    #axes[0].set_title('All clusters together')
    #axes[1].set_title('Cluster a')
    #axes[2].set_title('Cluster b')
    #axes[3].set_title('Cluster c')
    #axes[4].set_title('Cluster d')
    #axes[5].set_title('Cluster e')
    sns.histplot(data=t0, x='taxamount', alpha=0.5, color='red', ax=axes[0,0])
    sns.histplot(data=t1, x='taxamount', alpha=0.5, color='orange', ax=axes[0,0])
    sns.histplot(data=t2, x='taxamount', alpha=0.5, color='yellow', ax=axes[0,0])
    sns.histplot(data=t3, x='taxamount', alpha=0.5, color='green', ax=axes[0,0])
    sns.histplot(data=t4, x='taxamount', alpha=0.5, color='blue', ax=axes[0,0])
    axes[0,0].ticklabel_format(style='plain')
    axes[0,0].set_title('Tax clusters superimposed')
    axes[0,1].ticklabel_format(style='plain')
    axes[0,1].set_title('Cluster: 1000_to_3000')
    axes[1,0].ticklabel_format(style='plain')
    axes[1,0].set_title('Cluster 30000_to_40000')
    axes[1,1].ticklabel_format(style='plain')
    axes[1,1].set_title('Cluster 8500_to_12000')
    axes[2,0].ticklabel_format(style='plain')
    axes[2,0].set_title('Cluster 16000_to_22000')
    axes[2,1].ticklabel_format(style='plain')
    axes[2,1].set_title('Cluster 5000_to_6000')

    sns.histplot(data=t0, x='taxamount', alpha=0.5, color='red', ax=axes[0,1])
    sns.histplot(data=t1, x='taxamount', alpha=0.5, color='orange', ax=axes[1,0])
    sns.histplot(data=t2, x='taxamount', alpha=0.5, color='yellow', ax=axes[1,1])
    sns.histplot(data=t3, x='taxamount', alpha=0.5, color='green', ax=axes[2,0])
    sns.histplot(data=t4, x='taxamount', alpha=0.5, color='blue', ax=axes[2,1])
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c

    plt.ticklabel_format(style='plain')

    plt.show()

###########################
<<<<<<< HEAD


def taxvaluedollarcnt_corr():
    ''' Runs a correlation test between the age of a home and tax valuation,
    plots a box plot'''
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle3.wrangle()
    # State hypothesis:
    null_hypothesis = "There is no correlation between the taxvaluedollarcnt of a home and the logerror"
    alt_hypothesis = "There is a correlation between the taxvaluedollarcnt of a home and logerror"
    # alpha
    α = .05
    # set x and y
    x = X_train.taxvaluedollarcnt
    y = train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(
        f' The correlation between the taxvaluedollarcnt of a home and the the logerror: {corr:.2f}')
    print(
        f' The P value between the taxvaluedollarcnt of a home and the logerror:  {p:.2f}')
=======
def taxvaluedollarcnt_corr():
    ''' Runs a correlation test between the age of a home and tax valuation,
    plots a box plot'''
    #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    #State hypothesis: 
    null_hypothesis = "There is no correlation between the taxvaluedollarcnt of a home and the logerror"
    alt_hypothesis = "There is a correlation between the taxvaluedollarcnt of a home and logerror"
    #alpha
    α = .05
    # set x and y
    x = X_train.taxvaluedollarcnt
    y= train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(f' The correlation between the taxvaluedollarcnt of a home and the the logerror: {corr:.2f}')
    print(f' The P value between the taxvaluedollarcnt of a home and the logerror:  {p:.2f}')
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
    print(' ')
    if p < α:
        print(f"Reject null hypothesis:\n '{null_hypothesis}'")
        print('\n')
<<<<<<< HEAD
        print(
            f"We now move forward with our alternative hypothesis:\n '{alt_hypothesis}'")
=======
        print(f"We now move forward with our alternative hypothesis: '{alt_hypothesis}'")
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
        print('\n')
        if 0 < corr < .5:
            print("This is a weak positive correlation.")
        elif .5 < corr < 1:
            print("That is a strong positive correlation.")
        elif -.5 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.5:
            print("That is a strong negative correlation.")
<<<<<<< HEAD

    else:
        print("Fail to reject the null hypothesis.")
    # sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    sns.distplot(train.taxvaluedollarcnt, kde=True, color='red')


########
<< << << < HEAD


def structure_dollar_sqft_bin_pearsonr():
    ''' Runs a pearsons r test between the structure_dollar_sqft_bin and logerror,
    plots a box plot'''
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle3.wrangle()
    # State hypothesis:
    null_hypothesis = "There is no correlation between the structure_dollar_sqft_bin and the logerror"
    alt_hypothesis = "There is a correlation between the structure_dollar_sqft_bin and logerror"
    # alpha
    α = .05
    # set x and y
    x = X_train.structure_dollar_sqft_bin
    y = train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(
        f' The correlation between the structure_dollar_sqft_bin and the logerror: {corr:.4f}')
    print(
        f' The P value between the structure_dollar_sqft_bin and the logerror:  {p:.4}')
    print(' ')
    if p < α:
        print(f"Reject null hypothesis:\n '{null_hypothesis}'")
        print('\n')
        print(
            f"We now move forward with our alternative hypothesis:\n '{alt_hypothesis}'")
        print('\n')
        # if 0 < corr < .5:
        # print("This is a weak positive correlation.")
        if 0 < corr < 1:
            print("This is positive correlation with a low p-value.")
        elif -.5 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.5:
            print("That is a strong negative correlation.")

    else:
        print("Fail to reject the null hypothesis.")
        print("")
    # sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    sns.distplot(train.structure_dollar_sqft_bin, kde=True, color='red')


############

== == == =
>>>>>> > fb7365adf857c4517a306b97b171dbf65f1b28da


def calculatedfinishedsquarefeet_pearsonr():
    ''' Runs a pearsons r test between the calculatedfinishedsquarefeet and logerror,
    plots a box plot'''
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle3.wrangle()
    # State hypothesis:
    null_hypothesis = "There is no correlation between the calculatedfinishedsquarefeet and the logerror"
    alt_hypothesis = "There is a correlation between the calculatedfinishedsquarefeet and logerror"
    # alpha
    α = .05
    # set x and y
    x = X_train.calculatedfinishedsquarefeet
    y = train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(
        f' The correlation between the calculatedfinishedsquarefeet and the logerror: {corr:.4f}')
    print(
        f' The P value between the calculatedfinishedsquarefeet and the logerror:  {p:.4}')
    print(' ')
    if p < α:
        print(f"Reject null hypothesis:\n '{null_hypothesis}'")
        print('\n')
        print(
            f"We now move forward with our alternative hypothesis: '{alt_hypothesis}'")
        print('\n')
        # if 0 < corr < .5:
        # print("This is a weak positive correlation.")
        if 0 < corr < 1:
            print("This is positive correlation with a low p-value.")
        elif -.5 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.5:
            print("That is a strong negative correlation.")

    else:
        print("Fail to reject the null hypothesis.")
    # sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    sns.distplot(train.taxvaluedollarcnt, kde=True, color='red')

###########


def scaled_bathroomcnt_pearsonr():
    ''' Runs a pearsons r test between the scaled_bathroomcnt and logerror,
    plots a box plot'''
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle3.wrangle()
    # State hypothesis:
    null_hypothesis = "There is no correlation between the scaled_bathroomcnt and the logerror"
    alt_hypothesis = "There is a correlation between the scaled_bathroomcnt and logerror"
    # alpha
    α = .05
    # set x and y
    x = X_train.scaled_bathroomcnt
    y = train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(
        f' The correlation between the scaled_bathroomcnt and the logerror: {corr:.4f}')
    print(
        f' The P value between the scaled_bathroomcnt and the logerror:  {p:.4}')
=======
    
    else : 
        print("Fail to reject the null hypothesis.")
    #sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    sns.distplot(train.taxvaluedollarcnt, kde=True, color='red')

########
def calculatedfinishedsquarefeet_pearsonr():
    ''' Runs a pearsons r test between the calculatedfinishedsquarefeet and logerror,
    plots a box plot'''
    #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    #State hypothesis: 
    null_hypothesis = "There is no correlation between the calculatedfinishedsquarefeet and the logerror"
    alt_hypothesis = "There is a correlation between the calculatedfinishedsquarefeet and logerror"
    #alpha
    α = .05
    # set x and y
    x = X_train.calculatedfinishedsquarefeet
    y= train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(f' The correlation between the calculatedfinishedsquarefeet and the logerror: {corr:.4f}')
    print(f' The P value between the calculatedfinishedsquarefeet and the logerror:  {p:.4}')
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
    print(' ')
    if p < α:
        print(f"Reject null hypothesis:\n '{null_hypothesis}'")
        print('\n')
<<<<<<< HEAD


<< << << < HEAD
        print(
            f"We now move forward with our alternative hypothesis:\n '{alt_hypothesis}'")
== == == =
        print(
            f"We now move forward with our alternative hypothesis: \n '{alt_hypothesis}'")
        print('\n')
        # if 0 < corr < .5:
            # print("This is a weak positive correlation.")
=======
        print(f"We now move forward with our alternative hypothesis: \n '{alt_hypothesis}'")
        print('\n')
        #if 0 < corr < .5:
            #print("This is a weak positive correlation.")
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
        if 0 < corr < 1:
            print("This is positive correlation with a low p-value.")
        elif -.5 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.5:
            print("That is a strong negative correlation.")
<<<<<<< HEAD

    else : 
        print("Fail to reject the null hypothesis.")
    # sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    # sns.distplot(train.taxvaluedollarcnt, kde=True, color='red')
=======
    
    else : 
        print("Fail to reject the null hypothesis.")
    #sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    #sns.distplot(train.taxvaluedollarcnt, kde=True, color='red')
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c

########
def structure_dollar_sqft_bin_pearsonr():
    ''' Runs a pearsons r test between the structure_dollar_sqft_bin and logerror,
    plots a box plot'''
<<<<<<< HEAD
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    # State hypothesis: 
    null_hypothesis = "There is no correlation between the structure_dollar_sqft_bin and the logerror"
    alt_hypothesis = "There is a correlation between the structure_dollar_sqft_bin and logerror"
    # alpha
=======
    #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    #State hypothesis: 
    null_hypothesis = "There is no correlation between the structure_dollar_sqft_bin and the logerror"
    alt_hypothesis = "There is a correlation between the structure_dollar_sqft_bin and logerror"
    #alpha
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
    α = .05
    # set x and y
    x = X_train.structure_dollar_sqft_bin
    y= train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(f' The correlation between the structure_dollar_sqft_bin and the logerror: {corr:.4f}')
    print(f' The P value between the structure_dollar_sqft_bin and the logerror:  {p:.4}')
    print(' ')
    if p < α:
        print(f"Reject null hypothesis:\n '{null_hypothesis}'")
        print('\n')
        print(f"We now move forward with our alternative hypothesis: \n '{alt_hypothesis}'")
        print('\n')
<<<<<<< HEAD
        # if 0 < corr < .5:
            # print("This is a weak positive correlation.")
=======
        #if 0 < corr < .5:
            #print("This is a weak positive correlation.")
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
        if 0 < corr < 1:
            print("This is positive correlation with a low p-value.")
        elif -.5 < corr < 0:
            print("This is a negative correlation with a low p-value.")
        elif -1 < corr < -.5:
            print("That is a strong negative correlation.")
    
    else : 
        print("Fail to reject the null hypothesis.")
<<<<<<< HEAD
    # sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    # sns.distplot(train.structure_dollar_sqft_bin, kde=True, color='red')
=======
    #sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    #sns.distplot(train.structure_dollar_sqft_bin, kde=True, color='red')
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c

#########
def scaled_bathroomcnt_pearsonr():
    ''' Runs a pearsons r test between the scaled_bathroomcnt and logerror,
    plots a box plot'''
<<<<<<< HEAD
    # get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    # State hypothesis: 
    null_hypothesis = "There is no correlation between the scaled_bathroomcnt and the logerror"
    alt_hypothesis = "There is a correlation between the scaled_bathroomcnt and logerror"
    # alpha
=======
    #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    #State hypothesis: 
    null_hypothesis = "There is no correlation between the scaled_bathroomcnt and the logerror"
    alt_hypothesis = "There is a correlation between the scaled_bathroomcnt and logerror"
    #alpha
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
    α = .05
    # set x and y
    x = X_train.scaled_bathroomcnt
    y= train.logerror
    # run it
    corr, p = stats.pearsonr(x, y)
    print(f' The correlation between the scaled_bathroomcnt and the logerror: {corr:.4f}')
    print(f' The P value between the scaled_bathroomcnt and the logerror:  {p:.4}')
    print(' ')
    if p < α:
        print(f"Reject null hypothesis:\n '{null_hypothesis}'")
        print('\n')
        print(f"We now move forward with our alternative hypothesis: \n '{alt_hypothesis}'")
<<<<<<< HEAD
>>>>>>> fb7365adf857c4517a306b97b171dbf65f1b28da
        print('\n')
        # if 0 < corr < .5:
        # print("This is a weak positive correlation.")
=======
        print('\n')
        #if 0 < corr < .5:
            #print("This is a weak positive correlation.")
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
        if 0 < corr < 1:
            print("This is positive correlation with a low p-value.")
        elif -.5 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.5:
            print("That is a strong negative correlation.")
<<<<<<< HEAD

    else:
        print("Fail to reject the null hypothesis.")
        print("")
    # sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
<<<<<<< HEAD
    sns.distplot(train.scaled_bathroomcntn, kde=True, color='red')
=======
    # sns.distplot(train.scaled_bathroomcnt, kde=True, color='red')
>>>>>>> fb7365adf857c4517a306b97b171dbf65f1b28da
=======
    
    else : 
        print("Fail to reject the null hypothesis.")
    #sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    #sns.distplot(train.scaled_bathroomcnt, kde=True, color='red')
>>>>>>> bdf589374f4fc42fc8dd2c3f449a0535ce90801c
