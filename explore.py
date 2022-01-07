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


def find_k(X_train, cluster_vars, k_range):
    '''plots k for intertia, pct_delta, & delta - aid in finding K target for 
    clustered features applied to function'''

    #enter clusters/features, cluster_name, & range    
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

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

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
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
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

    return k_comparisons_df

################################
def plot_size_clusters():
        #get data from wrangle
    train, X_train, y_train, X_validate, y_validate, X_test, y_test=wrangle3.wrangle()
    #
    s0=X_train[X_train['size_cluster']=='a']
    s1=X_train[X_train['size_cluster']=='b']
    s2=X_train[X_train['size_cluster']=='c']
    s3=X_train[X_train['size_cluster']=='d']
    s4=X_train[X_train['size_cluster']=='e']
    s5=X_train[X_train['size_cluster']=='f']
    s6=X_train[X_train['size_cluster']=='g']

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
    axes[0,1].set_title('Cluster a')

    axes[1,0].ticklabel_format(style='plain')
    axes[1,0].set_title('Cluster b')
    axes[1,1].ticklabel_format(style='plain')
    axes[1,1].set_title('Cluster c')
    axes[2,0].ticklabel_format(style='plain')
    axes[2,0].set_title('Cluster d')
    axes[3,0].ticklabel_format(style='plain')
    axes[3,0].set_title('Cluster e')
    axes[3,1].ticklabel_format(style='plain')
    axes[3,1].set_title('Cluster f')

    sns.histplot(data=s0, x='calculatedfinishedsquarefeet', alpha=0.5, color='red', ax=axes[0,1])
    sns.histplot(data=s1, x='calculatedfinishedsquarefeet', alpha=0.5, color='orange', ax=axes[1,0])
    sns.histplot(data=s2, x='calculatedfinishedsquarefeet', alpha=0.5, color='yellow', ax=axes[1,1])
    sns.histplot(data=s3, x='calculatedfinishedsquarefeet', alpha=0.5, color='green', ax=axes[2,0])
    sns.histplot(data=s4, x='calculatedfinishedsquarefeet', alpha=0.5, color='blue', ax=axes[2,1])
    sns.histplot(data=s5, x='calculatedfinishedsquarefeet', alpha=0.5, color='purple', ax=axes[3,0])
    sns.histplot(data=s6, x='calculatedfinishedsquarefeet', alpha=0.5, color='pink', ax=axes[3,1])

    plt.ticklabel_format(style='plain')

    plt.show()


################################
def plot_prices_clusters():
    ''' '''
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
    print(' ')
    if p < α:
        print(f"Reject null hypothesis:\n '{null_hypothesis}'")
        print('\n')
        print(f"We now move forward with our alternative hypothesis: '{alt_hypothesis}'")
        print('\n')
        if 0 < corr < .5:
            print("This is a weak positive correlation.")
        elif .5 < corr < 1:
            print("That is a strong positive correlation.")
        elif -.5 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.5:
            print("That is a strong negative correlation.")
    
    else : 
        print("Fail to reject the null hypothesis.")
    sns.boxplot(y='logerror', x ='taxvaluedollarcnt', data = train, palette='Set2')
    #sns.distplot(train.taxvaluedollarcnt, kde=True, color='red')