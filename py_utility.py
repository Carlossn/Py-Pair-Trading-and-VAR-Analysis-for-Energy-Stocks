# basic Libraries:
import types
import IPython
import numpy as np
import pandas as pd
import itertools

# Data and ML libraries:
import pandas_datareader.data as web  # get data for factors
import scipy.stats as stats
from scipy.stats import friedmanchisquare
import statsmodels as sm
import statsmodels.api as sfm
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.tsatools import lagmat

# Plot Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab

# import datetime
import zmq
import datetime as dt


def Data_bulk(tickers, start_, end_, hdfs_name, price_only=True, source='yahoo'):
    '''
    Given a list of yahoo tickers, the function dowloads data and either stores into HDFS file or appends the
    the data to an existing HDFS database-like file easy to access via pandas or Pytables.

    This function solves issues inherent in Datareader function related to bulk download of tickers.

    Params
    ------
    tickers = list of tickers
    start_ = start date
    end_ = end date
    price_only = True. If True it only yields "Adj Close" data per each ticker.
    hdfs_name = h5 file name to store each ticker. Every ticker will have a different key assigned.
    '''
    store = pd.HDFStore(hdfs_name + '.h5')  # opens or create a new hdfs file name if is not existent.

    yr_diff = dt.datetime.strptime(end_, '%Y-%M-%d').year - dt.datetime.strptime(start_, '%Y-%M-%d').year
    dt_idx = pd.date_range(start_, end_, periods=yr_diff).date  # start_date to test and grab the first available

    for t in tickers:
        try:
            for d in dt_idx:  # test inception dates with data
                try:
                    data = web.DataReader(t, data_source=source, start=d, end=end_)
                    break
                except:
                    continue
        except:
            print(t, 'error: data not available')

        if price_only == True:
            data = data[['Adj Close']]

        data = data.astype('float64')  # transform to float64 to avoid compatibility problems for future appending to h5
        store.append(t, data, 'table')  # store or append in h5 file
    store.close()


def data_load(transform=True):
    '''
    Uploads 'cqf_data_df.csv' price level data from the notebook directory and, optionally, calculates
    differences, returns and cumulative returns. Output: dataframe format.

    Parameters
    ----------
    transform=True. If "True" it calculates and returns a dataframe with price levels, differences, returns and cumulative returns
    '''
    df_ = pd.read_csv('cqf_data_df.csv', index_col=0, parse_dates=True)
    df_.index.rename(None, inplace=True)
    if transform == True:
        for i in df_.columns:
            df_[i + '_d'] = df_[i] - df_[i].shift(1)
            df_[i + '_r'] = np.log(df_[i]).diff().dropna()
            df_[i + '_c'] = df_[i + '_r']
            df_[i + '_c'][0] = 1
            df_[i + '_c'] = df_[i + '_c'].cumsum()
    return df_


def adv_describe(df, alpha=0.05):
    '''
    pandas describe with kurtosis and skewness
    parameters
    ---------
    df = dataframe with series data
    alpha = significance level for normality test
    normality test based on D'Agostino and Pearson's  test that combines skew and kurtosis to produce an omnibus test of normality.'
      '''
    df_ = df.dropna()
    df_des = df_.describe().round(5)
    des = stats.describe(df_)
    df_des.loc['skew'] = des.skewness
    df_des.loc['kurt'] = des.kurtosis
    pvals = np.round(stats.normaltest(df_)[1] > alpha, decimals=0)
    df_des.loc['normal?'] = ['Yes' if x == True else "No" for x in pvals]
    print('normal test based on D\'Agostino and Pearson\'s  test that combines skew and kurtosis to produce an omnibus test of normality. ')
    return df_des


def heatmap_corr(df, cent=None, dropDuplicates=True):
    '''
    Returns correlation heatmap

    Parameters
    ----------
    df = dataframe containing time series data for several series
    cent=None. If it's a number it will consider that given correlation as the centre to calculate heatmap colors
    dropDuplicates=True. If "True" only the lower-left part of the correlation matrix will be ploted.

    '''
    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    # Set background color / chart style
    sns.set_style(style='white')
    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    # Draw correlation plot with or without duplicates
    if dropDuplicates == True:
        sns.heatmap(df, mask=mask, cmap=cmap, center=cent,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)


def friedman_similarity_test(df_, reg='_r'):
    '''
    Returns multiple distribution tests statistics and p-values for a list of variables (input in string format)
    Tests performed are:
     - MW_test = Mann-Whitney U
     - W_test = Wilcoxon
     - K_test = Kruskal
     - F_test = Friedman
    '''
    from scipy.stats import friedmanchisquare
    import pandas as pd
    import numpy as np

    var_df = df_.filter(regex=reg)
    var_names = var_df.columns
    nvar = len(var_names)
    # create variables:
    for i in var_df.columns:
        globals()[i] = var_df[i]

    names = ', '.join(var_names)

    print('Ho: Same Distribution')

    F_test = eval('friedmanchisquare(' + names + ')')

    # deleter variable names to free memory space:
    for i in df_.filter(regex=reg).columns:
        globals()[i] = None

    return F_test


def adf_test(df):
    '''
    Returns dataframe with ADF (augmented Dickey-Fuller. Ho: unit root = Non-Stationary) stationarity test p-value

    Parameters
    -----------
    df = Dataframe or Series.
    '''
    nobs = df.shape[0]
    max_l = int(round((12 * nobs / 100)**(0.25), 0))  # Schwert (1989) optimal max_lag ADF = (12*T/100)^0.25
    df_ = df.dropna()
    try:
        df_pv = pd.DataFrame()
        for i in df_.columns:
            temp = sm.tsa.stattools.adfuller(df_[i], maxlag=max_l, regression='nc')
            df_pv = df_pv.append(pd.DataFrame({'ADF Stat': temp[0], 'ADF p-value': [temp[1]], 'Lag': temp[2],
                                               '1% CV': temp[4]['1%'], '5% CV': temp[4]['5%'], '10% CV': temp[4]['10%']},
                                              index=[i]))
    except:
        temp = sm.tsa.stattools.adfuller(df_, maxlag=max_l)
        df_pv = df_pv.append(pd.DataFrame({'ADF Stat': temp[0], 'ADF p-value': [temp[1]], 'Lag': temp[2],
                                           '1% CV': temp[4]['1%'], '5% CV': temp[4]['5%'], '10% CV': temp[4]['10%']},
                                          index=['Series']))

    print('Reject Null Hypothesis: Unit Root (Non-Stationary) for p-values < threshold (e.g. 0.01 or 0.05)')
    return round(df_pv, 4)


def p_VAR(df, p, constant=True):
    '''
    Return VAR model with p lags (VAR(p) model) using linear algebra: 
     Y = BZ + U
     Therefore:
     B = (Z'Z)^(-1)8(Z'Y) is the matrix with VAR(p) coefficient estimates
    Where if constant=True:
    Y = K x (T+1) matrix
    B = K x (K*P +1) matrix
    Z = (K*P + 1) x (T +1) matrix 
    U = K x (T+1) matrix = model error 

    Parameters
    -----------
    df = dataframe with time series variables
    p = number of optimal logs for implementing VAR

    '''
    # clean data
    y_p = df.dropna()
    y_p = np.transpose(np.matrix(y_p))
    y = y_p[:, p:]
    # parameters:
    k_p = y_p.shape[0]
    T_p = y_p.shape[1]
    k = y.shape[0]
    T = y.shape[1]
    # Z matrix = explanatory lagged variables
    Z = list()
    for i in range(1, p + 1):
        for j in range(0, k):
            col = np.array(y_p[j, p - i:T_p - i])
            Z.append(col)
    Z = np.matrix([Z[i][0] for i in range(len(Z))])
    if constant == True:
        Z = np.vstack((np.ones(T), Z))
    # solving for B:
    B = ((Z * Z.T).I * (Z * y.T)).T
    # dataframe:
    if constant == True:
        idx = list(['const'])
    else:
        idx = list()

    cols = df.filter(regex='_r').columns
    for i in range(1, p + 1):
        for j in df.filter(regex='_r').columns:
            idx.append('L' + str(i) + '.' + j)

    B_df = pd.DataFrame(B.T, columns=cols, index=idx)
    return B_df


def coint_test_bulk(df, start, end, alpha=0.01, max_d=4, max_lag=None):
    '''
    Returns dataframe with Engle Granger cointegration test results. OLS regression uses constant.
    Two different ADF statistics returned: "ADF stat" and "ADF stat sm" with the latter calculated from the 
    function "adfuller" from Python's library StatsModels to compare results.
    Remember ADF test with Null Ho: Unit Root exists (non-stationary)

    Parameters
    ----------
    df = dataframe with columns as variables to test cointegration
    start = start date. Format 'YYYY-MM_DD'
    end = end date. Format 'YYYY-MM_DD'
    alpha = default 1%.  significance level for ADF test p-values.
    max_d= 4. Maximum number of difference transformations to test for ADF test.
    max_lag = None. Maximum number of lags to be used in ADF test. If None Schwert (1989) optimal max_lag ADF is used.

    '''
    if max_lag == None:
        nobs = int(df.shape[0])
        max_lag = int(round((12 * nobs / 100)**(0.25), 0))  # Schwert (1989) optimal max_lag ADF = (12*T/100)^0.25

    data = df.loc[start:end]
    combo = list(itertools.permutations(data.columns, 2))
    df_out = pd.DataFrame()
    for y, x in combo:
        for d in range(0, max_d + 1):
            # define variables:
            if d == 0:
                y_t = data[y].dropna()
                x_t = data[x].dropna()
                x_t = add_constant(x_t)  # add intercept = columns of 1s to x_t
            else:
                y_t = data[y].diff(d).dropna()
                x_t = data[x].diff(d).dropna()
                x_t = add_constant(x_t)  # add intercept = columns of 1s to x_t

            # OLS regression:
            ols = OLS(y_t, x_t).fit()  # validate result with statsmodels
            res = ols.resid
            res_diff = np.diff(res)
            adf_df = pd.DataFrame()
            # optimal lag choice:
            for l in range(1, max_lag + 1):
                res_dlags = lagmat(res_diff[:, None], l, trim='both', original='in')  # each row is a date with k lags
                n = res_dlags.shape[0]  # number of obs
                res_dlags[:, 0] = res[-n - 1:-1]  # replace first obs in each date (lag=0) with the "price" level of residual linked to that lag
                dy_t = res_diff[-n:]  # dependent variable ADF test
                dx_t = res_dlags  # independent variable ADF test
                ols_adf = OLS(dy_t, dx_t).fit()
                adf_sm = sm.tsa.stattools.adfuller(res, maxlag=l, regression='nc', autolag=False,)  # sm adf test function
                adf_df = adf_df.append({'AIC': ols_adf.aic, 'BIC': ols_adf.bic, 'ADF Lags': l, 'ADF Stat': ols_adf.tvalues[0], 'ADF Stat sm': adf_sm[0],
                                        '1%CV': adf_sm[4]['1%'], '5%CV': adf_sm[4]['5%'], '10%CV': adf_sm[4]['10%'],
                                        'ADF P-Value': adf_sm[1], 'Diff': d, 'Index': l}, ignore_index=True)
            adf_df.set_index('Index', inplace=True)
            adf_df.index.rename(None, inplace=True)
            best_model = adf_df.sort_values('AIC').iloc[0, :]
            df_out = df_out.append(pd.DataFrame({'y': [y], 'x': [x], 'diff': d, 'ADF Lags': best_model['ADF Lags'],
                                                 'ADF stat': round(best_model['ADF Stat'], 1), 'ADF stat sm': round(best_model['ADF Stat sm'], 1),
                                                 '1%CV': best_model['1%CV'], '5%CV': best_model['5%CV'], '10%CV': best_model['10%CV'],
                                                 'ADF p-value': best_model['ADF P-Value'],
                                                 'Cointegrated': best_model['ADF P-Value'] < alpha},
                                                index=[[(y, x) if d == 0 else (y + '_d' + str(d), x + '_d' + str(d))]]))
            if best_model['ADF P-Value'] > alpha:  # PV>alpha=> Ho: Unit Root Exists cannot be rejected => Non-stat => difference
                continue  # continue the loop and differentiate once more
            else:
                break  # y,x are cointegrated so we break the diff loop and go to a new y,x pair

    return df_out.round(3)
