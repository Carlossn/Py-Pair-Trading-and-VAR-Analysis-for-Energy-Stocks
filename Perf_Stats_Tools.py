########################## Performance Tools ############################
# This utility file contains performance functions to analize returns
#### Method Inputs are described below:
# x = return time series (np.array or list format) in regular format i.e. 1% represented as  0.01 and not 1
# y = benchmark return time series (np.array or list format) in regular format i.e. 1% represented as  0.01 and not 1
# freq = 'daily','weekly','monthly', 'quarterly' or 'annual' => quandl convention
# rf = default risk-free rate (float) => 
# pos = position time series with values -1(short), 0 (neutral) or 1 (long)
# ticker = Quandl Ticker e.g. WIKI/AAPL 
# local_bmk = Google Finance Ticker e.g. SPY = S&P 500 ETF
#
#### Module Methods:
# Metrics:
# profit_loss = % P&L Return or Cumulative Return
# pl_CAGR = % P&L CAGR Return or Annualised Return
# an_vol = annualised volatility
# an_vol = annualised volatility
# an_down_vol = annualised downside volatility
# sharpe = sharpe ratio 
# sortino = sortino ratio
# info_ratio = information ratio
# positive_per =  Percentage of positive periods
# max_draw = Maximum/Worst Drawdown
# max_dd_duration = Maximum/Worst Drawdown Duration
# worst3_draw_avg = Average 3 Worst Drawdown
# num_trades = Number of Trades calculated as number of time
# num_trades_to_periods = Number of trades as % of total bars/periods
# Comprehensive Analysis:
# quick_perf_st = comprehensive performance analysis after running run_strategy() class method
# strategy_factor_analysis = strategy/portfolio multifactor regression using strategy & bmk series input and frequency

### Basic libraries
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib as mpl
import datetime
mpl.rcParams['font.family'] = 'serif'
import numpy as np
import pandas as pd
import scipy.stats as s
import pandas_datareader.data as web

# Performance Metrics Functions: 
# x = strategy return series (np.array or list format)
# y = benchmark return series (np.array or list format)
# freq = 'daily','weekly','monthly', 'quarterly' or 'annual' => quandl convention
# rf = Float. Risk-free rate average during the period. 
# pos = list/array with 1(long) or -1(short) position information 

def profit_loss(x):
	'''% P&L Return or Cumulative Return'''
	return 100*(x.cumsum().apply(np.exp)[-1]-1)

def pl_CAGR(x):
	'''% P&L CAGR Return or Annualised Return'''
	y = float((x.index[-1]-x.index[0]).days)/365
	return 100*(x.cumsum().apply(np.exp)[-1]**(1/y)-1)

def an_vol(x,freq):
	''' Annualised Vol
	'''        
	if freq == 'daily':
		an = 365
	elif freq == 'weekly':
		an = 52
	elif freq == 'monthly':
		an = 12
	elif freq == 'quarterly':
		an = 4
	else:
		an = 1
	return 100*(np.std(x)*np.sqrt(an))

def an_down_vol(x,freq):
	''' Annualised Downside Vol'''
	if freq == 'daily':
		an = 365
	elif freq == 'weekly':
		an = 52
	elif freq == 'monthly':
		an = 12
	elif freq == 'quarterly':
		an = 4
	else:
		an = 1
	return 100*(np.std(x.loc[x<0])*np.sqrt(an))

def sharpe(x,freq,rf):
	''' Sharpe Ratio'''
	if freq == 'daily':
		an = 365
	elif freq == 'weekly':
		an = 52
	elif freq == 'monthly':
		an = 12
	elif freq == 'quarterly':
		an = 4
	else:
		an = 1
	return (pl_CAGR(x)-rf)/(an_vol(x,freq))
    
def sortino(x,freq,rf):
	''' Sharpe Ratio'''
	if freq == 'daily':
		an = 365
	elif freq == 'weekly':
		an = 52
	elif freq == 'monthly':
		an = 12
	elif freq == 'quarterly':
		an = 4
	else:
		an = 1    
	return (pl_CAGR(x)-rf)/(an_down_vol(x,freq))

def info_ratio(x,y,freq):
	''' Information Ratio'''
	if freq == 'daily':
		an = 365
	elif freq == 'weekly':
		an = 52
	elif freq == 'monthly':
		an = 12
	elif freq == 'quarterly':
		an = 4
	else:
		an = 1    
	return (pl_CAGR(x)-pl_CAGR(y))/an_vol(x-y,freq)

def positive_per(x):
    ''' Percentage of positive periods'''
    return round(100*float(len(x.loc[x>0]))/len(x),2)

def max_draw(x):
    ''' Maximum/Worst Drawdown'''
    cumret = x.cumsum().apply(np.exp)
    cummax = cumret.cummax()
    drawdown = abs(cumret/cummax-1).dropna()
    return 100*(drawdown.max())

def max_dd_duration(x):
    ''' Maximum/Worst Drawdown Duration'''
    cumret = x.cumsum().apply(np.exp)
    cummax = cumret.cummax()
    drawdown = abs(cumret/cummax-1).dropna()
    temp = drawdown[drawdown.values == 0]
    periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
    return periods.max()

def worst3_draw_avg(x):
    ''' Average 3 Worst Drawdown'''
    cumret = x.cumsum().apply(np.exp)
    cummax = cumret.cummax()	
    drawdown = abs(cumret/cummax-1).dropna()
    return 100*(np.mean(sorted(drawdown, reverse=True)[0:3]))

def num_trades(pos):
    ''' Number of Trades calculated as number of time '''
    return len(pos)-sum((pos.diff().dropna()==0))

def num_trades_to_periods(pos):
    ''' Number of trades as % of total bars/periods'''
    return round(float(num_trades(pos))/len(pos),4)

def quick_perf_st(x, y, freq, rf):
    ''' Comprehensive Performance Analysis after running run_strategy() class method'''
    dfx = x.dropna()
    dfy = y.dropna()
    stats = {'Statistics': ['P&L', 'CAGR','Anual_Vol', '%_Positive',
                            'Skew','Kurtosis','Kurtosis PV','Downside_Vol', 'Worst',
                            'Sharpe_Ratio', 'Sortino_Ratio', 'Information_Ratio', 
                            'Max_Drawdown','Worst_3_drawdown_avg', 'Max_DD_Duration'], 
            'Strategy': [profit_loss(dfx), pl_CAGR(dfx), an_vol(dfx,freq), positive_per(dfx),
                             s.skew(dfx),s.kurtosis(dfx), s.kurtosistest(dfx)[1], an_down_vol(dfx,freq), dfx.min(),
                            sharpe(dfx,freq,rf), sortino(dfx,freq,rf), info_ratio(dfx,dfy,freq),
                            max_draw(dfx), worst3_draw_avg(dfx),max_dd_duration(dfx)],
            'Benchmark': [profit_loss(dfy), pl_CAGR(dfy), an_vol(dfy,freq), positive_per(dfy),
                             s.skew(dfy),s.kurtosis(dfy), s.kurtosistest(dfy)[1], an_down_vol(dfy,freq), dfy.min(),
                             sharpe(dfy,freq,rf), sortino(dfy,freq,rf), info_ratio(dfy,dfy,freq),
                             max_draw(dfy), worst3_draw_avg(dfy), max_dd_duration(dfy)] 
                             }

    df = pd.DataFrame(stats)
    df.set_index('Statistics', inplace=True)
    return df.round(4)


def strategy_factor_analysis(series,bmk,freq='daily', US_factors=True):
    '''
    Returns multifactor performance analysis regression results and coefficient/pvalue charts. 
    
    Parameters
    ----------
    series = Series format. Strategy returns to be analized in decimal format i.e. 0.1 instead of 10
    bmk = Series format. Benchmark (Geographic or Industry Index) to be used in decimal format and described as LOC in
    the factor analysis output.
    freq = daily default. The user can choose between daily, monthly or quarterly return analysis for the analysis.
    US_Factors = True default. If "True" Fama-French Original factors for the North American market are used, otherwise 
    Fama-French Global factors are used for the analysis. 
    
    more info: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2287202
    
    '''
    
    import statsmodels.formula.api as sm
    x = series
    y = bmk

    # 1) LOAD DATA:
    start = x.index[0] # TR dependent variable
    end = x.index[-1] # TR local benchmark
    # 5 Factor:
    # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2287202
    if US_factors==True:
        ff_df= web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",start=start, end=end)[0]
        mom_df = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",start=start, end=end)[0]
    else:
        ff_df = web.DataReader("Global_5_Factors_daily", "famafrench",start=start, end=end)[0]
        mom_df = web.DataReader("Global_Mom_Factor_daily", "famafrench",start=start, end=end)[0]
        # next index transforms onlt required for US_factors=False
        ff_df.index = ff_df.index.to_timestamp() # pandas.tseries.period.PeriodIndex => pandas.tseries.index.DatetimeIndex
        mom_df.index = mom_df.index.to_timestamp()

    # Tidy dataframes:
    ff_df.index.rename(None,inplace=True)
    ff_df['MKT'] = ff_df['RF'] + ff_df['Mkt-RF']   
    mom_df.index.rename(None,'ignore')
    # Local Factor: Local market or Industry risk premium
    local_df = pd.DataFrame({'y': y*100})
    local_df.rename(columns={'y':'LOC_ret'}, inplace=True)
    local_df.index.rename(None,'ignore')

    # # FX Factor => UUP = long USD ETF 
    usd_df = web.DataReader('UUP', data_source='yahoo',start=start, end=end)[['Adj Close']]
    usd_df= pd.DataFrame(np.log(usd_df['Adj Close'] / usd_df['Adj Close'].shift(1)) *100)
    usd_df.rename(columns={'Adj Close':'USD'}, inplace=True)
    usd_df.index.rename(None,'ignore')
    # Dependent Var: Stock Price Total Return
    TR = pd.DataFrame({'x':x*100})
    TR.index.rename(None,inplace=True)
    TR.rename(columns={'x':'TR'}, inplace=True)

    ## 2) MERGE
    # Merge: ff & mom
    ff_df = pd.merge(ff_df,mom_df,how='inner',left_index=True,right_index=True)
    if US_factors==True:
        ff_df.rename(columns={'Mom   ':'MOM'},inplace=True)
    else:
        ff_df.rename(columns={'WML':'MOM','Mom':'MOM'},inplace=True)
    # Merge ff & local
    ff_df = pd.merge(ff_df,local_df,how='inner',left_index=True,right_index=True)
    # Merge ff & fx
    ff_df = pd.merge(ff_df,usd_df,how='inner',left_index=True,right_index=True)
    # Merge Dependent var(TR) and Explanatory Variables(ff_df):
    df = pd.merge(TR.dropna(),ff_df.dropna(), how='inner',left_index=True,right_index=True)

    ## 3) RISK PREMIUMS:
    # Calculate TR RP (Dependent Variable):
    df['TR-RF'] = df['TR'] - df['RF']
    # Calculate USD RP
    df['USD-RF'] = df['USD']-df['RF']
    # Calculate LOC RP
    df['LOC'] = df['LOC_ret']-df['MKT']
    df['LOC-RF'] = df['LOC']-df['RF']
    fr = 'D' if freq=='daily' else 'W-Mon' if freq=='weekly' else 'M' if freq=='monthly' else 'Q'
    df = df.resample(fr).last() # options are "W"=weekly, "M"=monthly, "W-Mon"=weekly starting on Mondays

    # 3) MULTIVARIATE ANALYSIS
    RF =  list(df.loc[:,'TR-RF']) # Risk-Free Rate Premium
    MKT = list(df.loc[:,'Mkt-RF']) # Market Premium
    SMB = list(df.loc[:,'SMB']) # Size Premium
    HML = list(df.loc[:,'HML']) # Value Premium
    RMW = list(df.loc[:,'RMW']) # Quality Premium (High Op. Profitability Premium) 
    CMA = list(df.loc[:,'CMA']) # Low CapEx Premium
    MOM = list(df.loc[:,'MOM']) # Momentum Premium
    LOC = list(df.loc[:,'LOC-RF']) # Local Premium
    USD = list(df.loc[:,'USD-RF']) # USD FX Premium
    TR = list(df.loc[:,'TR-RF']) # Strategy Premium
    dfr = pd.DataFrame({'RF':RF,'MKT':MKT,'SMB':SMB,'HML':HML,'RMW':RMW,'CMA':CMA,'MOM':MOM,'LOC':LOC,'USD':USD, 'TR':TR})
    result = sm.ols(formula="TR ~ MKT + SMB + HML + RMW + CMA + MOM + LOC + USD", data=dfr).fit()
    print(result.summary())
    plt.subplot(2,2,1)
    result.params.plot('bar', title='Factors - Coefficients',figsize=(15,10))
    plt.subplot(2,2,2)
    result.pvalues.plot('bar',title='Factors - P-Values')
    plt.axhline(0.1)
    return result