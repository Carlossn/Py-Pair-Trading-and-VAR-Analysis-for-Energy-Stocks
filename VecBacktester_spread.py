# Python Module with Class for Vectorized Backtesting
# Mean-Reversion-based Strategies
#
import numpy as np
import pandas as pd
import datetime
import quandl as q
from pandas_datareader import data as web
from Perf_Stats_Tools import *
from scipy.optimize import brute
import scipy.stats as stats
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS


class MRVectorBacktester(object):
    ''' Class for the vectorized backtesting of Momentum-based trading strategies.

    Attributes
    ==========
    x_series: datetime series
        Introduce dependent security price to trade
    y_series: datetime series
        Introduce independent security price to trade
    bmk_series: str
        Introduce benchmark security price to trade
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    start_in: str
        start date for spread coefficients calculation
    end_in: str
        end date for spread coefficients calculation    
    freq: str
        data frequency: daily, weekly, monthly, quarterly, annual
    trans: str
        data transformation: diff, rdiff, cumul, and normalize
    ctf: float 
        overall equity execution fees per trade (buy or sell) => trading fee + bid-offer + slippage (delay, broker fees, etc)
        Damodaran: Overall Costs US Large Cap = 0.3%-0.4% vs 3.8%-5.8% US LAGll Cap => ctf = 2% default assumption is reasonable
        http://people.stern.nyu.edu/adamodar/pdfiles/invphiloh/tradingcosts.pdf
    rf: float
        Annual Risk-Free Rate Assumption
    sd: int, defined in run_strategy()
        Standard Deviation times
    slip: int, defined in run_strategy()
        slippage in periods. Forinstance, slip=1 means the strategy will enter/exit into the trade one day after the signal is flagged

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    params:
        returns spread and zspread along with parameters necessary for pair spread trading strateg
    set_parameters:
        sets one or two new parameters and start/end periods
    run_strategy:
        Defines self.sma and self.sd to run the backtest for the Mom-based strategy
    perf_stats:
        creates df table with performance stats using run_strategy() last results
    plot_data:
        plots stock price and trading input
    plot_results:
        plots the performance of the strategy compared to the symbol
    plot_drawdown:
        Plots Drawdown metrics (DD and DD Periods) along with a rolling DD line chart and histogram
    update_and_run:
        updates lag parameters and returns the (negative) absolute performance
    optimize_parameters:
        implements a brute force optimizeation for the two lag parameters for the current time period entered in the instance
    optimal_param_multiper():
        Splits sample into multiple periods, gets optimal params for each time period and calculates out-of-the-sample 
        perf metrics for each params set to understand which set of params is more persistent.
    '''

    def __init__(self, y_series, x_series, bmk_series, start, end, start_in, end_in, freq='daily', trans=None, ctf=0.0017, rf=0.02):
        self.y_series = y_series
        self.x_series = x_series
        self.bmk_series = bmk_series
        self.start = start
        self.end = end
        self.start_in = start_in
        self.end_in = end_in
        self.freq = freq  # IMPORTANT BUG: for whatever reason default freq='daily' not working so when instantiating needs to be declare explicitly
        self.trans = trans  # data transformation
        self.ctf = ctf  # trading fees
        self.rf = rf
        self.sma = None
        self.sd = None
        self.results = None
        self.ratio_opt = None
        self.get_data()  # allows to download the data as a new instance is created

    def param(self):
        '''
        Returns spread from regression and other parameters:
        y = constant + beta*x
        '''
        y_t = self.y_series.loc[self.start_in:self.end_in].dropna()
        x_t = self.x_series.loc[self.start_in:self.end_in].dropna()
        x_t = add_constant(x_t)  # add intercept = columns of 1s to x_t
        # OLS regression: Static Equilibrium Model
        ols = OLS(y_t, x_t).fit()  # validate result with statsmodels
        c = ols.params[0]
        b = ols.params[1]
        x_t = x_t.iloc[:, 1:]  # exclude constant as it will be accounted in c
        res = y_t - c - b * x_t[x_t.columns[0]]
        # OLS regression: OU SDE Solution Regression: e_t = C + B*et_1 + eps_t_tau
        res_t = res[1:]
        res_t_1 = res.shift(1).dropna()
        x = add_constant(res_t_1)  # add intercept = columns of 1s to x_t
        x.rename(columns={0: 'res_t_1'}, inplace=True)
        ols_r = OLS(res_t, x).fit()
        # Backtesting Parameters
        mu_e = ols_r.params[0] / (1 - ols_r.params[1])  # equilibrium level = C/(1-B)
        tau = 1 / 252  # daily data frequency
        theta = - np.log(ols_r.params[1]) / tau  # speed of reversion = - log(B)/tau
        half_l = np.log(2) / theta  # half life
        sigma_OU = np.sqrt(2 * theta * np.var(ols_r.resid) / (1 - np.exp(-2 * theta * tau)))  # diffusion over small time scale (volatility coming from small ups and downs of BM)
        sigma_eq = sigma_OU / np.sqrt(2 * theta)  # use to determine exit/trading points = mu_e +/- sigma_eq
        # Backtesting Spread:
        y_t = self.y_series.loc[self.start:self.end].dropna()
        x_t = self.x_series.loc[self.start:self.end].dropna()
        spread = y_t - c - b * x_t  # using new dates
        # output: spread and parameters
        dic = {'price': spread, 'mu_e': [mu_e], 'tau': tau, 'theta': theta,
               'sigma_OU': sigma_OU, 'sigma_eq': sigma_eq, 'b': b}

        return dic

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        # price data
        raw = pd.DataFrame({'price_y': self.y_series, 'price_x': self.x_series, 'price_bmk': self.bmk_series,
                            'price': self.param()['price'], 'b': self.param()['b']})
        raw['return_y'] = np.log(raw['price_y'] / raw['price_y'].shift(1)).fillna(0)
        raw['return_x'] = np.log(raw['price_x'] / raw['price_x'].shift(1)).fillna(0)
        raw['return'] = raw['return_y'] - raw['b'] * raw['return_x']
        raw['bmk_return'] = np.log(raw['price_bmk'] / raw['price_bmk'].shift(1)).fillna(0)
        self.data = raw.round(4)

    def run_strategy(self, sd=1, slip=1):
        ''' 
        Backtests pair trading strategy.
        Params
        ------- 
        sd  = 1 default. Multiplier to be applied to sigma_eq to trade the spread.
        slip = 1 day default. Lag between signal and trade execution.
        Output:
        'Abs Net P&L | Abs Net P&L vs bmk | An_Vol | Sharpe'
        '''
        self.sd = sd
        self.slip = slip
        data = self.data
        price = self.param()['price']
        mu = self.param()['mu_e'][0]
        sigma = self.param()['sigma_eq']
        data['sma'] = mu
        data['dist'] = data['price'] - data['sma']
        data['uv_level'] = mu - sigma * sd
        data['ov_level'] = mu + sigma * sd
        data = data.copy().dropna()
        # positions
        data['position'] = np.where(data['dist'].shift(1) * data['dist'] < 0, 0, np.nan)
        data['position'] = np.where(data['price'] > data['ov_level'], -1, data['position'])  # sell signals
        data['position'] = np.where(data['price'] < data['uv_level'], 1, data['position'])  # buy signals
        data['position'].ffill(inplace=True)  # fill forward na values.
        data['position'].shift(self.slip)  # enter slippage assumption
        data['position'].fillna(0, inplace=True)  # fill na gaps.
        # returns:
        data['strategy'] = (data['position'] * data['return']) + 0  # add zero to avoid negative zeros
        data['fees'] = np.where(data['position'] == data['position'].shift(1), 0, self.ctf)
        data['fees'] = data['fees'].fillna(0)
        data['fees'][0] = int(0)  # first obs as it was a bug from the former rule.
        data['net_strategy'] = (data['strategy'] - data['fees']) + 0  # add zero to avoid negative zeros.
        data['creturns'] = data['return'].cumsum().apply(np.exp)  # long spread buy-and-hold
        data['cbmkreturns'] = data['bmk_return'].cumsum().apply(np.exp)  # long bmk buy-and-hold
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)  # our strategy
        data['cnetstrategy'] = data['net_strategy'].cumsum().apply(np.exp)  # our strategy after fees.
        self.results = data
        # absolute performance of the strategy
        aperf = data['cstrategy'].ix[-1] - 1
        aperf_net = data['cnetstrategy'].ix[-1] - 1
        # out-/underperformance vs bmk
        operf = aperf - (data['cbmkreturns'].ix[-1] - 1)
        operf_net = aperf_net - (data['cbmkreturns'].ix[-1] - 1)
        # vol and sharpe:
        a_vol = an_vol(data['net_strategy'], self.freq)
        sharpe_ratio = sharpe(data['net_strategy'], self.freq, self.rf)
        return {'strat cum P&L': round(aperf_net, 4), 'cum perf vs bmk': round(operf_net, 4), 'strat vol': round(a_vol, 4), 'strat sharpe': round(sharpe_ratio, 4)}

    def perf_stats(self):
        ''' Comprehensive Performance Analysis after running run_strategy() class method.
        '''
        df = self.results.dropna()
        rf = self.rf
        freq = self.freq
        stats = {'Statistics': ['Gross_P&L', 'Net_P&L', 'Gross_CAGR', 'Net_CAGR', 'Net_Anual_Vol', '%_Positive',
                                'Skew', 'Kurtosis', 'Kurtosis PV', 'Downside_Vol', 'Worst_Net',
                                'Sharpe_Ratio', 'Sortino_Ratio', 'Information_Ratio',
                                'Max_Drawdown', 'Worst_3_drawdown_avg', 'Max_DD_Duration',
                                'NumTrades', 'Num_Trades-to_Periods'],
                 'Buy&Hold': [profit_loss(df['return']), profit_loss(df['return']), pl_CAGR(df['return']), pl_CAGR(df['return']),
                              an_vol(df['return'], freq), positive_per(df['return']),
                              s.skew(df['return']), s.kurtosis(df['return']), s.kurtosistest(df['return'])[1],
                              an_down_vol(df['return'], freq), df['return'].min(),
                              sharpe(df['return'], freq, rf), sortino(df['return'], freq, rf),
                              info_ratio(df['return'], df['bmk_return'], freq),
                              max_draw(df['return']), worst3_draw_avg(df['return']), max_dd_duration(df['return']),
                              0, 0],

                 'Strategy': [profit_loss(df['strategy']), profit_loss(df['net_strategy']), pl_CAGR(df['strategy']), pl_CAGR(df['net_strategy']),
                              an_vol(df['net_strategy'], freq), positive_per(df['net_strategy']),
                              s.skew(df['net_strategy']), s.kurtosis(df['net_strategy']), s.kurtosistest(df['net_strategy'])[1],
                              an_down_vol(df['net_strategy'], freq), df['net_strategy'].min(),
                              sharpe(df['net_strategy'], freq, rf), sortino(df['net_strategy'], freq, rf),
                              info_ratio(df['net_strategy'], df['bmk_return'], freq),
                              max_draw(df['net_strategy']), worst3_draw_avg(df['net_strategy']), max_dd_duration(df['net_strategy']),
                              num_trades(df['position']), num_trades_to_periods(df['position'])],

                 'Benchmark': [profit_loss(df['bmk_return']), profit_loss(df['bmk_return']), pl_CAGR(df['bmk_return']), pl_CAGR(df['bmk_return']),
                               an_vol(df['bmk_return'], freq), positive_per(df['bmk_return']),
                               s.skew(df['bmk_return']), s.kurtosis(df['bmk_return']), s.kurtosistest(df['bmk_return'])[1],
                               an_down_vol(df['bmk_return'], freq), df['bmk_return'].min(),
                               sharpe(df['bmk_return'], freq, rf), sortino(df['bmk_return'], freq, rf),
                               None,
                               max_draw(df['bmk_return']), worst3_draw_avg(df['bmk_return']), max_dd_duration(df['bmk_return']),
                               0, 0]}

        df = pd.DataFrame(stats)
        df.set_index('Statistics', inplace=True)
        print('From %s to %s' % (self.start, self.end))
        return df.round(3)

    def plot_data(self):
        ''' Plots original data and trading indicators .
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | slip = %d, sd = %d' % ('Spread and Position', self.slip, self.sd)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.title(title)
        ax1.plot(self.results.index, self.results['price'])
        ax2.plot(self.results.index, self.results['position'], 'ro')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price', color='g')
        ax2.set_ylabel('Position', color='r')
        plt.show()

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')

        title = '%s | slip = %d, sd = %d' % ('Cumulative Performance', self.slip, self.sd)
        self.results[['creturns', 'cstrategy', 'cnetstrategy', 'cbmkreturns']].plot(title=title, figsize=(10, 6))
        self.results[['bmk_return', 'net_strategy']].hist(bins=50, figsize=(10, 6))

    def plot_drawdown(self):
        ''' Plots Drawdown metrics (DD and DD Periods) along with a rolling DD line chart and histogram
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')

        cumret = self.results['net_strategy'].cumsum().apply(np.exp).dropna()
        cummax = cumret.cummax()
        drawdown = 100 * abs(cumret / cummax - 1).dropna()
        temp = drawdown[drawdown == 0]
        periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())

        print(pd.DataFrame({'3 Worst DD': sorted(drawdown, reverse=True)[0:3],
                            '3 Worst DD Periods': sorted(periods, reverse=True)[0:3]}))
        print('#' * 50)
        print(pd.DataFrame({'Rolling DrawDown Stats': drawdown}).describe())
        print('#' * 50)
        title = 'Rolling Drawdown | slip = %d, sd = %d' % (self.slip, self.sd)
        drawdown.plot(figsize=(10, 6), title=title)
        pd.DataFrame({'Rolling Drawdown': drawdown}).hist(bins=50, figsize=(10, 6))

    def set_parameters(self, sd=None, slip=None, start=None, end=None, ratio=None):
        ''' Updates parameters and and datetime.
        '''
        if sd is not None:
            self.sd = sd
        if slip is not None:
            self.slip = slip
        if start is not None and end is not None:
            self.start = start
            self.end = end
            date_mask = (self.data.index > start) & (self.data.index < end)
            self.data = self.data[date_mask]
        if ratio is not None:
            self.ratio = ratio

    def update_and_run(self, par):
        ''' Updates parameters and returns negative absolute performance for minimazation algo.

        Parameters
        -----------
        par: tuple
            parameter tuple (sd, slip)

        '''

        ratio = self.ratio_opt

        #  maximizing + Return  = minimizing -Return
        if ratio == 1:
            return -self.run_strategy(int(par[0]), int(par[1]))['strat cum P&L']
        elif ratio == 2:
            return self.run_strategy(int(par[0]), int(par[1]))['strat vol']
        elif ratio == 3:
            return -self.run_strategy(int(par[0]), int(par[1]))['strat sharpe']
        else:
            return -self.run_strategy(int(par[0]), int(par[1]))['strat cum P&L']

    def optimize_parameters(self, sd_range, slip_range, ratio):
        ''' Finds global maximum given the slip and sd parameter ranges.

        Parameters
        -----------
        slip_range, sd_range: tuple
            tuples of the form (start, end, step size)
        ratio: int
            ratio to be optimized 1 = 'Max P&L', 2= 'Min Vol', 3 = 'Max Sharpe'

        Output:
        '[Optimal sd, Optinal slip], {Abs Net P&L, Abs Net P&L vs bmk, Annual. Vol, Sharpe Ratio'}
        '''
        self.ratio_opt = ratio
        opt = brute(self.update_and_run, (sd_range, slip_range,), finish=None)

        # # brute=> brute(func, ranges, args=(), Ns=20, full_output=0, finish=<function fmin at 0x00000000079A6DD8>, disp=False)
        # Minimize a function over a given range by brute force.

        # return opt and related results:
        # temp = MRVectorBacktester(self.y_series, self.x_series, self.bmk_series, self.start, self.end,
        #                           self.start_in, self.end_in, freq=self.freq, trans=None, ctf=self.ctf)

        opt_p = list(map(lambda x: int(x), opt))
        return opt_p, self.run_strategy(int(opt[0]), int(opt[1]))

    def optimal_param_multiper(self, sd_range, slip_range, ratio, n_split=3):
        ''' Splits sample into multiple periods and gets optimal params for each time period
        ratio: int
            ratio to be optimized 1 = 'Max P&L', 2= 'Min Vol', 3 = 'Max Sharpe' 
        Parameters
        ----------
        sd_range = (start,end, step). Range of values to optimize standard deviation multipler (sd)
        slip_range = (start,end, step).Range of values to optimize number of slippage in terms of days (slip)
        n_split = 3 default. Number of periods to split the time series data to optimize.

        '''
        series = self.data['price']
        k = int(np.floor(len(series) / n_split))  # size per fold

        type = []
        par1 = []
        par2 = []
        start = []
        end = []
        pl = []
        vol = []
        sharpe = []
        # backup info:
        backup = self.data
        start_bku = self.start
        end_bku = self.end
        slip_bku = self.slip
        sd_bku = self.sd

        for i in range(1, n_split + 1):
            x = series[k * (i - 1):k * i]
            n_start = str(x.index.to_pydatetime()[0].date())  # Timestampt to string date format
            n_end = str(x.index.to_pydatetime()[len(x) - 1].date())  # Timestampt to string date format
            self.set_parameters(start=n_start, end=n_end)  # sets new time ranges within the original instance dataframe self.data
            op = self.optimize_parameters(sd_range, slip_range, ratio)  # new params according to new time period
            run = self.run_strategy()
            type.append('Mean-Reversion')
            par1.append(int(op[0][0]))
            par2.append(int(op[0][1]))
            start.append(n_start)
            end.append(n_end)
            pl.append(run['strat cum P&L']), vol.append(run['strat vol']), sharpe.append(run['strat sharpe'])
            self.data = backup

        self.start = start_bku
        self.end = end_bku
        self.slip = slip_bku
        self.sd = sd_bku

        df = pd.DataFrame({'type': type, 'par1': par1, 'par2': par2, 'start': start, 'end': end, 'pl': pl, 'vol': vol, 'sharpe': sharpe})
        print(df)
        results = pd.DataFrame({})
        for i in range(len(df)):
            type = df.iloc[i, :]['type']
            par1 = df.iloc[i, :]['par1']
            par2 = df.iloc[i, :]['par2']
            start = df.iloc[i, :]['start']
            end = df.iloc[i, :]['end']
            pl = df.iloc[i, :]['pl']
            vol = df.iloc[i, :]['vol']
            sharpe = df.iloc[i, :]['sharpe']
            temp = self.run_strategy(sd=par1, slip=par2)
            results.loc[i, 'type'] = type
            results.loc[i, 'par1'] = par1
            results.loc[i, 'par2'] = par2
            results.loc[i, 'start'] = start
            results.loc[i, 'end'] = end
            results.loc[i, 'pl'] = temp['strat cum P&L']
            results.loc[i, 'vol'] = temp['strat vol']
            results.loc[i, 'sharpe'] = temp['strat sharpe']

        rat = 'pl' if ratio == 1 else 'vol' if ratio == 2 else 'sharpe'
        print('Peformance Metrics for the period %s - %s' % (self.start, self.end))
        return results.sort_index(by=rat, ascending=False)
