from ib_insync import ib, IB, Stock, util, Option, Index
import asyncio
import pandas as pd
import yoptions as yo
import inspect
import plotly.express as px
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime, timedelta
from math import exp, log, sqrt
from scipy.stats import norm
import torch
import plotly.graph_objects as go



class IbConnect:
    """ Class to connect to the database of Interactive Brokers """

    def __init__(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.ib = IB()
        self.connection = self.ib.connect('127.0.0.1', 7497, clientId=9)

    def __delete__(self):
        if hasattr(self, "connection"):
            if self.connection.isConnected():
                self.connection.disconnect()

    def read_historical_data(self, ticker, currency):
        self.check_connection()
        hist_data = get_hist_data(self.ib, ticker, currency)
        self.connection.disconnect()
        return hist_data

    def read_exchanges(self, ticker, currency, pexchange):
        """ Read available exchanges

        :param ticker:
        :param currency:
        :param pexchange:
        :return:
        """
        self.check_connection()
        chains, df_chains = get_exchanges(self.ib, ticker, currency, pexchange)
        self.connection.disconnect()
        return chains, df_chains

    def read_expirations(self, chains):

        self.check_connection()
        expirations = get_expiration(self.ib, chains)
        self.connection.disconnect()
        return expirations

    def read_option_data(self, chains, ticker, right, exchange, currency, pexchange):
        """ Read options chain

        :param chains:
        :param ticker:
        :param right:
        :param exchange:
        :param currency:
        :param pexchange:
        :return:
        """
        self.check_connection()

        stock = Stock(ticker, "SMART", currency, primaryExchange=pexchange)
        self.ib.qualifyContracts(stock)
        self.ib.reqMarketDataType(3)
        [stock_px] = self.ib.reqTickers(stock)
        spxValue = stock_px.marketPrice()

        chains = chains[(chains["exchange"] == exchange)]  # (chains["tradingClass"] == ticker) &
        if chains.shape[0] != 1:
            chains = chains.iloc[0, :]
        tradingClass = chains["tradingClass"].values[0]

        # chains = next(c for c in chains if c.tradingClass == ticker and c.exchange == exchange)

        # chains2 = chains[(chains["strike"] % 5 == 0) & (chains["strike"] < spxValue - 20) & (chains["strike"] > spxValue + 20)]
        # chains3 = chains2.sort(by="expirations")
        if spxValue < 100:
            multiplier = 20
        elif spxValue < 500:
            multiplier = 50
        elif spxValue < 1000:
            multiplier = 100
        elif spxValue < 5000:
            multiplier = 300
        else:
            multiplier = 500
        strikes = [strike for strike in chains["strikes"].values[0]
                   if strike % 5 == 0
                   and spxValue - multiplier < strike < spxValue + multiplier]
        expirations = sorted(exp for exp in chains["expirations"].values[0][:5])

        contracts = [Option(ticker, expiration, strike, right, 'SMART', tradingClass=tradingClass)
                     for expiration in expirations
                     for strike in strikes]

        # contracts = []
        # for row, item in chains.iterrows():
        #    expiration = item['expiration']
        #    strike = item['strike']
        #    contracts.append(Option(ticker, expiration, strike, right, exchange, tradingClass='AMD'))

        contracts = self.ib.qualifyContracts(*contracts)
        list_tickers = self.ib.reqTickers(*contracts)

        def as_dict(ticker):
            return {'strike': ticker.contract.strike, 'symbol': ticker.contract.symbol,
                    "right": ticker.contract.right, "multiplier": ticker.contract.multiplier,
                    'expiration': ticker.contract.lastTradeDateOrContractMonth,
                    'undPrice': ticker.modelGreeks.undPrice,
                    # 'bidDelta': ticker.bidGreeks.delta, 'bidGamma': ticker.bidGreeks.gamma,
                    # 'askDelta': ticker.askGreeks.delta,
                    # 'askGamma': ticker.askGreeks.gamma,
                    'modelDelta': ticker.modelGreeks.delta,
                    'modelGamma': ticker.modelGreeks.gamma, 'modelIV': ticker.modelGreeks.impliedVol,
                    'modelPrice': ticker.modelGreeks.optPrice, "modelTheta": ticker.modelGreeks.theta,
                    "modelVega": ticker.modelGreeks.vega}

        # check = list_tickers[0]
        # if inspect.isclass(check):
        df_opotion_chain = pd.DataFrame([as_dict(x) for x in list_tickers])  # if as_dict(x) != None
        df_opotion_chain['maturity'] = pd.to_datetime(df_opotion_chain["expiration"]) - pd.Timestamp.now().normalize()
        # else:
        #    df_opotion_chain = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type=right, risk_free_rate=None)
        self.connection.disconnect()
        return df_opotion_chain

    def check_connection(self):
        """ Check is connection with database is open. If not, connect """
        if not self.connection.isConnected():
            self.connection = self.ib.connect('127.0.0.1', 7497, clientId=9)


def get_expiration(ib, chains, ticker):
    """ Get expiration dates for options """
    chain = next(c for c in chains if c.tradingClass == ticker and c.exchange == 'SMART')  # todo replate ticker

    expirations = sorted(exp for exp in chain.expirations)
    return expirations


def get_hist_data(ib, ticker, currency):
    """ Get historical data """
    contract = Stock(ticker, 'SMART', currency)
    ib.reqMarketDataType(4)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='5 Y',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1)
    df_bars = util.df(bars)

    return df_bars


def get_exchanges(ib, ticker, currency, pexchange):
    """ Function to retrive options chains data"""

    stock = Stock(ticker, 'SMART', currency, primaryExchange=pexchange)
    ib.reqMarketDataType(4)
    ib.qualifyContracts(stock)
    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
    df_chains = util.df(chains)
    print(df_chains)
    return chains, df_chains


def check_optype(op_type):
    if (op_type not in ['p', 'c']):
        raise ValueError("Input 'p' for put and 'c' for call!")


def check_trtype(tr_type):
    if (tr_type not in ['b', 's']):
        raise ValueError("Input 'b' for Buy and 's' for Sell!")


def payoff_calculator(x, op_type, strike, op_pr, tr_type, n):
    y = []
    if op_type == 'c':
        for i in range(len(x)):
            y.append(max((x[i] - strike - op_pr), -op_pr))
    else:
        for i in range(len(x)):
            y.append(max(strike - x[i] - op_pr, -op_pr))
    y = np.array(y)

    if tr_type == 's':
        y = -y
    return y * n


def multi_plotter(spot_range=20, spot=100, op_list=[],
                  save=False, file='fig.png'):
    x = spot * np.arange(100 - spot_range, 101 + spot_range, 0.01) / 100
    y0 = np.zeros_like(x)

    y_list = []
    for op in op_list:
        op_type = str.lower(op['op_type'])
        tr_type = str.lower(op['tr_type'])
        check_optype(op_type)
        check_trtype(tr_type)

        strike = op['strike']
        op_pr = op['op_pr']
        try:
            contract = op['contract']
        except:
            contract = 1
        y_list.append(payoff_calculator(x, op_type, strike, op_pr, tr_type, contract))
    df_pay = pd.DataFrame(y_list).T.sum(axis=1)
    df_pay.index = x
    df_pay = df_pay.reset_index()
    df_pay.columns = ["strike", "payoff"]
    return df_pay


import yfinance as yf
import numpy as np


class OptionPortflio:

    def __init__(self, df_options):
        self.df_options = df_options
        self.df_proc = self.preprocessing(df_options)
        self.stock_list = self.df_proc['symbol'].values
        #self.df_stock_px, self.d_last_quote = self.get_hist_stock_px(self.stock_list)
        self.top_table, self.net_value_greeks,self.surface_fig = self.get_value_greeks(self.df_proc)

    def preprocessing(self, z_data):
        z_data1 = z_data.copy()
        z_data1['maturity'] = pd.to_datetime(z_data1["expiration"], format='%Y%m%d') - pd.Timestamp.now().normalize()
        z_data1["maturity"] = z_data1["maturity"].astype(str).str.replace("days", "").astype(float)
        z_data1["moneyness"] = z_data1["strike"] / z_data1["undPrice"]
        return z_data1

    def get_value_greeks(self, z_data):
        condition = ((z_data["right"] == "C") & (z_data["Trade"] == "Buy")) | (
                (z_data["right"] == "P") & (z_data["Trade"] == "Sell"))

        z_data['indicator'] = np.where(condition, 1, -1)
        z_data["posDelta"] = z_data['indicator'] * z_data['modelDelta'] * z_data["Quantity"]
        z_data["posGamma"] = z_data['indicator'] * z_data['modelGamma'] * z_data["Quantity"]

        z_data["volBeta"] = [(1 + np.random.random(1))[0] for i in range(0, z_data.shape[0])]#todo sistemare
        z_data["posVega"] = z_data['indicator'] * z_data['modelVega'] * z_data["Quantity"]
        z_data = z_data.set_index("symbol")

        self.dict_betas, self.df_betas = self.get_betas(self.stock_list)
        merge = z_data.merge(self.df_betas, left_index=True, right_index=True, how="left")
        merge["%Δ index"] = 0.01
        merge["%Δ stock"] = merge["%Δ index"] * merge["beta"]
        merge.index.name = "symbol"
        order_col = ["expiration", "Trade", "right", "Quantity", "multiplier", "undPrice",  "%Δ index", "beta", "%Δ stock", "posDelta", "posGamma", "volBeta", "posVega"]
        out_table = merge[order_col].reset_index()

        net_value_greeks = out_table.groupby("symbol").agg({'undPrice': 'last', 'multiplier': 'last', 'beta':'last', 'posDelta': 'sum',
                                                         'posGamma': 'sum',  'volBeta':'last', 'posVega': 'sum'})
        net_value_greeks["valueDelta"] = net_value_greeks["posDelta"] * net_value_greeks["undPrice"] * net_value_greeks["multiplier"]
        net_value_greeks["valueGamma"] = net_value_greeks["posGamma"] * net_value_greeks["undPrice"] * net_value_greeks["multiplier"]
        net_value_greeks["valueVega"] = net_value_greeks["posVega"] * net_value_greeks["multiplier"]

        #merge = net_value_greeks.merge(self.df_betas, left_index=True, right_index=True, how="left")
        p_valuedelta = sum(net_value_greeks["valueDelta"] * net_value_greeks["beta"])
        p_valuegamma = sum(np.power(net_value_greeks["beta"], 2) * net_value_greeks["valueDelta"])
        p_valuebeta = sum(net_value_greeks["valueVega"] * net_value_greeks["volBeta"])

        variation = 0.01
        TotPL = p_valuedelta * variation + 0.5 * variation ** 2 * p_valuegamma + p_valuebeta * variation
        x = np.arange(-0.2, 0.2, 0.01)
        T = [p_valuedelta * v + 0.5 * v ** 2 * p_valuegamma for v in x]
        Tv = [p_valuebeta * v for v in x]

        dg = torch.Tensor(T)
        v = torch.Tensor(Tv)
        dgv_3d = dg.reshape(-1, 1) + v
        a = pd.DataFrame(dgv_3d, index=x, columns=x)
        print(a)
        fig = go.Figure(data=[go.Surface(z=a.values, x=a.index, y=a.columns)])


        return out_table, net_value_greeks, fig

    def get_betas(self, stock_list):
        dict_betas = {}
        for i in stock_list:
            yftick = yf.Ticker(i)
            beta = yftick.info['beta']
            dict_betas[i] = beta
        df_betas = pd.DataFrame(dict_betas, index=[0])
        betas_T = df_betas.T
        betas_T.columns = ["beta"]
        return dict_betas, betas_T

    def get_hist_stock_px(self, list):
        d_last_quote = dict()
        ts = []
        for ticker in list:
            ticker_yahoo = yf.Ticker(ticker)
            data = ticker_yahoo.history(period='5y')["Close"]
            data.columns = ticker
            d_last_quote[ticker] = data.iloc[-1]
            ts.append(data)
            index = data.index
        df_hist_px = pd.DataFrame(ts).T
        df_hist_px.index = index
        return df_hist_px, d_last_quote

    def get_payoff(self, stock, expiration, plot=False):
        df = self.df_proc
        condition = (df["symbol"] == stock) & (df["expiration"] == int(expiration))
        df_filt = df[condition]
        op_list = []
        for row, item in df_filt.iterrows():
            strike = item["strike"]
            right = item["right"].lower()
            trade = "s" if item["Trade"] == "Sell" else "b"
            if trade == "Buy":
                op_price = item["ask"]  # todo change item["modelBid"]
            else:
                op_price = item["bid"]
            undPrice = item["undPrice"]
            op = {'op_type': right, 'strike': strike, 'tr_type': trade, 'op_pr': op_price}
            op_list.append(op)
        spot_range = abs(strike - undPrice + 20)
        df_pay = multi_plotter(spot=undPrice, spot_range=spot_range, op_list=op_list)
        fig = px.line(df_pay, x="strike", y="payoff", title='Payoff')
        return fig

    def jump_diffusion_process(self, index, r, sigma, plot = False):

        def gen_paths(S0, r, sigma, T, n_scenarios, n_timesteps, last_date):
            dt = float(T) / n_scenarios
            paths = np.zeros((n_scenarios + 1, n_timesteps), np.float64)
            list_dates = []
            list_dates.append(pd.to_datetime(last_date))
            paths[0] = S0

            end_date = last_date
            for t in range(1, n_scenarios + 1):
                end_date = pd.to_datetime(end_date) + timedelta(days=1)
                list_dates.append(end_date)
                rand = np.random.standard_normal(n_timesteps)
                paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                                 sigma * np.sqrt(dt) * rand)
            df_paths = pd.DataFrame(paths)
            df_paths["Date"] = list_dates

            return paths, list_dates, df_paths

        index = yf.Ticker(index)  # "^GSPC"
        index_hist = index.history(period="5Y").reset_index()
        last_px = index_hist["Close"].values[-1]
        last_date = index_hist["Date"].values[-1]
        a, list_dates, df_paths = gen_paths(last_px, r, sigma, 2, 100, 50, last_date)

        df_paths = df_paths.set_index(["Date"])

        all_data = []
        return_list = []
        for col in df_paths.columns:
            df_1 = index_hist.set_index("Date")["Close"][:-1]
            series = pd.concat([df_1, df_paths[col]], axis=0)
            return_path = (df_paths[col][-1] - df_1[-1]) / df_paths[col][-1]
            all_data.append(series)
            return_list.append(return_path)

        df_final = pd.concat(all_data, axis=1).reset_index("Date")
        if plot:
            fig = px.line(df_final, x="Date", y=df_final.columns,
                          hover_data={"Date": "|%B %d, %Y"},
                          title='custom tick labels with ticklabelmode="period"')
            fig.update_xaxes(
                dtick="M1",
                tickformat="%b\n%Y",
                ticklabelmode="period")
            fig.show()
        return df_final, r, sigma

    def simulate_scenario(self, index_change, r, time_in_months):
        time_days = time_in_months * 22
        df_scenario = self.df_proc.copy()
        new_exp = "Exp in {} months".format(time_in_months)
        df_scenario[new_exp] = np.where(df_scenario["maturity"] > time_days, df_scenario["maturity"] - time_days,
                                        "Expired")

        new_und_price = dict()
        for stock in self.stock_list:
            new_und_price[stock] = (1 + index_change) * self.betas[stock] * self.d_last_quote[stock]

        # attach to new dataframe
        # get vol betas
        # calculate new vols
        # apply bsm


        #https://www.quantstart.com/articles/European-Vanilla-Call-Put-Option-Pricing-with-Python/

        def d_j(j, S, K, r, v, T):
            """
            d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)}
            """
            return (log(S / K) + (r + ((-1) ** (j - 1)) * 0.5 * v * v) * T) / (v * (T ** 0.5))

        def vanilla_call_price(S, K, r, v, T):
            """
            Price of a European call option struck at K, with
            spot S, constant rate r, constant vol v (over the
            life of the option) and time to maturity T
            """
            return S * norm.cdf(d_j(1, S, K, r, v, T)) - \
                   K * exp(-r * T) * norm.cdf(d_j(2, S, K, r, v, T))

        def vanilla_put_price(S, K, r, v, T):
            """
            Price of a European put option struck at K, with
            spot S, constant rate r, constant vol v (over the
            life of the option) and time to maturity T
            """
            return -S * norm.cdf(-d_j(1, S, K, r, v, T)) + \
                   K * exp(-r * T) * norm.cdf(-d_j(2, S, K, r, v, T))




"""
import pickle
with open('company_data.pkl', 'wb') as outp:
    pickle.dump(ticker, outp, pickle.HIGHEST_PROTOCOL)
    
with open('company_data.pkl', 'rb') as inp:
    ticker = pickle.load(inp)



    strikes = [strike for strike in chain.strikes
           if strike % 5 == 0
           and spxValue - 20 < strike < spxValue + 20]
    rights = ['P', 'C']  #todo type of option as input

    contracts = [Option('AMD', expiration, strike, right, 'SMART', tradingClass='AMD')
                 for right in rights
                 for expiration in expirations
                 for strike in strikes]

    contracts = ib.qualifyContracts(*contracts)
    tickers = ib.reqTickers(*contracts)
    return tickers



@app.callback(
    Output("option-chain-table", "data"),
    Input("select-ticker", "value"),
    prevent_initial_call = True
)
def get_option_data(ticker):
    if ticker == '':
        return dash.no_update
    else:
        ib = IB()
        ib.qualifyContracts(ticker)
        ib.reqMarketDataType(3)
        chains = ib.reqSecDefOptParams(ticker.symbol, '', ticker.secType, ticker.conId)
        option_chain = util.df(chains)
        return option_chain



@app.callback(
    Output("economic-calendar-table", "data"),
    Input("select-date-calendar", "date"),
    prevent_initial_call = True
)
def select_date_calendar(value):
    import investpy
    date_0 = datetime.strptime(value, "%Y-%m-%d")
    date_1 = date_0 + timedelta(days=1)
    date_0 = date_0.strftime("%d/%m/%Y")
    date_1 = date_1.strftime("%d/%m/%Y")
    df = investpy.economic_calendar(
                        from_date= date_0,
                        to_date=date_1)
    return df.to_dict("records")


"""
