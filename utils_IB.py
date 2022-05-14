from ib_insync import ib, IB, Stock, util, Option, Index
import asyncio
import  pandas as pd
import yoptions as yo
import inspect

class IbConnect:

    def __init__(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.ib = IB()
        self.connection = self.ib.connect('127.0.0.1', 7497, clientId=1)

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
        self.check_connection()

        stock = Stock(ticker, "SMART", currency, primaryExchange=pexchange)
        self.ib.qualifyContracts(stock)
        self.ib.reqMarketDataType(3)
        [stock_px] = self.ib.reqTickers(stock)
        spxValue = stock_px.marketPrice()


        chains = chains[(chains["exchange"] == exchange)] #(chains["tradingClass"] == ticker) &
        if chains.shape[0] != 1:
            chains = chains.iloc[0, :]
        tradingClass = chains["tradingClass"].values[0]

        #chains = next(c for c in chains if c.tradingClass == ticker and c.exchange == exchange)

        #chains2 = chains[(chains["strike"] % 5 == 0) & (chains["strike"] < spxValue - 20) & (chains["strike"] > spxValue + 20)]
        #chains3 = chains2.sort(by="expirations")
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

        #contracts = []
        #for row, item in chains.iterrows():
        #    expiration = item['expiration']
        #    strike = item['strike']
        #    contracts.append(Option(ticker, expiration, strike, right, exchange, tradingClass='AMD'))

        contracts = self.ib.qualifyContracts(*contracts)
        list_tickers = self.ib.reqTickers(*contracts)

        def as_dict(ticker):
            return {'strike': ticker.contract.strike, 'expiration': ticker.contract.lastTradeDateOrContractMonth,
                    'bidDelta': ticker.bidGreeks.delta, 'bidGamma': ticker.bidGreeks.gamma,
                    'askDelta': ticker.askGreeks.delta,
                    'askGamma': ticker.askGreeks.gamma, 'modelDelta': ticker.modelGreeks.delta,
                    'modelGamma': ticker.modelGreeks.gamma, 'modelIV': ticker.modelGreeks.impliedVol,
                    'modelPrice': ticker.modelGreeks.optPrice, "modelTheta": ticker.modelGreeks.theta,
                    "modelVega": ticker.modelGreeks.vega}

        check = list_tickers[0]
        if inspect.isclass(check):
            df_opotion_chain = pd.DataFrame([as_dict(x) for x in list_tickers if as_dict(x) != None])
            df_opotion_chain['maturity'] = df_opotion_chain["expiration"] - pd.Timestamp.now().normalize()
        else:
            df_opotion_chain = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type=right, risk_free_rate=None)
        self.connection.disconnect()
        return df_opotion_chain

    def check_connection(self):

        if not self.connection.isConnected():
            self.connection = self.ib.connect('127.0.0.1', 7497, clientId=1)


def get_expiration(ib, chains, ticker):

    chain = next(c for c in chains if c.tradingClass == ticker and c.exchange == 'SMART') #todo replate ticker

    expirations = sorted(exp for exp in chain.expirations)
    return expirations
    



def get_hist_data(ib, ticker, currency):
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
