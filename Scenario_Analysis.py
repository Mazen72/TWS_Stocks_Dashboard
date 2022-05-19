import dash
import pandas as pd
import base64
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
import io
from dash import Dash, Input, Output, dash_table, callback_context, State
import dash_bootstrap_components as dbc
from dash import dcc, html
from datetime import date, datetime, timedelta
import yfinance as yf
import numpy as np
from math import exp, log, sqrt
from scipy.stats import norm
import torch
import asyncio
from utils_IB import OptionPortflio

components_colors={ 'Main Header Background': ['#0b1a50', '#0b1a50'], 'Main Background': ['#e7f0f9', '#e7f0f9'],
                    'Main Header Text': ['white', 'white']}
text_font_size = '1.5vh'

dummy_df=pd.read_csv('temp_files/dummy.csv')

def get_scenario_layout(tabs):

    drift_input_text = html.Div(html.H1('Drift',
                                style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                           marginTop='')),
                        style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    drift_input =dbc.Input(
        placeholder='Enter Value',n_submit=0,type='number',
        id='drift_input', autocomplete='off',style=dict(border='1px solid #0b1a50',width='8vw',textAlign="center"),

    )



    input1_div = html.Div([drift_input_text, drift_input],
                            style=dict(fontSize=text_font_size,
                                       marginLeft='', marginBottom='', display='inline-block'))


    volatility_input_text = html.Div(html.H1('Volatility',
                                        style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                                   marginTop='')),
                                style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    volatility_input = dbc.Input(
        placeholder='Enter Value', n_submit=0, type='number',
        id='volatility_input', autocomplete='off', style=dict(border='1px solid #0b1a50', width='8vw',textAlign="center"),

    )

    input2_div = html.Div([volatility_input_text, volatility_input],
                          style=dict(fontSize=text_font_size,
                                     marginLeft='2vw', marginBottom='', display='inline-block'))

    inputs_row1=html.Div([input1_div, input2_div],style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                           'justify-content': 'center'})

    prob_input_text = html.Div(html.H1('Prob',
                                        style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                                   marginTop='')),
                                style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    prob_input = dbc.Input(
        placeholder='Enter Value', n_submit=0, type='number',
        id='prob_input', autocomplete='off', style=dict(border='1px solid #0b1a50', width='8vw',textAlign="center"),

    )

    input3_div = html.Div([prob_input_text, prob_input],
                          style=dict(fontSize=text_font_size,
                                     marginLeft='', marginBottom='', display='inline-block'))

    intensity_input_text = html.Div(html.H1('Intensity',
                                             style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                                        marginTop='')),
                                     style=dict(display='inline-block', marginLeft='', textAlign="center",
                                                width='100%'))

    intensity_input = dbc.Input(
        placeholder='Enter Value', n_submit=0, type='number',
        id='intensity_input', autocomplete='off', style=dict(border='1px solid #0b1a50', width='8vw',textAlign="center"),

    )

    input4_div = html.Div([intensity_input_text, intensity_input],
                          style=dict(fontSize=text_font_size,
                                     marginLeft='2vw', marginBottom='', display='inline-block'))

    inputs_row2 = html.Div([input3_div, input4_div], style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                            'justify-content': 'center'})

    risk_input_text = html.Div(html.H1('Risk Free',
                                        style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                                   marginTop='')),
                                style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    risk_input = dbc.Input(
        placeholder='Enter Value', n_submit=0, type='number',
        id='risk_input', autocomplete='off', style=dict(border='1px solid #0b1a50', width='8vw',textAlign="center"),

    )

    input5_div = html.Div([risk_input_text, risk_input],
                          style=dict(fontSize=text_font_size,
                                     marginLeft='', marginBottom='', display='inline-block'))

    horizon_input_text = html.Div(html.H1('Horizon( Months )',
                                             style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                                        marginTop='')),
                                     style=dict(display='inline-block', marginLeft='', textAlign="center",
                                                width='100%'))

    horizon_input = dbc.Input(
        placeholder='Enter Value', n_submit=0, type='number',
        id='horizon_input', autocomplete='off', style=dict(border='1px solid #0b1a50', width='8vw',textAlign="center"),

    )

    input6_div = html.Div([horizon_input_text, horizon_input],
                          style=dict(fontSize=text_font_size,
                                     marginLeft='2vw', marginBottom='', display='inline-block'))

    inputs_row3 = html.Div([input5_div, input6_div], style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                            'justify-content': 'center'})

    index_menu = dcc.Dropdown(
                         options=[{'label': "S&P500", 'value': "S&P500"},  {'label': "EURO STOCXX500", 'value': "EURO STOCXX500"},  {'label': "MSCI World", 'value': "MSCI World"}],
                         value='S&P500',
                         id='index_menu',
                            style=dict(color='white', fontWeight='bold', textAlign='center',
                                       width='9vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
                            )

    index_menu_text = html.Div(html.H1('Index',
                                style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                           marginTop='')),
                        style=dict(display='', marginLeft='', textAlign="center", width='100%'))

    index_menu_div = html.Div([index_menu_text, index_menu])

    index_menu_row=html.Div([index_menu_div],style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                           'justify-content': 'center'})

    simulate_button = html.Div([dbc.Button('Simulate Scenario', color="primary", size='lg', n_clicks=0, id='simulate_button'
                                        , style=dict(fontSize=text_font_size,backgroundColor='#119DFF',width='10vw')
                                        )], style=dict(width='100%',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))



    options_menu = dcc.Dropdown(
                         options=[{'label': "Yield Curve", 'value': "Yield Curve"},
                                  {'label': "Index Simulations", 'value': "Index Simulations"}],
                         value=None,
                         id='options_menu',
                            style=dict(color='white', fontWeight='bold', textAlign='center',
                                       width='9vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
                            )


    options_menu_div=html.Div([options_menu],style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                           'justify-content': 'center'})

    fig1=px.line()

    fig1.update_layout(
        #title_text='<b>Payoff<b>',title_x=0.5, xaxis_title='<b>Strike<b>',yaxis_title='<b>Payoff<b>',
        font=dict(size=14, family='Arial', color='#0b1a50'), hoverlabel=dict(
            font_size=14, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        xaxis=dict(

            tickwidth=2, tickcolor='#80ced6',
            ticks="outside",
            tickson="labels",
            rangeslider_visible=False
        ) ,margin=dict(l=0, r=0, t=40, b=0)
    )

    fig1.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    fig1.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')



    chart1_div=html.Div([
            dcc.Graph(id='chart1', config={'displayModeBar': True, 'scrollZoom': True,'displaylogo': False},
                style=dict(height='32vh',backgroundColor='#F5F5F5') ,figure=fig1
            ) ] ,id='mychart1_div'
        )

    middle_table_header = html.Div(html.H1('Portfolio in 6 Months',
                                     style=dict(fontSize='2vh', fontWeight='bold', color='#0b1a50',
                                                marginTop='')),
                             style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    bottom_table_header = html.Div(html.H1('Scenario Portfolio',
                                     style=dict(fontSize='2vh', fontWeight='bold', color='#0b1a50',
                                                marginTop='')),
                             style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    middle_table=html.Div(
        id = 'middle_table_div')

    bottom_table=html.Div(

        id='bottom_table_div')

    fig2=go.Figure()
    fig2.update_layout(
        #title_text='<b>Payoff<b>',title_x=0.5, xaxis_title='<b>Strike<b>',yaxis_title='<b>Payoff<b>',
        font=dict(size=14, family='Arial', color='#0b1a50'), hoverlabel=dict(
            font_size=14, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        xaxis=dict(

            tickwidth=2, tickcolor='#80ced6',
            ticks="outside",
            tickson="labels",
            rangeslider_visible=False
        ) ,margin=dict(l=0, r=0, t=40, b=0)
    )

    fig2.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    fig2.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')

    chart2_div=html.Div([
            dcc.Graph(id='chart2', config={'displayModeBar': True, 'scrollZoom': True,'displaylogo': False},
                style=dict(height='28vh',backgroundColor='#F5F5F5') ,figure=fig2
            ) ] ,id='mychart2_div'
        )


    layout=[
        dbc.Col([dbc.Card(dbc.CardBody([
            tabs

        ])
            , style=dict(backgroundColor='transparent',
                         border='1px solid {}'.format(components_colors['Main Background'][0])), id='card1',
            className='tabs-card'), html.Br()
        ], xl=dict(size=2, offset=0), lg=dict(size=2, offset=0),
            md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
            style=dict(paddingLeft='', paddingRight='', border='')),


        dbc.Col([dbc.Card(dbc.CardBody([index_menu_row, html.Br(), inputs_row1, html.Br(), inputs_row2,
                                        html.Br(),inputs_row3,html.Br(),simulate_button,

                                        html.Div(id='inputs_exception')
                                        ])
                          , style=dict(backgroundColor='#F5F5F5',
                                       border=''), id='card2',
                          className='menus-card'), html.Br()
                 ], xl=dict(size=3, offset=0), lg=dict(size=3, offset=0),
                md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                style=dict(paddingLeft='', paddingRight='', border='')),

        dbc.Col([dbc.Card(dbc.CardBody([options_menu_div,html.Br(),
            dbc.Spinner([chart1_div], size="lg", color="primary", type="border", fullscreen=False)

                     ])
            , style=dict(backgroundColor='#F5F5F5')), html.Br()
        ], xl=dict(size=6, offset=0), lg=dict(size=6, offset=0),
            md=dict(size=10, offset=1), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1)),

        dbc.Col([dbc.Card(dbc.CardBody([middle_table_header, dbc.Spinner([middle_table],
                                                                   size="lg", color="primary", type="border",
                                                                   fullscreen=False)

                                        ])
                          , style=dict(backgroundColor='#F5F5F5',
                                       border='', style=dict(height='')), id='card1',
                          className='card1'), html.Br()
                 ], xl=dict(size=10, offset=2), lg=dict(size=10, offset=2),
                md=dict(size=8, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                style=dict(paddingLeft='', paddingTop='1vh', border='',paddingRight='2vw')),

        dbc.Col([dbc.Card(dbc.CardBody([bottom_table_header,
                                        dbc.Spinner([bottom_table], size="lg", color="primary",
                                                    type="border", fullscreen=False)

                                        ])
                          , style=dict(backgroundColor='#F5F5F5', style=dict(height=''))), html.Br()
                 ], xl=dict(size=5, offset=2), lg=dict(size=5, offset=2),
                md=dict(size=4, offset=0), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1)),

        dbc.Col([dbc.Card(dbc.CardBody(
            [dbc.Spinner([chart2_div], size="lg", color="primary", type="border", fullscreen=False)
                ,
             ])
            , style=dict(backgroundColor='#F5F5F5', style=dict(height=''))), html.Br()
        ], xl=dict(size=5, offset=0), lg=dict(size=5, offset=0),
            md=dict(size=4, offset=0), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1),
            style=dict(paddingRight='2vw'))




    ]

    return layout

def get_yield_curve():
    rates =pd.read_csv(r"temp_files/yield_curve.csv")
    rates.columns = ["DATE", "1M", "6M", "1Y", "3Y", "5Y", "7Y", "10Y", "15Y", "30Y"]
    rates2 = rates[rates['1Y'].notna()]
    rates2 = rates2.fillna(method='ffill')
    rates2 = rates2.set_index("DATE")
    df = rates2.unstack()
    df = df.reset_index()

    #df.columns=['level_0','level_1','DATE']
    df["DATE"] = df["DATE"].astype(str)

    fig1 = px.line(df, x="level_0", y=0, animation_frame="DATE",
    animation_group="level_0", range_y=[0, 5], markers=True)
    fig1.update_layout(
        xaxis_title='<b>Maturity<b>',
        yaxis_title='<b>Rates<b>',
    )
    return fig1

'''
r = Risk free

time_in_months=horizon

lam = prob

intensity=jump

sigma=volat

'''
def jump_diffusion_process(index, drift, sigma, r,  time_in_months, lam, jump, df_proc,df_betas ,plot = False):

    def gen_paths(S0, drift, sigma, T, n_scenarios, n_timesteps, last_date, lam, jump):
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
            paths[t] = paths[t - 1] * np.exp((drift - 0.5 * sigma ** 2) * dt +
                                                 sigma * np.sqrt(dt) * rand) + np.random.poisson(lam) * jump
        df_paths = pd.DataFrame(paths)
        df_paths["Date"] = list_dates

        return paths, list_dates, df_paths

    index = yf.Ticker(index)  # "^GSPC"
    index_hist = index.history(period="5Y").reset_index()
    last_px = index_hist["Close"].values[-1]
    last_date = index_hist["Date"].values[-1]
    a, list_dates, df_paths = gen_paths(last_px, drift, sigma, 2, 100, 50, last_date, lam, jump)

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

        #calculate returns at t+1
    returns = df_final.set_index("Date").iloc[-1, :] / last_px
    portfoliovalue = []
    for i in returns:
        df, new_opvalue = simulate_baseline_scenario(i, sigma, r, time_in_months,df_proc,df_betas)
        lista = df[new_opvalue].values
        portfoliovalue.append(sum([i for i in lista if not isinstance(i, str)]))

    scenarios = pd.DataFrame(zip(returns.values, portfoliovalue), index=returns.index, columns=["index_ret", "portfoliovalue"])
    scenarios.index.name = "scenario"
    scenarios = scenarios.reset_index()

    min_ret = scenarios["portfoliovalue"].min()

    max_ret = scenarios["portfoliovalue"].max()

    fig_hist = px.histogram(

        scenarios, x="portfoliovalue"
     #  , range_x=[min_ret - 50, max_ret - 50]

        ,

        hover_data=scenarios.columns)

    fig = px.line(df_final, x="Date", y=df_final.columns,
                  hover_data={"Date": "|%B %d, %Y"},
                  title='custom tick labels with ticklabelmode="period"')
    fig.update_xaxes(
     #   dtick="M1",
     #   tickformat="%b\n%Y",
        ticklabelmode="period")
    if plot:

        fig.show()
        fig_hist.show()

    return df_final,  scenarios,fig_hist,fig


def simulate_baseline_scenario(drift, vol, r, time_in_months,df_proc,df_betas):
    # baseline
    time_days = time_in_months * 22
    df_scenario = df_proc.copy()
    new_exp = "Exp in {} months".format(time_in_months)
    df_scenario[new_exp] = np.where(df_scenario["maturity"] > time_days, df_scenario["maturity"] - time_days,
                                    "Expired")

    df_scenario["Time to maturity"] = df_scenario["maturity"] / 252

    df_scenario[new_exp] = df_scenario[new_exp].apply(lambda x: int(float(x)) if x != "Expired" else "Expired")
    new_S0 = "S0 in {} months".format(time_in_months)

    df_scenario=df_scenario.reset_index()
    df_betas=df_betas.reset_index()

    df_scenario = df_scenario.merge(df_betas, left_index=True, right_index=True, how="left")
    df_scenario = df_scenario.set_index("symbol")
    df_scenario[new_S0] = df_scenario["undPrice"] * df_scenario["beta"] * (1 + drift)

    new_vol = "Vol in {} months".format(time_in_months)
    df_scenario[new_vol] = df_scenario["volBeta"] * vol

    # https://www.quantstart.com/articles/European-Vanilla-Call-Put-Option-Pricing-with-Python/

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

    new_opvalue = "Option Value in {} months".format(time_in_months)

    df_scenario2 = df_scenario.copy()
    df_scenario2[new_opvalue] = "Expired"
    condition = (df_scenario2[new_exp] != "Expired")
    df_scenario2.loc[condition, new_opvalue] = df_scenario2[condition].apply(
        lambda x: vanilla_call_price(x[new_S0], x["strike"],
                                     r, x[new_vol], x["Time to maturity"]) if x["right"] == "C"
        else vanilla_put_price(x[new_S0], x["strike"],
                               r, x[new_vol], x["Time to maturity"]), axis=1)
    return df_scenario2, new_opvalue

