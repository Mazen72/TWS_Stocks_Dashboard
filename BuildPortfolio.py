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

text_font_size = '1.5vh'
components_colors={ 'Main Header Background': ['#0b1a50', '#0b1a50'], 'Main Background': ['#e7f0f9', '#e7f0f9'],
                    'Main Header Text': ['white', 'white']}

df_tickers = pd.read_excel(r"IB_TickerMapping.xlsx")
ticker_list = list(df_tickers["TICKER"])
exchange_list = set(df_tickers["EXCHANGE"])
df_temp_table = pd.read_csv("temp_files/TSLA_PUT.csv")
column_means = df_temp_table.mean()
df_temp_table = df_temp_table.fillna(column_means)
cols_to_format2f = ["bid", "ask", "undPrice"]
cols_to_format3f = ["modelDelta", "modelGamma", "modelIV", "modelPrice", "modelTheta", "modelVega"]
df_temp_table[cols_to_format2f] = df_temp_table[cols_to_format2f].applymap('{:,.2f}'.format)
df_temp_table[cols_to_format3f] = df_temp_table[cols_to_format3f].applymap('{:,.3f}'.format)
try:
    df_temp_table=df_temp_table.drop('Unnamed: 0')
except:
    pass



def get_layout(tabs):

    ###### temp files ############
    #### tickers


    ### exchanges
    #df_exchanges = pd.read_csv("temp_files/df_exchanges.csv")["exchange"].values
    ### expirations
    #df_exp = pd.read_csv("temp_files/df_exp.csv")
    ### expirations



    menu1 = dcc.Dropdown(
                         options=[{'label': name, 'value': name} for name in exchange_list],
                         value='NASDAQ',
                         id='select-stock-exchange',
                            style=dict(color='white', fontWeight='bold', textAlign='center',
                                       width='9vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
                            )

    menu1_text = html.Div(html.H1('Select a Stock Exchange',
                                style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                           marginTop='')),
                        style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    menu1_div = html.Div([menu1_text, menu1],
                            style=dict(fontSize=text_font_size,
                                       marginLeft='', marginBottom='', display='inline-block'))

    menu2 = dcc.Dropdown(
        options=[{'label': 'TSLA', 'value': 'TSLA'},{'label': 'AAPL', 'value': 'AAPL'},{'label': 'GS', 'value': 'GS'}
                         ],
        value=None,
        id='select-ticker'
        , style=dict(color='white', fontWeight='bold', textAlign='center',
                                                width='9vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
    )

    # text apears above resolution dropdown
    menu2_text = html.Div(html.H1('Select a ticker',
                                       style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                                  marginTop='')),
                               style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    # the div that contains both the text and dropdown of resolution
    menu2_div = html.Div([menu2_text, menu2],
                                   style=dict(fontSize=text_font_size,
                                              marginLeft='2vw', marginBottom='', display='inline-block'))

    menues_row1=html.Div([menu1_div, menu2_div],style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                           'justify-content': 'center'})

    menu3 = dcc.Dropdown(
                         id='opt-type', options=[{'label': 'P', 'value': 'P'},
                                                 {'label': 'C', 'value': 'C'},
                         ], value='P',
                         style=dict(color='white', fontWeight='bold', textAlign='center',
                                    width='9vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
                         )

    menu3_text = html.Div(html.H1('Select Option Type',
                                  style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                             marginTop='')),
                          style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    menu3_div = html.Div([menu3_text, menu3],
                         style=dict(fontSize=text_font_size,
                                    marginLeft='', marginBottom='', display='inline-block'))

    menu4 = dcc.Dropdown(
        id='exchanges-out', options=[] ,value=None
        , style=dict(color='white', fontWeight='bold', textAlign='center',
                                                width='9vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
    )

    # text apears above resolution dropdown
    menu4_text = html.Div(html.H1('Select Option Exchange',
                                       style=dict(fontSize=text_font_size, fontWeight='bold', color='#0b1a50',
                                                  marginTop='')),
                               style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    # the div that contains both the text and dropdown of resolution
    menu4_div = html.Div([menu4_text, menu4],
                                   style=dict(fontSize=text_font_size,
                                              marginLeft='2vw', marginBottom='', display='inline-block'))

    menues_row2=html.Div([menu3_div, menu4_div],style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                           'justify-content': 'center'})

    get_options_button = html.Div([dbc.Button('Get Option Chain', color="primary", size='lg', n_clicks=0, id='get-chain-in'
                                        , style=dict(fontSize=text_font_size,backgroundColor='#119DFF')
                                        )], style=dict(width='100%',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))


    fig=go.Figure(go.Candlestick())

    fig.update_layout(
        title='Time Series Chart', xaxis_title='Date',
        font=dict(size=14, family='Arial', color='#0b1a50'), hoverlabel=dict(
            font_size=16, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        xaxis=dict(

            tickwidth=2, tickcolor='#80ced6',
            ticks="outside",
            tickson="labels",
            rangeslider_visible=False
        ) ,margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    fig.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')

    candle_div=html.Div([
            dcc.Graph(id='candle', config={'displayModeBar': True, 'scrollZoom': True,'displaylogo': False},
                style=dict(height='35vh',backgroundColor='#F5F5F5') ,figure=fig
            ) ] ,id='flow_line_div'
        )



    main_table=html.Div(id='display-option-chain')



    portifolio_table=html.Div(id='display_selected_row')

    portifolio_button=html.Div(dbc.Button(
            "Create Portfolio", id="create-portfolio-button", className="", n_clicks=0, size='lg',
            style=dict(fontSize='1.7vh', backgroundColor='#119DFF')
        ), style=dict(textAlign='center', marginTop='', paddingLeft='', width='100%',
                      display='none', alignItems='center', justifyContent='center'),
        id='portifolio_button')

    portifolio_header = html.Div(html.H1('My Portfolio',
                                       style=dict(fontSize='1.9vh', fontWeight='bold', color='#0b1a50',
                                                  marginTop='')),
                               style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    portifolio_options= html.Div(html.H1('Portfolio Options',
                                       style=dict(fontSize='1.9vh', fontWeight='bold', color='#0b1a50',
                                                  marginTop='')),
                               style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    portifolio_created=html.Div(id='portfolio_msg')

    options_content=html.Div('', style=dict(height=''),id='portfolio_content')

    layout=[ dbc.Col([dbc.Card(dbc.CardBody([
                                                         tabs

                                                          ])
                                            , style=dict(backgroundColor='transparent',
                                            border='1px solid {}'.format(components_colors['Main Background'][0])), id='card1',
                                            className='tabs-card'), html.Br()
                                   ], xl=dict(size=2, offset=0), lg=dict(size=2, offset=0),
                                  md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                                  style=dict(paddingLeft='', paddingRight='', border='')),

        dbc.Col([dbc.Card(dbc.CardBody([menues_row1,html.Br(),menues_row2,html.Br(),get_options_button ,
                                        html.Br(),html.Div(id='options_exception')
                                                          ])
                                            , style=dict(backgroundColor='#F5F5F5',
                                            border=''), id='card2',
                                            className='menus-card'), html.Br()
                                   ], xl=dict(size=3, offset=0), lg=dict(size=3, offset=0),
                                  md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                                  style=dict(paddingLeft='', paddingRight='', border='')),

        dbc.Col([dbc.Card(dbc.CardBody(
            [html.Div([dbc.Spinner([candle_div], size="lg", color="primary", type="border", fullscreen=False)

                       ], style=dict(height=''))])
            , style=dict(backgroundColor='#F5F5F5')), html.Br()
        ], xl=dict(size=6, offset=0), lg=dict(size=6, offset=0),
            md=dict(size=10, offset=1), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1)),

        dbc.Col([dbc.Card(dbc.CardBody(
            [portifolio_options,dbc.Spinner([options_content],
                                size = "lg", color = "primary", type = "border", fullscreen = False)

])
            , style=dict(backgroundColor='#F5F5F5')), html.Br()
        ], xl=dict(size=10, offset=1), lg=dict(size=10, offset=1),
            md=dict(size=10, offset=1), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1)),

        dbc.Col([dbc.Card(dbc.CardBody([portifolio_header,
                        html.Div([dbc.Spinner([portifolio_table], size="lg", color="primary", type="border", fullscreen=False),
                    html.Br(),   portifolio_button , html.Br() ,dbc.Spinner([portifolio_created],
                                size = "lg", color = "primary", type = "border", fullscreen = False)

                       ], style=dict(height=''))])
            , style=dict(backgroundColor='#F5F5F5')), html.Br()
        ], xl=dict(size=10, offset=1), lg=dict(size=10, offset=1),
            md=dict(size=10, offset=1), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1))

    ,dcc.Store(id="store-options-exch", data=None, storage_type="memory"),
    dcc.Store(id="store-option-chain", data=[], storage_type="memory"),
    dcc.Store(id="portifolio_in_progress", data=pd.DataFrame().to_dict(), storage_type="memory"),

    ]


    return layout