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

    horizon_input_text = html.Div(html.H1('Horizon',
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
                         options=[{'label': "Yield Curve", 'value': "Yield Curve"},{'label': "Rf assumption", 'value': "Rf assumption"},
                                  {'label': "Index Simulations", 'value': "Index Simulations"}],
                         value='Yield Curve',
                         id='options_menu',
                            style=dict(color='white', fontWeight='bold', textAlign='center',
                                       width='9vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
                            )


    options_menu_div=html.Div([options_menu],style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                           'justify-content': 'center'})

    fig1=go.Figure()

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
                style=dict(height='34vh',backgroundColor='#F5F5F5') ,figure=fig1
            ) ] ,id='chart1_div'
        )

    middle_table_header = html.Div(html.H1('Portfolio in 6 Months',
                                     style=dict(fontSize='2vh', fontWeight='bold', color='#0b1a50',
                                                marginTop='')),
                             style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    bottom_table_header = html.Div(html.H1('Scenario Portfolio',
                                     style=dict(fontSize='2vh', fontWeight='bold', color='#0b1a50',
                                                marginTop='')),
                             style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    middle_table=html.Div(dash_table.DataTable(
                id='middle_table',
                columns=[
                    {"name": i, "id": i} for i in dummy_df.columns
                ],
                data=dummy_df.to_dict("records"),
                editable=False,
                row_deletable=False,
        style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                        , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
        style_header=dict(backgroundColor='#0b1a50', color='white',
                          fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
        style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%','border':'1px solid #0b1a50'}
            )
        ,
        id = 'middle_table_div')

    bottom_table=html.Div(
        dash_table.DataTable(
            id='bottom_table',
            columns=[
                {"name": i, "id": i} for i in dummy_df.columns
            ],
            data=dummy_df.to_dict("records"),
            editable=False,
            row_deletable=False,
            style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                            , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
            style_header=dict(backgroundColor='#0b1a50', color='white',
                              fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
            style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%', 'border': '1px solid #0b1a50'}
        )

        ,
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
            ) ] ,id='chart2_div'
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

        dbc.Col([dbc.Card(dbc.CardBody([options_menu_div,
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
    print(df)
    #df.columns=['level_0','level_1','DATE']
    df["DATE"] = df["DATE"].astype(str)

    fig1 = px.line(df, x="level_0", y=0, animation_frame="DATE",
    animation_group="level_0", range_y=[0, 5], markers=True)
    fig1.update_layout(
        xaxis_title='<b>Maturity<b>',
        yaxis_title='<b>Rates<b>',
    )
    return fig1