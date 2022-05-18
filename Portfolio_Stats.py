
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
from app import components_colors
from utils_IB import OptionPortflio

text_font_size = '1.5vh'
#top_table_df=pd.read_csv('temp_files/df_top_table.csv')
#net_positions_df=pd.read_csv('temp_files/df_netpositions.csv')


def prepare_stats_layout(tabs):


    table1_header = html.Div(html.H1('My Portfolio',
                                     style=dict(fontSize='2vh', fontWeight='bold', color='#0b1a50',
                                                marginTop='')),
                             style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))

    table2_header = html.Div(html.H1('Net Value Exposures vs ΔS&P500=1%',
                                     style=dict(fontSize='2vh', fontWeight='bold', color='#0b1a50',
                                                marginTop='')),
                             style=dict(display='inline-block', marginLeft='', textAlign="center", width='100%'))



    layout=[ dbc.Col([dbc.Card(dbc.CardBody([ tabs ])
                                            , style=dict(backgroundColor='transparent',
                                            border='1px solid {}'.format(components_colors['Main Background'][0])), id='card1',
                                            className='tabs-card'), html.Br()
                                   ], xl=dict(size=2, offset=0), lg=dict(size=2, offset=0),
                                  md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                                  style=dict(paddingLeft='', paddingRight='', border='')),

        dbc.Col([dbc.Card(dbc.CardBody([table1_header,dbc.Spinner([html.Div(id='created_portfolio_div')],
                                                    size="lg", color="primary", type="border", fullscreen=False)

                                                          ])
                                            , style=dict(backgroundColor='#F5F5F5',
                                            border='',style=dict(height='')), id='card1',
                                            className='card1'), html.Br()
                                   ], xl=dict(size=10, offset=0), lg=dict(size=10, offset=0),
                                  md=dict(size=8, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                                  style=dict(paddingLeft='', paddingTop='1.5vh', border='')),

        dbc.Col([dbc.Card(dbc.CardBody([table2_header,
            dbc.Spinner([html.Div(id='net_values_table_div')], size="lg", color="primary", type="border", fullscreen=False)

           ])
            , style=dict(backgroundColor='#F5F5F5',style=dict(height='25vh'))), html.Br()
        ], xl=dict(size=5, offset=2), lg=dict(size=5, offset=2),
            md=dict(size=4, offset=0), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1)),

        dbc.Col([dbc.Card(dbc.CardBody(
            [dbc.Spinner([html.Div(id='graph1_div')], size="lg", color="primary", type="border", fullscreen=False)
                       ,
                       ])
            , style=dict(backgroundColor='#F5F5F5',style=dict(height='25vh'))), html.Br()
        ], xl=dict(size=5, offset=0), lg=dict(size=5, offset=0),
            md=dict(size=4, offset=0), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1)),

        dbc.Col
             ([dbc.Card(dbc.CardBody([html.Div(id='mymenus'),dbc.Spinner([html.Div(id='graph2_div')],
                                                 size="lg", color="primary", type="border", fullscreen=False)

                                ])
            , style=dict(backgroundColor='#F5F5F5',style=dict(height=''))), html.Br()
        ], xl=dict(size=10, offset=2), lg=dict(size=10, offset=2),
            md=dict(size=10, offset=1), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1)

              , style=dict(paddingRight='2vw')
              )

    , dcc.Store(id="df_proc", data=pd.DataFrame().to_dict(), storage_type="memory")


    ]
    return layout

def get_stats_layout(dff):

    print('heeeeeeeeeeey3')

    op=OptionPortflio(dff)
    print('heeeeeeeey2222')


    top_table_df=op.top_table



    net_positions_df=op.net_value_greeks


    surface_fig=op.surface_fig

    top_table_df=top_table_df.reset_index()
    top_table_df.drop('index',axis=1,inplace=True)
    net_positions_df=net_positions_df.reset_index()[['symbol', 'valueDelta' , 'valueGamma', 'valueVega']]



    sum_valuepos = net_positions_df[['symbol', 'valueDelta', 'valueGamma', 'valueVega']].sum(axis=0)

    sum_valuepos.symbol = "Portfolio"

    net_positions_df = pd.concat([net_positions_df, sum_valuepos.to_frame().T], ignore_index=True, axis=0)

    tickers_list=list(set(top_table_df['symbol'].values))

    dates_list=list(set(top_table_df['expiration'].values))

    net_fig=op.get_payoff(tickers_list[0],dates_list[0])

    df_proc=op.df_proc

    created_portfolio_table=dash_table.DataTable(
                id='created_portfolio_table',
                columns=[
                    {"name": i, "id": i} for i in top_table_df.columns
                ],
                data=top_table_df.to_dict("records"),
                editable=False,
                row_deletable=False,
        style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                        , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
        style_header=dict(backgroundColor='#0b1a50', color='white',
                          fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
        style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%','border':'1px solid #0b1a50'}
            )



    net_values_table=dash_table.DataTable(
                id='net_values_table',
                columns=[
                    {"name": i, "id": i} for i in net_positions_df.columns
                ],
                data=net_positions_df.to_dict("records"),
                editable=False,
                row_deletable=False,
        style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                        , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
        style_header=dict(backgroundColor='#0b1a50', color='white',
                          fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
        style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%','border':'1px solid #0b1a50'}
            )




    surface_fig.update_layout(
        title_text='<b>Delta-Gamma-Vega Approximation<b>',title_x=0.5,
        font=dict(size=13, family='Arial', color='#0b1a50'), hoverlabel=dict(
            font_size=13, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        xaxis=dict(

            tickwidth=2, tickcolor='#80ced6',
            ticks="outside",
            tickson="labels",
            rangeslider_visible=False
        ) ,margin=dict(l=0, r=0, t=30, b=0)
    )

    surface_fig.update_xaxes(showgrid=True, showline=True, zeroline=False, linecolor='#0b1a50')
    surface_fig.update_yaxes(showgrid=True, showline=True, zeroline=False, linecolor='#0b1a50')

    graph1_div=dcc.Graph(id='graph1', config={'displayModeBar': True, 'scrollZoom': True,'displaylogo': False},
                style=dict(height='30vh',backgroundColor='#F5F5F5') ,figure=surface_fig
            )


    tickers_dropdown = dcc.Dropdown(
        id='tickers_dropdown', options=[{'label': name, 'value': name} for name in tickers_list],
        value=tickers_list[0],
        style=dict(color='white', fontWeight='bold', textAlign='center',
                   width='8vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
    )

    tickers_dropdown_div = html.Div([tickers_dropdown],
                                    style=dict(fontSize=text_font_size,
                                               marginLeft='', marginBottom='', display='inline-block'))

    expirations_dropdown = dcc.Dropdown(
        id='expirations_dropdown', options=[{'label': name, 'value': name} for name in dates_list],
        value=dates_list[0]
        , style=dict(color='white', fontWeight='bold', textAlign='center',
                     width='8vw', backgroundColor='#0b1a50', border='1px solid #0b1a50')
    )

    expirations_dropdown_div = html.Div([expirations_dropdown],
                                        style=dict(fontSize=text_font_size,
                                                   marginLeft='2vw', marginBottom='', display='inline-block'))

    menues_row = html.Div([tickers_dropdown_div, expirations_dropdown_div],
                          style={'width': '100%', 'display': 'flex', 'align-items': 'left',
                                 'justify-content': 'left', 'padding-left': '2vw'})

    net_fig.update_layout(
        title_text='<b>Payoff<b>',title_x=0.5, xaxis_title='<b>Strike<b>',yaxis_title='<b>Payoff<b>',
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

    net_fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    net_fig.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')

    graph2_div=dcc.Graph(id='graph2', config={'displayModeBar': True, 'scrollZoom': True,'displaylogo': False},
                style=dict(height='35vh',backgroundColor='#F5F5F5') ,figure=net_fig
            )


    return (created_portfolio_table,net_values_table,menues_row,graph1_div,graph2_div,df_proc.to_dict('records'))