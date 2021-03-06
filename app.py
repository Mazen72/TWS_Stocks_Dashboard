import time

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
import BuildPortfolio , Portfolio_Stats ,Scenario_Analysis
#from ScenarioAnalysis import SCENARIOANALYSIS
from utils_IB import IbConnect ,OptionPortflio,multi_plotter
from dash.exceptions import PreventUpdate


#from ScenarioAnalysis import df_indices
import yfinance as yf

#print(op.get_value_greeks())
server = Flask(__name__)
app = dash.Dash(
    __name__,server=server,
    meta_tags=[
        {
            'charset': 'utf-8',
        },
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0, shrink-to-fit=no'
        }
    ] ,
)

app.title='OptionAnalytica'
app.config.suppress_callback_exceptions = True

components_colors={ 'Main Header Background': ['#0b1a50', '#0b1a50'], 'Main Background': ['#e7f0f9', '#e7f0f9'],
                    'Main Header Text': ['white', 'white']}
text_font_size = '1.5vh'


header_text=html.Div('OptionAnalytica',id='main_header_text',className='main-header',
                     style=dict(color=components_colors['Main Header Text'][0],
                     fontWeight='bold',fontSize='2.5vh',marginTop='',marginLeft='',width='100%',paddingTop='1vh',paddingBottom='',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))

db_header_text=  dbc.Col([ header_text] ,
        xs=dict(size=10,offset=0), sm=dict(size=10,offset=0),
        md=dict(size=8,offset=0), lg=dict(size=6,offset=0), xl=dict(size=6,offset=0))

encoded = base64.b64encode(open(r"images/UNIGE_logo.jpg", 'rb').read())
logo_img=html.Img(src='data:image/jpg;base64,{}'.format(encoded.decode()), id='logo_img', height='60vh',
                  style=dict(paddingLeft='',border=''.format(['Main Header Background'][0])))
db_logo_img=dbc.Col([ logo_img] ,
        xs=dict(size=2,offset=0), sm=dict(size=2,offset=0),
        md=dict(size=2,offset=0), lg=dict(size=3,offset=0), xl=dict(size=3,offset=0))


tabs_styles = {
    'height': '',
    'width':'15vw'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',

'backgroundColor': components_colors['Main Header Background'][0],
'color': 'white'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'fontWeight': 'bold',
    'padding': '6px'
}

tabs=dcc.Tabs(id="tabs", value='Main Page',vertical=True, children=[
    dcc.Tab(label='Main Page', value='Main Page', style=tab_style, selected_style=tab_selected_style),
    dcc.Tab(label='Build Portfolio', value='Build Portfolio', style=tab_style, selected_style=tab_selected_style),
    dcc.Tab(label='Portfolio Stats', value='Portfolio Stats', style=tab_style, selected_style=tab_selected_style),
    dcc.Tab(label='Scenario Analysis', value='Scenario Analysis', style=tab_style, selected_style=tab_selected_style)
], style=tabs_styles)

image_header1=html.Div('OptionAnalytica',id='image_header1',className='main-header',
                     style=dict(color='#0b1a50',
                     fontWeight='bold',fontSize='2.5vh',marginTop='',marginLeft='',width='100%',paddingTop='',paddingBottom='',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))

image_header2=html.Div("The first dashboard for options scenario analysis",id='image_header2',className='main-header',
                     style=dict(color='#0b1a50',
                     fontWeight='bold',fontSize='2.5vh',marginTop='',marginLeft='',width='100%',paddingTop='',paddingBottom='',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))

image_header3=html.Div("Version: Beta 1.0.0",id='image_header3',className='main-header',
                     style=dict(color='#0b1a50',
                     fontWeight='bold',fontSize='1.8vh',marginTop='',marginLeft='',width='100%',paddingTop='',paddingBottom='',
                     display= 'flex', alignItems= 'left', justifyContent= 'left'))

image_header4=html.Div("Devs: Gianluca Baglini, David Alcal??",id='image_header4',className='main-header',
                     style=dict(color='#0b1a50',
                     fontWeight='bold',fontSize='1.8vh',fontColor='black',marginLeft='',width='100%',paddingTop='',paddingBottom='',
                     display= 'flex', alignItems= 'left', justifyContent= 'left'))

encoded2 = base64.b64encode(open(r"images/main_page.jpg", 'rb').read())
main_img=html.Img(src='data:image/jpg;base64,{}'.format(encoded2.decode()), id='main_img',className='myimg',
                  style=dict(paddingLeft='',border=''.format(['Main Header Background'][0]),width='100%',height='65vh',
                             minHeight='65vh') )

main_page_layout=[ dbc.Col([dbc.Card(dbc.CardBody([
                                                         tabs

                                                          ])
                                            , style=dict(backgroundColor='transparent',
                                            border='1px solid {}'.format(components_colors['Main Background'][0])), id='card1',
                                            className='tabs-card'), html.Br()
                                   ], xl=dict(size=2, offset=0), lg=dict(size=2, offset=0),
                                  md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                                  style=dict(paddingLeft='', paddingRight='', border='')),

        dbc.Col([image_header2,html.Br(), main_img, html.Br(), html.Br(),image_header3 ,html.Br(),image_header4
                                   ], xl=dict(size=8, offset=0), lg=dict(size=8, offset=0),
                                  md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                                  style=dict(paddingLeft='', paddingRight='', border=''))
                ]

app.layout=html.Div([

dbc.Row([db_logo_img,db_header_text],
                              style=dict(backgroundColor=components_colors['Main Header Background'][0]),id='main_header' )
,html.Br(),
                       dbc.Row([    dbc.Col([dbc.Card(dbc.CardBody([
                                                         tabs

                                                          ])
                                            , style=dict(backgroundColor='transparent',
                                            border='1px solid {}'.format(components_colors['Main Background'][0])), id='card1',
                                            className='tabs-card'), html.Br()
                                   ], xl=dict(size=2, offset=0), lg=dict(size=2, offset=0),
                                  md=dict(size=4, offset=0), sm=dict(size=12, offset=0), xs=dict(size=12, offset=0),
                                  style=dict(paddingLeft='', paddingRight='', border='')),




                           ],id='content') ,dcc.Store(id="portfolio_created", data=pd.DataFrame().to_dict(), storage_type="memory")
,dcc.Store(id="df_proc2", data=pd.DataFrame().to_dict(), storage_type="session"),
dcc.Store(id="df_betas", data=pd.DataFrame().to_dict(), storage_type="session")
,dcc.Store(id="stats_tab", data='no', storage_type="memory")
                     ,html.Br(),html.Br(),html.Br()]

,style=dict(backgroundColor=components_colors['Main Background'][0])
,className='main'
)


@app.callback([Output('created_portfolio_div','children'),Output('net_values_table_div','children'),
               Output('mymenus', 'children'), Output('graph1_div', 'children') ,Output('graph2_div', 'children'),
               Output('df_proc','data')],
              Input('stats_tab','data') ,State('portfolio_created','data'))

def get_stats_tab_layout(tab3_state,portfolio_created):
    if tab3_state=='pressed':
        print("heeeeeeeeeeeeeeeey")
        dff = pd.DataFrame(portfolio_created)
        print(dff.info())
      #  try:
        return Portfolio_Stats.get_stats_layout(dff)
       # except:
      #      raise PreventUpdate


    else:
        raise PreventUpdate

@app.callback([Output('content','children'),Output('stats_tab','data')],
              Input("tabs", "value"))
def update_tab_content(selected_tab):
    if selected_tab=='Main Page':
        tabs = dcc.Tabs(id="tabs", value=selected_tab, vertical=True, children=[
            dcc.Tab(label='Main Page', value='Main Page', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Build Portfolio', value='Build Portfolio', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Portfolio Stats', value='Portfolio Stats', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Scenario Analysis', value='Scenario Analysis', style=tab_style,
                    selected_style=tab_selected_style)
        ], style=tabs_styles)

        return (main_page_layout,'')

    elif selected_tab=='Build Portfolio':
        tabs = dcc.Tabs(id="tabs", value=selected_tab, vertical=True, children=[
            dcc.Tab(label='Main Page', value='Main Page', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Build Portfolio', value='Build Portfolio', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Portfolio Stats', value='Portfolio Stats', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Scenario Analysis', value='Scenario Analysis', style=tab_style,
                    selected_style=tab_selected_style)
        ], style=tabs_styles)
        return (BuildPortfolio.get_layout(tabs) ,'')

    elif selected_tab=='Portfolio Stats':
        tabs = dcc.Tabs(id="tabs", value=selected_tab, vertical=True, children=[
            dcc.Tab(label='Main Page', value='Main Page', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Build Portfolio', value='Build Portfolio', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Portfolio Stats', value='Portfolio Stats', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Scenario Analysis', value='Scenario Analysis', style=tab_style,
                    selected_style=tab_selected_style)
        ], style=tabs_styles)

        return (Portfolio_Stats.prepare_stats_layout(tabs),'pressed')

    elif selected_tab=='Scenario Analysis':
        tabs = dcc.Tabs(id="tabs", value=selected_tab, vertical=True, children=[
            dcc.Tab(label='Main Page', value='Main Page', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Build Portfolio', value='Build Portfolio', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Portfolio Stats', value='Portfolio Stats', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Scenario Analysis', value='Scenario Analysis', style=tab_style,
                    selected_style=tab_selected_style)
        ], style=tabs_styles)
        return ( Scenario_Analysis.get_scenario_layout(tabs),'')

@app.callback(Output('chart1','figure'),Input('options_menu','value'),

              [State('drift_input', 'value'), State('volatility_input', 'value'), State('prob_input', 'value'),
               State('risk_input', 'value'), State('horizon_input', 'value'),
               State('intensity_input', 'value'),
               State('df_proc2', 'data'), State('df_betas', 'data')]

              )
def update_scenarios_chart1(option,drift_input,volatility_input,prob_input,risk_input,horizon_input,
                          intensity_input,df_proc,df_betas):
    if option=='Yield Curve':
        fig1=Scenario_Analysis.get_yield_curve()
        fig1.update_layout(
            # title_text='<b>Payoff<b>',title_x=0.5, xaxis_title='<b>Strike<b>',yaxis_title='<b>Payoff<b>',
            font=dict(size=14, family='Arial', color='#0b1a50'), hoverlabel=dict(
                font_size=14, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            xaxis=dict(

                tickwidth=2, tickcolor='#80ced6',
                ticks="outside",
                tickson="labels",
                rangeslider_visible=False
            ), margin=dict(l=0, r=0, t=40, b=0)
        )

        fig1.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
        fig1.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')

        return fig1

    elif option=='Index Simulations':
        df_proc = pd.DataFrame(df_proc)
        df_betas = pd.DataFrame(df_betas)
        fig2 = Scenario_Analysis.jump_diffusion_process("^GSPC", drift_input, volatility_input, risk_input, horizon_input
                                                 , prob_input, intensity_input, df_proc, df_betas)[3]

        fig2.update_layout(
            title_text='<b>Custom tick labels with ticklabelmode="period"<b>', title_x=0.5,
            font=dict(size=14, family='Arial', color='#0b1a50'), hoverlabel=dict(
                font_size=14, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            xaxis=dict(

                tickwidth=2, tickcolor='#80ced6',
                ticks="outside",
                tickson="labels",
                rangeslider_visible=False
            ), margin=dict(l=0, r=0, t=40, b=0)
        )

        fig2.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
        fig2.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')

        return fig2

    else:
        raise PreventUpdate


#Input("tickers_dropdown", "value")
@app.callback(Output('expirations_dropdown','options'),
              Input('tickers_dropdown','value'),
              State('created_portfolio_table','data')
              )
def update_expirations(ticker,table_data):
    top_table_df=pd.DataFrame(table_data)
    exp_list=top_table_df[top_table_df['symbol']==ticker]['expiration'].to_list()
    return [{'label': exp, 'value': exp} for exp in exp_list]



@app.callback(Output('graph2','figure'),
              Input('expirations_dropdown','value'),
              [State('portfolio_created','data'),State('df_proc','data'),State("tickers_dropdown", "value") ]
              )
def update_line_chart(expiration,portfolio_created,df_proc,stock):
    '''
    portfolio_df=pd.DataFrame(portfolio_created)
    portfolio_df['undPrice']=portfolio_df['Quantity']
    portfolio_df['maturity']='20 days'
    portfolio_df['expiration']=20220527
    portfolio_df=portfolio_df[['Unnamed: 0','strike','expiration','undPrice','Trade','modelDelta','modelGamma','modelIV','modelPrice','modelTheta','modelVega','maturity'
               ]]
    start=time.time()
    op=OptionPortflio(portfolio_df)
    print(time.time()-start)
    line_fig=op.get_payoff(selected_ticker,selected_expiration)
    '''

    df = pd.DataFrame(df_proc)
    condition = (df["symbol"] == stock) & (df["expiration"] == int(expiration))
    df_filt = df[condition]
    op_list = []
    for row, item in df_filt.iterrows():
        strike = item["strike"]
        op_price = 10  # todo change item["modelBid"]
        right = item["right"].lower()
        trade = "s" if item["Trade"] == "Sell" else "b"
        undPrice = item["undPrice"]
        op = {'op_type': right, 'strike': strike, 'tr_type': trade, 'op_pr': op_price}
        op_list.append(op)

    spot_range = abs(strike - undPrice + 20)
    df_pay = multi_plotter(spot=undPrice, spot_range=spot_range, op_list=op_list)

    line_fig = px.line(df_pay, x="strike", y="payoff")

    line_fig.update_layout(
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
    line_fig.add_hline(y=0.0, line_color="red")
    line_fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    line_fig.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    return line_fig

@app.callback([Output('exchanges-out', 'options'),
              Output('store-options-exch', 'data'),
              Output('candle', 'figure')],
              [Input('select-ticker', 'value')],
              prevent_initial_call=True)
def get_exchanges(ticker):
    ibc = IbConnect()
    currency = BuildPortfolio.df_tickers.loc[BuildPortfolio.df_tickers["TICKER"] == ticker, "CURRENCY"].values[0]
    pexchange = BuildPortfolio.df_tickers.loc[BuildPortfolio.df_tickers["TICKER"] == ticker, "EXCHANGE"].values[0]
    chains, df_chains = ibc.read_exchanges(ticker, currency, pexchange)
    df_data = ibc.read_historical_data(ticker, currency)
    exchange = df_chains["exchange"].values

    fig = go.Figure(data=[go.Candlestick(x=df_data['date'],
                                   open=df_data['open'], high=df_data['high'],
                                   low=df_data['low'], close=df_data['close'])
                    ])
    fig.update_layout(
        title='{}'.format(ticker), xaxis_title='Date', yaxis_title="Stock Price",
        font=dict(size=14, family='Arial', color='#0b1a50'), hoverlabel=dict(
            font_size=16, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        xaxis=dict(

            tickwidth=2, tickcolor='#80ced6',
            ticks="outside",
            tickson="labels",
            rangeslider_visible=False
        )
    )
    fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    fig.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    return exchange, df_chains.to_dict("records"), fig




'''
@app.callback(Output('store-option-chain', 'data'),
              [Input('datatable-selection', 'data')],
              prevent_initial_call=True)
def save_row(row):
    return row.to_dict("records")
'''

@app.callback([Output('display_selected_row', "children"),Output('portifolio_button', "style"),
               Output('portifolio_in_progress','data')],
              Input('adding-option-button', 'n_clicks'),
              [State('datatable-selection', "selected_rows"),State('datatable-selection', "data"),
               State('quantity_input','value'),State('trade_options','value'),
               State('portifolio_in_progress','data')],
               prevent_initial_call=True)
def add_option_to_portfolio(n_clicks,selected_rows,rows,quantity_input,trade_option,current_portifolio_data):



    if quantity_input == None and n_clicks!=0:
        return ( html.Div([
                'please enter the quantity in the input box',
            ],style=dict(fontSize='1.7vh',fontWeight='bold',color='red',textAlign='center')) ,dash.no_update,dash.no_update)


    else:
        if selected_rows is None:
            selected_rows = []

        if rows is None:
            dff = pd.DataFrame()
        else:
            dff = pd.DataFrame(rows)

        dff = dff.iloc[selected_rows]


        dff['Quantity']=quantity_input
        dff['Trade']=trade_option

        dff=pd.concat([pd.DataFrame(current_portifolio_data),dff])

        create_portifolio = html.Div(dbc.Button(
            "Create Portfolio", id="create-portfolio-button", className="", n_clicks=0, size='lg',
            style=dict(fontSize='1.7vh', backgroundColor='#119DFF')
        ), style=dict(textAlign='center', marginTop='', paddingLeft='', width='100%',
                      display='flex', alignItems='center', justifyContent='center'))

        return (
            dash_table.DataTable(
                id='selected_portifolio',
                columns=[
                    {"name": i, "id": i} for i in dff.columns
                ],
                data=dff.to_dict("records"),
                editable=True,
                row_deletable=True,
        style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                        , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
        style_header=dict(backgroundColor='#0b1a50', color='white',
                          fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
        style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%','border':'1px solid #0b1a50'}
            ),

            dict(textAlign='center', marginTop='', paddingLeft='', width='100%',
        display='flex', alignItems='center', justifyContent='center')
            ,dff.to_dict("records"))

'''
html.Div([
                'Portfolio Created Successfully',
            ],style=dict(fontSize='1.7vh',fontWeight='bold',color='green',textAlign='center'))
'''
# df_proc2 , df_betas
@app.callback([Output('portfolio_msg','children'),Output('portfolio_created','data'),
               Output('df_proc2','data'),Output('df_betas','data')
               ],
              [Input('create-portfolio-button','n_clicks'),Input('select-ticker','value')],
              State('portifolio_in_progress','data'),
              prevent_initial_call=True)
def create_portfolio(clicks,ticker_changed,portfolio_data):
    if clicks==0:
        raise PreventUpdate
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    df = pd.DataFrame(portfolio_data)
    op=OptionPortflio(df)
    df_proc=op.df_proc
    df_betas=op.df_betas

    if input_id=='select-ticker':
        return ('',dash.no_update,dash.no_update,dash.no_update)

    if clicks > 0:
        return ( html.Div([
                'Portfolio Created Successfully',
            ],style=dict(fontSize='1.7vh',fontWeight='bold',color='green',textAlign='center')) , portfolio_data,
                df_proc.to_dict('records'),df_betas.to_dict('records'))

    else:
        raise PreventUpdate

    # drift_input volatility_input prob_input risk_input horizon_input options_menu Index Simulations Yield Curve
    # simulate_button
    #  chart2 middle_table_div bottom_table_div

#(index, drift, sigma, r,  time_in_months, lam, jump, df_proc,df_betas ,plot = True)
@app.callback([Output('chart2','figure'),Output('middle_table_div','children'),
               Output('bottom_table_div','children')
               ],
              Input('simulate_button','n_clicks'),
              [State('drift_input','value'),State('volatility_input','value'),State('prob_input','value'),
               State('risk_input','value'),State('horizon_input','value'),State('options_menu','value'),
               State('intensity_input','value'),
               State('df_proc2','data'),State('df_betas','data')],
              prevent_initial_call=True)
def simulate(n_clicks,drift_input,volatility_input,prob_input,risk_input,horizon_input,options_menu,intensity_input,df_proc,df_betas):
    df_proc=pd.DataFrame(df_proc)
    df_betas=pd.DataFrame(df_betas)
    df,bottom_table_df,hist,fig1=Scenario_Analysis.jump_diffusion_process("^GSPC",drift_input,volatility_input,risk_input,horizon_input
                                                 ,prob_input,intensity_input,df_proc,df_betas)



    hist.update_layout(
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

    hist.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
    hist.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')


    ret3,ret4=Scenario_Analysis.simulate_baseline_scenario(drift_input, volatility_input, risk_input, horizon_input,df_proc,df_betas)
    middle_table_df=ret3

    middle_table=dash_table.DataTable(
                id='middle_table',
                columns=[
                    {"name": i, "id": i} for i in middle_table_df.columns
                ],
                data=middle_table_df.to_dict("records"),
                editable=False,
                row_deletable=False, page_size=6,
        style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                        , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
        style_header=dict(backgroundColor='#0b1a50', color='white',
                          fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
        style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%','border':'1px solid #0b1a50'}
            )


    bottom_table=dash_table.DataTable(
            id='bottom_table',
            columns=[
                {"name": i, "id": i} for i in bottom_table_df.columns
            ],
            data=bottom_table_df.to_dict("records"),
            editable=False,
            row_deletable=False,page_size=6,
            style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                            , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
            style_header=dict(backgroundColor='#0b1a50', color='white',
                              fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
            style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%', 'border': '1px solid #0b1a50'}
        )



    return (hist,middle_table,bottom_table)







@app.callback([Output('options_exception','children'),
               Output('portfolio_content','children')],
               Input('get-chain-in', 'n_clicks'),
              [State('store-options-exch', 'data'),State('opt-type', 'value'),
               State('exchanges-out', 'value'),State('select-ticker', 'value')],
               prevent_initial_call=True)
def get_option_chain(n_clicks ,dict_exchange , right, exchange,ticker ):


    if dict_exchange==None or right==None or exchange==None or ticker==None :
        return (html.Div([
                'Please fill the rest of the dropdowns',
            ],style=dict(fontSize='1.6vh',fontWeight='bold',color='red',textAlign='center')),dash.no_update)
    else:
        my_df=pd.DataFrame()
        if ticker == "AAPL" and right == "P":
            my_df=pd.read_csv("temp_files/AAPL_PUT.csv")
            column_means = my_df.mean()
            my_df = my_df.fillna(column_means)

        elif ticker == "GS" and right == "C":
            my_df=pd.read_csv("temp_files/GS_CALL.csv")
            column_means = my_df.mean()
            my_df = my_df.fillna(column_means)
        elif ticker == "TSLA" and right == "P":
            my_df=pd.read_csv("temp_files/TSLA_PUT.csv")
            column_means = my_df.mean()
            my_df = my_df.fillna(column_means)
        else:
            raise PreventUpdate

        cols_to_format2f = ["bid", "ask", "undPrice"]
        cols_to_format3f = ["modelDelta", "modelGamma", "modelIV", "modelPrice", "modelTheta", "modelVega"]
        my_df[cols_to_format2f] = my_df[cols_to_format2f].applymap(lambda num :round(num,2))
        my_df[cols_to_format3f] = my_df[cols_to_format3f].applymap(lambda num :round(num,5))

        undPrice = my_df["undPrice"].values[0]
        my_df['maturity'] = pd.to_datetime(my_df["expiration"], format='%Y%m%d') - pd.Timestamp.now().normalize()
        my_df["maturity"] = my_df["maturity"].astype(str).str.replace("days", "").astype(float)
        my_df["moneyness"] = my_df["strike"] / my_df["undPrice"].astype(float)
        # z_data = z_data.set_index("maturity")
        data_3D = my_df.pivot_table(index='moneyness', columns='maturity', values="modelIV")

        fig3D = go.Figure(data=[go.Surface(z=data_3D.values)])
        fig3D.update_layout(title='Volatilty Surface')
      #  fig3D.show()


        data_smile = pd.pivot_table(my_df, values='modelIV', index=['strike'],columns='expiration')

        # ploty
        figline = go.Figure()
        for col in data_smile.columns:
            figline.add_trace(go.Scatter(x=data_smile.index, y=data_smile[col].values,
                                     name=col,
                                     mode='markers+lines',
                                     line=dict(shape='linear'),
                                     connectgaps=True
                                     )
                          )
        figline.add_vline(x=undPrice, line_width=3, line_dash="dash", line_color="green")
       # figline.show()

        table= dash_table.DataTable(
            id='datatable-selection',
            columns=[
                {'name': str(i), 'id': str(i), 'deletable': False} for i in my_df.columns
                # omit the id column
                if i != 'id'
            ],
            data=my_df.to_dict('records'),
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            row_selectable='multi',
            selected_rows=[],
            page_action='native',
            page_current=0,
            page_size=8,

        style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                        , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
        style_header=dict(backgroundColor='#0b1a50', color='white',
                          fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
        style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%','border':'1px solid #0b1a50'}
        )

        quantity_input_header = html.H1('Please Select a Quantity',
                                        style=dict(fontSize='1.4vh', fontWeight='bold', color='black',
                                                   textAlign='center'))
        quantity_input = html.Div([quantity_input_header,
                                   dbc.Input(
                                       placeholder='select a number ',
                                       n_submit=0,
                                       type='number',
                                       id='quantity_input', autocomplete='off', style=dict(border='1px solid #0b1a50')
                                   )], style=dict(width='10vw', display='inline-block'))

        add_to_portifolio = html.Div(dbc.Button(
            "Add To Portfolio", id="adding-option-button", className="ms-auto", n_clicks=0, size='lg',
            style=dict(fontSize=text_font_size, backgroundColor='#119DFF')
        ), style=dict(textAlign='center', display='inline-block', marginTop='1.5%', paddingLeft='2vw'))

        trade_text = html.Div(html.H1('Trade', className='filters-header', id='allowance_text',
                                      style=dict(fontSize='1.7vh', fontWeight='bold',
                                                 color='#0b1a50',
                                                 marginTop='')),
                              style=dict(display='', marginLeft='', textAlign="center"))

        trade_options = html.Div(
            [
                dbc.RadioItems(options=[{"label": "Buy", "value": 'Buy'},
                                        {"label": "Sell", "value": 'Sell'}, ],
                               value='Buy',
                               id="trade_options",
                               inline=False, label_class_name='filter-label', input_class_name='filter-button',
                               input_checked_class_name='filter-button-checked',
                               input_style=dict(border='1px solid #0b1a50'),
                               input_checked_style=dict(backgroundColor='#0b1a50', border='1px solid #0b1a50')
                               ),
            ]
        )

        trade_options_div = html.Div([trade_text, trade_options],
                                     style=dict(fontSize='', display='inline-block', marginLeft='2vw', textAlign=""))

        add_to_portifolio_div = html.Div([quantity_input, trade_options_div, add_to_portifolio],
                                         className='add-portifolio-div',
                                         style=dict(width='100%', display='flex', alignItems='center',
                                                    justifyContent='center')
                                         )

        figline.update_layout(
            title='<b>undPrice<b>', title_x=0.5,
            font=dict(size=13, family='Arial', color='#0b1a50'), hoverlabel=dict(
                font_size=13, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            xaxis=dict(

                tickwidth=2, tickcolor='#80ced6',
                ticks="outside",
                tickson="labels",
                rangeslider_visible=False
            ), margin=dict(l=0, r=0, t=40, b=0)
        )

        figline.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')
        figline.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='#0b1a50')

        graph1_div = html.Div([
            dcc.Graph(id='figline', config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                      style=dict(height='30vh', backgroundColor='#F5F5F5'), figure=figline
                      )], id='figline_div'
        )

        fig3D.update_layout(
            title_text='<b>Volatilty Surface<b>', title_x=0.5,
            font=dict(size=13, family='Arial', color='#0b1a50'), hoverlabel=dict(
                font_size=13, font_family="Rockwell", font_color='white', bgcolor='#0b1a50'), plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            xaxis=dict(

                tickwidth=2, tickcolor='#80ced6',
                ticks="outside",
                tickson="labels",
                rangeslider_visible=False
            ), margin=dict(l=0, r=0, t=30, b=0)
        )

        fig3D.update_xaxes(showgrid=True, showline=True, zeroline=False, linecolor='#0b1a50')
        fig3D.update_yaxes(showgrid=True, showline=True, zeroline=False, linecolor='#0b1a50')

        graph2_div = html.Div([
            dcc.Graph(id='fig3D', config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                      style=dict(height='30vh', backgroundColor='#F5F5F5'), figure=fig3D
                      )], id='fig3D_div'
        )

        graph1_col=dbc.Col([dbc.Card(dbc.CardBody(
            [html.Div([dbc.Spinner([graph1_div], size="lg", color="primary", type="border", fullscreen=False)

                       ], style=dict(height=''))])
            , style=dict(backgroundColor='#F5F5F5',border='0.5px solid #0b1a50')), html.Br()
        ], xl=dict(size=5, offset=1), lg=dict(size=5, offset=1),
            md=dict(size=10, offset=1), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1))

        graph2_col=dbc.Col([dbc.Card(dbc.CardBody(
            [html.Div([dbc.Spinner([graph2_div], size="lg", color="primary", type="border", fullscreen=False)

                       ], style=dict(height=''))])
            , style=dict(backgroundColor='#F5F5F5',border='0.5px solid #0b1a50')), html.Br()
        ], xl=dict(size=5, offset=0), lg=dict(size=5, offset=0),
            md=dict(size=10, offset=1), sm=dict(size=10, offset=1), xs=dict(size=10, offset=1))

        figures_row=dbc.Row([graph1_col,graph2_col])
        return ('',[table, figures_row,add_to_portifolio_div
                       ])

#@app.callback([Output('display-option-chain', 'children'), Output('options_exception', 'children')],
#              Input('simulate_button', 'n_clicks'),
#              [State('drift_input', 'value'), State('volatility_input', 'value'), State('prob_input', 'value'),
#               State('intensity_input', 'value'), State('risk_input', 'value'), State('horizon_input', 'value')]
#             )
#def get_simulation_baseline(n_clicks, drift, vol, prob, intensity, rf, horizon):
#    if n_clicks == 0 or drift == None or vol == None or prob == None or intensity == None or rf == None or horizon == None:
#       return (dash.no_update,html.Div([
#                'Please fill the rest of the inputs',
#            ],style=dict(fontSize='1.6vh',fontWeight='bold',color='red',textAlign='center')))
#    else:


if __name__ == '__main__':
    app.run_server(host='localhost',port=8050,debug=True,dev_tools_silence_routes_logging=True)




