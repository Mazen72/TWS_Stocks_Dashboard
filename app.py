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
import BuildPortfolio , Portfolio_Stats
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

app.title='Risk Management Dashboard'
app.config.suppress_callback_exceptions = True

components_colors={ 'Main Header Background': ['#0b1a50', '#0b1a50'], 'Main Background': ['#e7f0f9', '#e7f0f9'],
                    'Main Header Text': ['white', 'white']}


header_text=html.Div('Risk Management Dashboard',id='main_header_text',className='main-header',
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
    dcc.Tab(label='Intraday Stress Test', value='Intraday Stress Test', style=tab_style, selected_style=tab_selected_style),
    dcc.Tab(label='Scenario Analysis', value='Scenario Analysis', style=tab_style, selected_style=tab_selected_style)
], style=tabs_styles)

image_header1=html.Div('OptionAnalytica',id='image_header1',className='main-header',
                     style=dict(color='#0b1a50',
                     fontWeight='bold',fontSize='2.5vh',marginTop='',marginLeft='',width='100%',paddingTop='',paddingBottom='',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))

image_header2=html.Div("The first dashboard for options portfolio stress testing",id='image_header2',className='main-header',
                     style=dict(color='#0b1a50',
                     fontWeight='bold',fontSize='2vh',marginTop='',marginLeft='',width='100%',paddingTop='',paddingBottom='',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))

image_header3=html.Div("Version: Beta 1.0.0",id='image_header3',className='main-header',
                     style=dict(color='#0b1a50',
                     fontWeight='bold',fontSize='1.8vh',marginTop='',marginLeft='',width='100%',paddingTop='',paddingBottom='',
                     display= 'flex', alignItems= 'left', justifyContent= 'left'))

image_header4=html.Div("Devs: Gianluca Baglini, Davide Alcala",id='image_header4',className='main-header',
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

        dbc.Col([image_header1,html.Br(),image_header2,html.Br(), main_img, html.Br(), html.Br(),image_header3 ,html.Br(),image_header4
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
        dff = pd.DataFrame(portfolio_created)
        return Portfolio_Stats.get_stats_layout(dff)
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
            dcc.Tab(label='Intraday Stress Test', value='Intraday Stress Test', style=tab_style,
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
            dcc.Tab(label='Intraday Stress Test', value='Intraday Stress Test', style=tab_style,
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
            dcc.Tab(label='Intraday Stress Test', value='Intraday Stress Test', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Scenario Analysis', value='Scenario Analysis', style=tab_style,
                    selected_style=tab_selected_style)
        ], style=tabs_styles)

        return (Portfolio_Stats.prepare_stats_layout(tabs),'pressed')

    elif selected_tab=='Intraday Stress Test':
        return ( html.Div('{} Content'.format(selected_tab),style=dict(textAlign='center')) ,'')

    elif selected_tab=='Scenario Analysis':
        return ( html.Div('{} Content'.format(selected_tab),style=dict(textAlign='center')),'')


@app.callback(Output('graph2','figure'),
              [Input("tickers_dropdown", "value") ,Input('expirations_dropdown','value')],
              [State('portfolio_created','data'),State('df_proc','data')]
              )
def update_line_chart(stock,expiration,portfolio_created,df_proc):
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
        title='Time Series Chart', xaxis_title='Date',
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
    if quantity_input == None:
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
        print('2')
        print(dff)
        print(type(dff))

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
@app.callback([Output('portfolio_msg','children'),Output('portfolio_created','data')],
              [Input('create-portfolio-button','n_clicks'),Input('select-ticker','value')],
              State('portifolio_in_progress','data'),
              prevent_initial_call=True)
def create_portfolio(clicks,ticker_changed,portfolio_data):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if input_id=='select-ticker':
        return ('',dash.no_update)

    if clicks > 0:
        df=pd.DataFrame(portfolio_data)
        df.to_csv('por.csv',index=False)
        return ( html.Div([
                'Portfolio Created Successfully',
            ],style=dict(fontSize='1.7vh',fontWeight='bold',color='green',textAlign='center')) , portfolio_data)
    else:
        raise PreventUpdate

@app.callback([Output('display-option-chain', 'children'),Output('options_exception','children')],
               Input('get-chain-in', 'n_clicks'),

              [State('store-options-exch', 'data'),State('opt-type', 'value'),
               State('exchanges-out', 'value'),State('select-ticker', 'value')],
               prevent_initial_call=True)
def get_option_chain(n_clicks ,dict_exchange , right, exchange,ticker ):
    if n_clicks == 0 or dict_exchange==None or right==None or exchange==None or ticker==None :
        return (dash.no_update,html.Div([
                'Please fill the rest of the dropdowns',
            ],style=dict(fontSize='1.6vh',fontWeight='bold',color='red',textAlign='center')))
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


        return ( dash_table.DataTable(
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
            page_size=10,

        style_cell=dict(textAlign='center', border='1px solid #0b1a50'
                        , backgroundColor='white', color='black', fontSize='1.6vh', fontWeight=''),
        style_header=dict(backgroundColor='#0b1a50', color='white',
                          fontWeight='bold', border='1px solid #d6d6d6', fontSize='1.6vh'),
        style_table={'overflowX': 'auto', 'width': '100%', 'min-width': '100%','border':'1px solid #0b1a50'}
        ) ,'')
if __name__ == '__main__':
    app.run_server(host='localhost',port=8050,debug=True,dev_tools_silence_routes_logging=True)




