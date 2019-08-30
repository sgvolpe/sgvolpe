import collections, json, os, plotly, time
import dash
import dash_auth
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_table_experiments as dt
import dash_table
#import dash_bootstrap_components as dbc
from flask import request

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import datetime

# # # # # # # # # # #
# App Configuration #


USERNAME_PASSWORD_PAIRS = [
    ['guest', 'guest'],['admin', 'admin']
]




#### AUX FUNCTIONS ###

def generate_table(dataframe, max_rows=100):
    return html.Table(
        children=
            # Header
            [html.Tr([html.Th(dataframe.index.name)]+[html.Th(col) for col in dataframe.columns])] +
            # Body
            [html.Tr([html.Td(dataframe.index[i])] + [
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))],
        className='table'
    )

def catch_figure(func):
    def wrapper(*args, **kwargs):
        if DEBUG: print (func)
        if DEBUG: l = open('log.txt','a')
        try:
            if DEBUG: a = datetime.datetime.now()
            ret = func(*args, **kwargs)

            if DEBUG:
                b = datetime.datetime.now()
                c = b - a
                l.write(str(c.total_seconds() * 1000.0)+','+str(func)+'\n')
                l.close()
            return ret
        except Exception as e:
            print ('ERROR'*20)
            print (func)
            print (str(e))
            print ('ERROR'*20)
            return go.Figure([])
    return wrapper

def decode_label(label_id):
    decode_dict = {'price_weight':'Price Weight', 'carrier_weight':'Carrier Weight','travel_time_weight':'Travel Time Weight',
        'search_date': 'Search Date','ap_los':'Advanced Purchase - Length of Stay','ond': 'Origin & Destination', 'elc': 'Enhanced Long Connect',
        'add_ns': 'Additional Non-stops', 'mt':'Multi-ticket', 'radius_search': 'Radius Search', 'ap': 'Advanced Purchases', 'los': 'Length-of-Stay', 'pcc':'PCC',
    }
    try:
        return decode_dict[label_id]
    except:
        try:
            return label_id.replace('_',' ').title()
        except:
            return str(label_id)

def generate_controls():
    'Based on Configuration Files determines and creates the type and possible values of the Input Controls of the Dashboard'
    # Basic Children
    controls_list = [
        html.H3('Controls'),
        html.Hr(),
        #html.Label('Base: ', className= 'col-2'),
        html.P( dcc.Dropdown(
                options=[ {'label': df, 'value': df} for df in dataframes],
                value= 'Base.csv',
                className= 'col-10',
                id='base'
            )
            , style={'display': 'none'}
        ),
    ]

    with open('gui_config.txt') as json_file:
        configuration_dict = json.load( json_file)

    controls_dict = configuration_dict['controls']

    # Add Custom Controls
    control_inputs =  [Input('base','value')]
    for el in controls_dict:

        control_inputs.append( Input(el['id'], 'value') ) # For later callback

        L = html.Label(decode_label( el['id']  ) )  # Control Label
        values = [ str(val) for val in config_df[ el['id'] ].unique() ] # Control Values

        if el['type'] == 'slider':
            C = dcc.Slider()
            C.id = el['id']
            C.min = el['min']
            C.max = el['max']
            C.step = el['step']
            C.value = C.min
            C.marks={i: i for i in range(C.min,C.max,C.step)}
        elif el['type'] == 'dropdown':
            C = dcc.Dropdown(
                    options=[ {'label': val, 'value': val} for val in values],
                    value = values[0],
                    className = 'col-10',
                    id=el['id']
            )
        D = html.Div(className='row', style={'margin-top':'25px'},
            children=[
                html.Div(children=[L], className='col-4'),
                html.Div(children=[C], className='col-8')
            ]
        )
        if el['guest_hidden'] == 'True' :  D.style ={'display': 'none'}
        controls_list.append(D)
        #controls_list.append(html.P(C))
    return controls_list, control_inputs


#########



app = dash.Dash(__name__) #external_stylesheets=external_stylesheets)
app.title = 'BFM Interactive Plotter '
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server


DEBUG = False
MASTER_DF = pd.DataFrame()
BASE_DF = pd.DataFrame()
NEW_DF = pd.DataFrame()
BASE_NAME = ''
NEW_NAME = ''

dataframes = os.listdir('dataframes')
config_df = pd.read_csv('configuration.csv')

inputs = {}
config_df = pd.read_csv('configuration.csv')
colors = {
    'background': '#f4f4f4f4',
    'panel': '#ffffff',
    'primary': '#E50000',
    'chart-title':"#767676",
}
COLORS = {'Alternate': '#E50000', 'Baseline': '#3399CC', 'Common':'#31B98E'}
OPACITY = 1
TEMPLATE = 'seaborn'# 'plotly_dark'

CHART_FONT=font=dict(family="inherit", size=20, color=colors['chart-title'])


CURR = config_df['currency'][0]
origin = config_df['origin'][0]
destination = config_df['destination'][0]
ap = config_df['ap'][0]
los = config_df['los'][0]


CONTROLS_LIST, CONTROL_INPUTS = generate_controls()

OUTPUT = [
            dcc.Loading (type="circle",
                children= [
                    html.Div(className='col-12',
                        children=[ dcc.Graph(id=comp_id, animate=True)]
                    ),
                    html.Div(id='', className='col-12', style={'height':'10px', 'background-color':colors['background']})
                ]
            )
            for comp_id in ['live-graph', 'carriers-graph2']
]





# # # # # # # # # # # # # # # # # # # #
# Layout

app.layout = html.Div(
        id='body-div', className= '',
        style={'background-color': colors['background']},
        children = [
            dcc.ConfirmDialog(
                id='help-dialog',
                message='Help. Contact your sales team.',
                displayed=False,
            ),

            html.Nav(className='row header navbar-expand-lg navbar-light test',
                children=[
                    html.Div(className='col-6',children=[
                        html.H1(children=[
                            html.A([html.Img(src='/assets/sabre-logo-slab.png')] , href='https://developer.sabre.com/home'),
                            'BFM Interactive Plotter'
                        ]),
                    ]),
                    html.Div(className='col-2 collapse navbar-collapse', children=html.H6(id='application-rt', style={'color': 'black'}))
                ]
            ),
            html.Div( id='content', className='container-fluid', style={'background-color':colors['background']},
                children=[
                    html.Div(id='central', className= 'row justify-content-md-center',
                        children=[
                            html.Div(id='controls', className= 'col-3 mainpanel',
                                children= [
                                    html.Div(children=CONTROLS_LIST),
                                ]
                            ),
                            html.Div(id='plots', className='col-8 mainpanel',
                                children=[
                                    html.H3('Plots'),
                                    html.Hr(),
                                    html.Div(className= 'row',
                                        children=OUTPUT
                                    )
                                ]
                            ),
                        ]
                    ), # central

                    html.Div(id='new_name', style={'display': 'none'}, children='' ),          # auxiliary children=dataframes[0]
                    html.Div(id='base_name', style={'display': 'none'}, children=['asdasd']),        # auxiliary children=dataframes[0]
                    html.Div(id='AUX', style={'display': 'none'}, children=['asdasd']),        # auxiliary children=dataframes[0]
                ]
            ), # content
            html.Div(id='', className='col-12', style={'height':'50px'}
            ),
            html.Div(id='', className='col-12 footer',
                children=[ html.A(['@ Online Technical Sales EMEA'],href='mailto:santiago.gonzalez@sabre.com)')]
            ),
        ]) # ALL # 'body-div


# Figures & Functions

@app.callback([Output('base_name', 'children'),Output('new_name', 'children')] , CONTROL_INPUTS,) #FullTable: Output('datatablediv', 'children') #[  ,
@catch_figure
def update_dfs(*p):#price_weight,travel_time_weight,carrier_weight,ond,ap_los,elc,add_ns,mt):
    ''' Updates the global dataframes, receives the dataframes names, and if change updates the names too '''
    if DEBUG: print ('UPDATE DFS TRIGGERED')
    base_name = p[0]
    new_name = p[1]

    params = [str(x) for x in p[1:] ] # [ond,ap_los,price_weight,travel_time_weight,carrier_weight,elc,add_ns,mt] ]
    if DEBUG: print (params)
    new_name = '_'.join(params) + '.csv'

    global BASE_NAME
    global NEW_NAME
    global MASTER_DF
    global BASE_DF
    global NEW_DF

    if DEBUG:
        print ('Base Name: {} -> {} '.format(BASE_NAME, base_name ))
        print ('New Name: {} - > {} '.format(NEW_NAME, new_name))
    try:
        if BASE_NAME != base_name:
            if DEBUG: print ('    BASE UPDATED')
            BASE_DF, BASE_NAME = pd.read_csv('dataframes/{}'.format(base_name)), base_name
        if NEW_NAME != new_name:
            if DEBUG: print ('    NEW UPDATED')
            NEW_DF, NEW_NAME   = pd.read_csv('dataframes/{}'.format(new_name)), new_name
        BASE_DF['provider'], NEW_DF['provider'] = 'Baseline', 'Alternate'
        MASTER_DF = pd.concat([BASE_DF, NEW_DF])                                    # combine dfs
        if DEBUG: MASTER_DF.to_csv('master_df-pre.csv')
        MASTER_DF['duplicate'] = MASTER_DF.duplicated(subset=['itinerary','price'], keep=False) # mark duplicates
        MASTER_DF = MASTER_DF.drop_duplicates(subset=['itinerary','price'], keep='first') # drop duplicates
        def f(x):
            if x['duplicate']: return 'Common'
            else: return x['provider']
        MASTER_DF['provider'] = MASTER_DF.apply(f, axis=1)



        if DEBUG: MASTER_DF.to_csv('master_df.csv')
        return new_name, #base_name,


    except Exception as e:
        print (str(e))
        return  'ERROR' #FullTable: , dash_table.DataTable() #base_name,


@app.callback( [ Output('application-rt','children'), Output('live-graph', 'figure'), Output('carriers-graph2', 'figure')],
   [Input('base_name', 'children'), Input('new_name', 'children')])
def update_output(base_name, new_name):
    t0 = datetime.datetime.now()
    price_time_graph = update_price_vs_traveltime(base_name, new_name)
    carriers_graph = update_carriers_bar_graph(base_name, new_name)

    t1 = datetime.datetime.now()
    app_stats = str((t1 - t0).total_seconds() * 1000.0) + 'ms.'


    return app_stats, price_time_graph, carriers_graph


#@catch_figure
def update_price_vs_traveltime(base_name, new_name ):
    if new_name == 'ERROR':
        return go.Figure(data=[],layout={})

    master_df = MASTER_DF
    data = [
        dict(
            type = 'scatter',
            mode = 'markers',
            x = dff['travel_time'],
            y = dff['price'],
            text = ['Itinerary: {}'.format(a) for a in dff['itinerary'] ],
            name = provider,
            opacity=OPACITY,
            marker = dict(size = 15, sizemode = 'area', color=COLORS[provider],line = dict(color = 'rgb(255, 255, 255)',width = 2) ),
            hovertemplate = "<b>%{text}</b><br><br>" + CURR + " %{y:,.0f} | %{x} minutes",

        ) for dff, provider in [(master_df[master_df.provider == provider], provider) for provider in master_df.provider.unique()]
    ]

    layout = {
        'title': go.layout.Title( text="Options Distribution" , font=CHART_FONT),
        'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="Option Total Travel Time (mins.)", font=CHART_FONT) ),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="Option Total Price ( {} )".format(CURR), font=CHART_FONT) ),
        'template': TEMPLATE,
    }

    return go.Figure(data= data, layout= layout)


#@catch_figure #,AUX
def update_carriers_bar_graph(base_name, new_name):
    if new_name == 'ERROR':
        return go.Figure(data=[],layout={})
    master_df = MASTER_DF
    data = master_df[['provider','airlines']].groupby(['airlines','provider'])['provider'].count().unstack() #TODO: sort Overlapping, Old, new

    trace=[
        go.Bar(name=provider, x=data.index, y=data[provider], marker = dict(color=COLORS[provider], opacity=OPACITY))
        for provider in data.columns
    ]

    layout = {
        'title': go.layout.Title( text="Airline Distribution per Configuration" , font=CHART_FONT),
        'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="Marketing Airlines Combination (Alphabetically Sorted)", font=CHART_FONT) ),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="Count of Itineraries", font=CHART_FONT)),
        'template': TEMPLATE,
        'barmode':'stack',
        'legend_orientation': 'h'
        #'showlegend':True
    }
    return go.Figure( data=trace,layout=layout)




if __name__ == '__main__':
    app.run_server()
