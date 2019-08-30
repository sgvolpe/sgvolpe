import datetime, collections, functools, json, os, plotly, time
import dash
import dash_auth
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_table_experiments as dt
import dash_table
import dash_bootstrap_components as dbc
from flask import request
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

DEBUG = True

if DEBUG: print ('STARTING SCRIPT')

# # # # # # # # # # #
# App Configuration #


USERNAME_PASSWORD_PAIRS = [
    ['guest', 'guest'],['admin', 'admin']
]



#### AUX FUNCTIONS ###




def generate_table2(dataframe, max_rows=100):
    return html.Table(
        children=
            # Header
            [html.Tr(
            #[html.Th(dataframe.index.name)] +
            [html.Th(col) for col in dataframe.columns])] +
            # Body
            [html.Tr(

            #[html.Td(dataframe.index[i])] +
            [
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))],
        className='table'
    )

def generate_table3(dataframe, max_rows=350):
    table_header = [ html.Thead(html.Tr( [html.Th(col) for col in dataframe.columns])) ] #[html.Th(dataframe.index.name, scope="col")] +
    rows = [ html.Tr(children=[html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]) for i in range(min(len(dataframe), max_rows)) ] # html.Th(dataframe.index[i], scope="row")] +
    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_header + table_body, bordered=True,
    dark=True,
    hover=True,
    responsive=True,
    striped=True,)
    return table



def generate_table(df):
    return dash_table.DataTable(
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in df.columns #+ [df.index.name]
            # omit the id column
            #if i != 'id'
        ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        #row_selectable='multi',
        #row_deletable=True,
        selected_rows=[],
        page_action='native',
        page_current= 0,
        page_size= 25,
    ),


def printme(func):
    def wrapper(*args, **kwargs):
        a = datetime.datetime.now()
        ret = func(*args, **kwargs)
        b = datetime.datetime.now()
        c = b - a
        print (f'{func} | {str(c.total_seconds() * 1000.0)}' )
        return ret
    return wrapper

def catchfigure(func):
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
        'travel_time_common': 'Quickest Common','travel_time_alternate': 'Quickest Alternate','travel_time_baseline': 'Quickest Baseline'
    }
    try:
        return decode_dict[label_id]
    except:
        try:
            return label_id.replace('_',' ').title()
        except:
            return str(label_id)


def generate_controls(demo_case='Default'):
    'Based on Configuration Files determines and creates the type and possible values of the Input Controls of the Dashboard'

    # Basic Children
    controls_list = [
        html.H3('Controls'),
        html.Hr(),
        #html.Label('Base: ', className= 'col-2'),
        #dataframes =
        html.P( dcc.Dropdown(
                options= [ {'label': df, 'value': df} for df in os.listdir(os.path.join('demo_cases',demo_case,'dataframes'))],
                value= 'Base.csv',
                className= 'col-10',
                id='base'
            )
            , style={'display': 'none'}
        ),
    ]

    configuration_df = pd.read_csv(os.path.join('demo_cases',demo_case,'configuration.csv'))
    gui_conf_path = os.path.join('demo_cases',demo_case,'gui_config.txt')

    json_file = open(gui_conf_path)
    configuration_dict = json.load( json_file)
    controls_dict = configuration_dict['controls']


    # Add Custom Controls
    control_inputs =  [Input('base','value')]
    for el in controls_dict:
        control_inputs.append( Input(el['id'], 'value') ) # For later callback
        L = html.Label(decode_label( el['id']  ) )  # Control Label
        values = [ str(val) for val in configuration_df[ el['id'] ].unique() ] # Control Values

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
        if str(el['guest_hidden']) == 'True' :
            D.style ={'display': 'none'}
        controls_list.append(D)

    return controls_list, control_inputs


def generate_output(demo_case='Default'):
    print ('GENERATIONG OUTPUTS')
    gui_conf_path = os.path.join('demo_cases',demo_case,'gui_config.txt')
    json_file = open(gui_conf_path)
    configuration_dict = json.load( json_file)
    OUTPUT_DICT = configuration_dict['plots']
    #print (OUTPUT_DICT)
    OUTPUT = [
        #dbc.Card(
        html.Div(className=OUTPUT_DICT[comp_id]['className'], children=
                [
                    dcc.Graph(id=comp_id, animate=False, style=OUTPUT_DICT[comp_id]['style'])
                ]

        )
        for comp_id in OUTPUT_DICT
    ]
    OUTPUT_LIST = [Output('application-rt','children')] + [ Output(out_id, 'figure') for out_id in OUTPUT_DICT.keys()  ] # For the Callbacks of the update Output
    return OUTPUT, OUTPUT_LIST

def it_distance(it1, it2, sep='-'):
    '''Receives 2 strings representing 2 schedules calculate the ratio of
        Flights in Common / Maximum Number of Flights ( based on the longer sched)
    '''
    f_count = max( len(it1.replace(sep*2,sep).split(sep)), len(it2.replace(sep*2,sep).split(sep)) )
    legs1, legs2 = it1.split(sep*2),it2.split(sep*2)
    in_common = 0
    for i in range(len(legs1)):
        l = zip(legs1[i].split(sep),legs2[i].split(sep))
        in_common += sum([a==b for (a,b) in l])
    return 1.00 - (in_common*1.00/f_count)


def get_matrix_distance(it1, it2, sep='-'):
    '''Receives 2 lists of strings of length m & n, ie. each of them representing the
        full list of schedules returned by a shopping query.
        Returns a matrix m*n, each cell x,y represents the distance between itinerary x and y
    '''
    #it1, it2 = it1.replace(sep*2,sep).split(sep), it2.replace(sep*2,sep).split(sep)
    m, n =len(it1), len(it2)
    M = [[0 for w in range(n)] for w in range(m)]
    for i in range(m):
        for j in range(n):
            M[i][j] = it_distance(it1[i],it2[j],sep=sep)
    return M


def itineraries_distance(it1, it2=None, sep='-'):
    'For 2 different set of itineraries'
    #if it2 is None: it2 = [i for i in it1]
    M = get_matrix_distance(list(it1), list(it1),sep=sep)
    result = 0
    for row_i in range(1,len(M)):
        if it2 is None: result += min(M[row_i][:row_i:])
        else: result += min(M[row_i])
    return result


@printme
def update_kpis(demo_case='Default'):
    df_path = os.path.join('demo_cases', demo_case, 'dataframes')

    if 'kpi_df.csv' not in os.listdir(os.path.curdir):
        D = []
        for df_name in os.listdir(df_path):
            print (df_name)
            d = {}
            df = pd.read_csv(os.path.join(df_path, df_name))
            d['name'] = df_name
            d['price_min'] = df['price'].min()
            d['price_mean'] = df['price'].mean()
            d['price_std'] = df['price'].std()
            d['price_25%'] = df['price'].quantile(.25)
            d['price_50%'] = df['price'].quantile(.50)
            d['price_75%'] = df['price'].quantile(.75)
            d['travel_time_min'] = df['travel_time'].min()
            d['travel_time_mean'] = df['travel_time'].mean()
            d['travel_time_std'] = df['travel_time'].std()
            d['travel_time_25%'] = df['travel_time'].quantile(.25)
            d['travel_time_50%'] = df['travel_time'].quantile(.50)
            d['travel_time_75%'] = df['travel_time'].quantile(.75)
            d['options_count'] = df.shape[0]
            d['cxr_count'] = len(df['airlines'].unique())
            d['payload_size'] = os.path.getsize(os.path.join(df_path, df_name))
            d['schedule_distance'] = itineraries_distance(it1=df['itinerary'])
            D.append(d)
        kpi_df = pd.DataFrame(D)
        kpi_df.to_csv('kpi_df.csv')
    else: kpi_df = pd.read_csv('kpi_df.csv')

    fig = go.Figure()
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['options_count'],name='options_count'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['price_min'],name='Cheapest'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['travel_time_min'],name='Quickest'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['cxr_count'],name='cxr_count'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['payload_size'],name='payload_size'))

    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['price_25%'],name='price_25%'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['price_50%'],name='price_50%'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['price_75%'],name='price_75%'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['travel_time_25%'],name='travel_time_25%'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['travel_time_50%'],name='travel_time_50%'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['travel_time_75%'],name='travel_time_75%'))
    fig.add_trace(go.Scatter(mode='lines',x = kpi_df.index,y = kpi_df['schedule_distance'],name='schedule_distance'))

    fig.layout = {
        'title': go.layout.Title( text="KPI Metric per Configuration" , font=CHART_FONT),
        'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="Configuration ID", font=CHART_FONT) ),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="", font=CHART_FONT)),
        'template': TEMPLATE,
        'legend': dict(x=1.1, y=1.2,  bordercolor="Black",borderwidth=2),
    }


    return fig


###############



######### Aesthetic
colors = {
    'background': '#f4f4f4f4',
    'panel': '#ffffff',
    'primary': '#E50000',
    'chart-title':"#767676",
}
COLORS = {'Alternate': '#E50000', 'Baseline': '#3399CC', 'Common':'#31B98E'}
OPACITY = 1
TEMPLATE = 'seaborn' # 'plotly+presentation'## 'plotly_dark'
CHART_FONT = font=dict(family="inherit", size=15, color=colors['chart-title'])

app = dash.Dash(__name__) #external_stylesheets=external_stylesheets)
app.title = 'BFM Interactive Plotter '
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server


demo_cases = os.listdir('demo_cases')
config_df = pd.read_csv(os.path.join('demo_cases', 'Default', 'configuration.csv'))
example_file = os.listdir(os.path.join('demo_cases', 'Default', 'dataframes'))[0]
if example_file == 'Base.csv': example_file = os.listdir(os.path.join('demo_cases', 'Default', 'dataframes'))[1]
search_date, origin,destination,ap,los = example_file.split('_')[:5]


# Generation & Display of Inputs
CONTROLS_LIST, CONTROL_INPUTS = generate_controls()
OUTPUTS, OUTPUT_LIST = generate_output()
DEMO_CONFIGURATIONS = [{'label': demo, 'value': demo} for demo in os.listdir('demo_cases') ]


# # # # # # # # # # # # # # # # # # # #
# Layout

app.layout = html.Div(
        id='body-div', className= '',
        style={'background-color': colors['background']},
        children = [
            dbc.Nav(
                className="row header navbar navbar-expand-lg navbar-dark bg-dark flex-md-nowrap p-0 shadow",
                children=[
                    html.A(className="navbar-brand col-sm-3 col-md-2 mr-0", children=[html.Img(src='/assets/sabre-logo-slab.png')] , href='https://developer.sabre.com/home'),

                    html.Div(className="col-md-6", children=['BFM Interactive Plotter']),
                    html.Div(className="col-md-3", children=html.H3(id='application-rt', style={'color': 'black'})),
                    html.Div(className="col-md-1", children=['?']),
                    html.P( children=[
                            dcc.Dropdown(
                                options=DEMO_CONFIGURATIONS,
                                className= 'col-10',
                                value='Default',
                                id='demo_case'
                            ),
                            html.Button(id='submit-button', n_clicks=0, children='Submit')
                        ],
                        style={'display':'none'}
                    )

                ],

            ),html.Div(className="container-fluid",
                children=[
                    html.Div(className="row", children=[
                        html.Div(className="col-md-2 card", children=[
                            html.Div(id='controls', className= '',
                                children= [
                                    html.Div(children=dcc.Loading (type="circle",id='dynamic_controls',color='red',
                                        children= CONTROLS_LIST)),
                                ]
                            ),
                            ]
                        ),
                        html.Div(className="col-md-10", children=[
                            dbc.Tabs(
                            children=[
                                dbc.Tab(label='Index', id='index-div', className='row',
                                    children=[
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                        [
                                                            html.H4("BFM Interactive Plotter", className="card-title"),
                                                            html.P("Use the controls on the left to change the shopping qualifers and discover how BFM response changes by exploring the Charts & Tables tabs",
                                                                className="card-text",
                                                            ),
                                                            html.P(f"The datapoints used for this demo tool were extracted on: {search_date} for {origin} to {destination} for an advanced purchase of {ap} and a length-of-stay for {los} days.",
                                                                className="card-text",
                                                            ),
                                                            html.P("Enhanced Long Connect: when true maximizes the maximum connection time to 1439 minutes per connection. By default this value is 780 minutes. The number of connections per bound is up to 3.",
                                                                className="card-text",
                                                            ),
                                                            #dbc.Button("Go somewhere", color="primary", className='btn'),
                                                        ]
                                                    )
                                            ]
                                        )
                                    ]

                                ),
                                dbc.Tab(label='Charts',id='charts-div', className='row',
                                    children=[
                                        #html.H3('Plots'),
                                        #html.Hr(),
                                        html.Div(className= 'col row', id='output_plots',
                                            children=[html.Div(id='results_table', className='col-4 table-responsive', style= {
                                			"margin-top": "1rem", "box-shadow": "2px 2px 2px lightgrey"
                                			}
                                            )] +OUTPUTS
                                        ),
                                    ]
                                ),
                                dbc.Tab(label='Tables',id='tables-div', className='container-fluid card',
                                    children=[
                                        html.Div(className='row', style={'margin-top':'25px'},
                                            children=[
                                                html.Div(id='results_table2', className='col-6 table-responsive'),
                                                html.Div(id='results_table3', className='col-6 table-responsive'),
                                            ]
                                        ),
                                    ]
                                ),
                                dbc.Tab(label='KPIs', id='kpis-div',
                                    children=[
                                        html.Div(className='col-12',children=[dcc.Graph(id='kpi-graph', animate=False, figure=update_kpis())])#

                                    ]
                                )
                            ]
                        )
                        ]),
                    ]),
                    html.Div(id='', className='col-12', style={'height':'50px'}
                    ),
                    html.Div(id='', className='col-12 footer',
                        children=[ html.A(['@ Online Technical Sales EMEA'],href='mailto:santiago.gonzalez@sabre.com)')]
                    ),
                    html.Div(className="row", children=[]),

                ]
            )
        ]
    )



# Figures & Functions

@app.callback( OUTPUT_LIST + [Output('results_table','children'), Output('results_table2','children'), Output('results_table3','children')],# + [Output('kpi-graph','figure')], #KPI does not get updated
    CONTROL_INPUTS + [ Input('submit-button', 'n_clicks'), Input('live-graph','selectedData') ], [State('demo_case','value')] )
@printme
def update_output(*input_parameters):
    # Add Sells Volume
    sells = pd.read_csv('sells/sells_LON_MIA_60.csv') #TODO:
    itin_dict = dict(sells['flight'].value_counts())
    def get_itin_sells(itin):
        itin = itin.replace('||','|').replace('|', ' ')
        if itin +' ' in itin_dict:
            return itin_dict[itin+' ']
        else:
            return 0

    n_clicks = input_parameters[-3]
    selected_data = input_parameters[-2]
    demo_case = input_parameters[-1]

    t0 = datetime.datetime.now()
    try:
        base_name = 'Base.csv'
        new_name = '_'.join([str(x) for x in input_parameters[1:-3] ] ) + '.csv'
        base_df = pd.read_csv( os.path.join('demo_cases', demo_case,'dataframes',base_name) )
        new_df = pd.read_csv( os.path.join('demo_cases', demo_case,'dataframes',new_name) )

        if selected_data is not None:
            selected_itineraries = [ opt['text'].split('<br>')[0].replace('<b>','').replace(r'</b>','') for opt in selected_data['points'] ]
            base_df = base_df[base_df['itinerary'].isin(selected_itineraries)]
            new_df = new_df[new_df['itinerary'].isin(selected_itineraries)]

        # combine dfs
        base_df['provider'], new_df['provider'] = 'Baseline', 'Alternate'
        master_df = pd.concat([base_df, new_df])
        master_df['sells'] = master_df['itinerary'].apply(get_itin_sells)
        # Identify, and Drop Duplicate Options: Same Schedule+Price
        master_df['duplicate'] = master_df.duplicated(subset=['itinerary','price'], keep=False) # mark duplicates
        master_df = master_df.drop_duplicates(subset=['itinerary','price'], keep='first') # drop duplicates
        master_df['provider'] = master_df.apply(lambda x: 'Common' if x['duplicate'] else x['provider'], axis=1)

        if DEBUG: master_df.to_csv('master_df.csv')

        #schedule_heatmap =
        table1, table2, table3 = update_results_tables(master_df, base_df=base_df, new_df=new_df)

        retorno = []
        ids_list = [i.component_id for i in OUTPUT_LIST]
        if 'live-graph' in ids_list: retorno.append(update_price_vs_traveltime(master_df, demo_case))
        if 'carriers-graph2' in ids_list: retorno.append(update_carriers_bar_graph(master_df))
        if 'overlap-graph' in ids_list: retorno.append(update_overlap_graph(master_df))
        if 'violin-graph' in ids_list:
            violing_graph1, violing_graph2 = update_violin_graph(master_df)
            retorno.append(violing_graph1)
            if 'violin-graph2' in ids_list: retorno.append(violing_graph2)
        if 'times-graph' in ids_list:
            time_graph, time_graph2 = update_time_graph(master_df,demo_case)
            retorno.append(time_graph)
            retorno.append(time_graph2)
        if 'schedule_heatmap' in ids_list:
            gui_conf_path = os.path.join('demo_cases',demo_case,'gui_config.txt')

            json_file = open(gui_conf_path)
            gui = json.load( json_file)['plots']

            if 'schedule_heatmap' in gui:
                retorno.append( heatmap(base_df, new_df) )
            else: retorno.append('not displayed')

        #kpis = 'Not avail.'#update_kpis(demo_case)
        retorno.append(table1)
        retorno.append(table2)
        retorno.append(table3)
        #retorno.append(kpis)


        t1 = datetime.datetime.now()
        app_stats = str((t1 - t0).total_seconds() * 1000.0) + 'ms.'
        retorno = [app_stats] + retorno


    except Exception as e:
        print ('ERROR ON UPDATE_OUTPUT')
        print (f'Democase {demo_case}')
        print ( str(e))
        return [go.Figure(data=[], layout={}) for i in OUTPUT_LIST + ['kpi']] + ['','']
    return retorno


@printme
def update_price_vs_traveltime(master_df, demo_case ):

    config_df = pd.read_csv(os.path.join('demo_cases', demo_case, 'configuration.csv'))
    CURR = config_df['currency'][0]

    data = [
        dict(
            type = 'scatter',
            mode = 'markers',
            x = dff['travel_time'],
            y = dff['price'],

            #text = [f'{a}' for a in dff['itinerary'] ],
            text= [f"{dff.loc[i]['itinerary']}<br>Position: <b>{i}</b> | Sells:<b>{dff.loc[i]['sells']}</b><br>{CURR}<b>{dff.loc[i]['price']}</b> | <b>{dff.loc[i]['travel_time']}</b> minutes"
            for i in dff.index],
            name = provider,
            opacity=OPACITY,
            marker = dict(size = dff['sells'].astype(int)*2 + 60, sizemode = 'area', color=COLORS[provider], line = dict(color = 'rgb(255, 255, 255)', width = 2) ),
            hovertemplate = "%{text}<br>" + CURR + " %{y:,.0f} | %{x} minutes ",

        ) for dff, provider in [ (master_df[master_df.provider == provider], provider) for provider in master_df.provider.unique() ]
    ]

    layout = {
        'title': go.layout.Title( text="Options Distribution" , font=CHART_FONT),
        'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="Option Total Travel Time (mins.)", font=CHART_FONT) ),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="Option Total Price ( {} )".format(CURR), font=CHART_FONT) ),
        'template': TEMPLATE,
    }

    return go.Figure(data= data, layout= layout)

@printme
def update_carriers_bar_graph(master_df):
    data = master_df[['provider','airlines']].groupby(['airlines','provider'])['provider'].count().unstack() #TODO: sort Overlapping, Old, new

    trace=[
        go.Bar(name=provider, x=data.index, y=data[provider], marker = dict(color=COLORS[provider], opacity=OPACITY))
        for provider in data.columns
    ]

    layout = {
        'title': go.layout.Title( text="Airline Distribution per Configuration" , font=CHART_FONT),
        'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="Marketing Airlines Combination", font=CHART_FONT) ),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="Count of Itineraries", font=CHART_FONT)),
        'template': TEMPLATE,
        'barmode':'stack',
        'legend_orientation': 'h'
        #'showlegend':True
    }
    return go.Figure( data=trace,layout=layout)

@printme
def update_overlap_graph(master_df):

    n_rows = master_df.shape[0]
    values_dict = dict(master_df['provider'].value_counts() )
    aux = {}
    for k,v in values_dict.items():  aux[k]=v*100.0/n_rows
    values_dict = aux
    data=[
        go.Bar(name=name, x=[''], y=[values_dict[name]], hovertemplate= ' %{y:,.0f}%', marker = dict(color=COLORS[name], opacity=OPACITY), textposition= 'auto')
        for name in list( values_dict.keys())
    ]
    layout = {
        'title': go.layout.Title( text="Options Overlap between Configurations" , font=CHART_FONT),
        'template': TEMPLATE, 'barmode':'stack',
        #'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="", font=CHART_FONT) ),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="%", font=CHART_FONT) ),
        'showlegend': True,
        'legend_orientation': 'h'
    }
    return go.Figure(data= data, layout= layout)

@printme
def update_violin_graph(master_df):

    #fig = plotly.tools.make_subplots(print_grid =False, rows=1, cols=1, subplot_titles=('Fare Distribution by Provider',  )) # 'Travel Time Distribution per Provider'
    #fig2 = plotly.tools.make_subplots(print_grid =False, rows=1, cols=1, subplot_titles=('Travel Time Distribution per Provider',  ))

    providers = master_df['provider'].unique()

    data=[
        go.Violin( marker = dict(color=COLORS[provider], opacity=OPACITY),
            name=provider, x=master_df[master_df['provider']==provider]['provider'], y=master_df[master_df['provider']==provider]['price']
        ) for provider in list(master_df['provider'].unique())
    ]
    data2=[
            go.Box(marker = dict(color=COLORS[provider], opacity=OPACITY),
                name=provider, x=master_df[master_df['provider']==provider]['provider'], y=master_df[master_df['provider']==provider]['travel_time']
            ) for provider in list(master_df['provider'].unique())
    ]

    return go.Figure(data= data, layout= {'legend_orientation': 'h','title': go.layout.Title( text="Price Distribution per Configuration" , font=CHART_FONT),'template': TEMPLATE}), go.Figure(data= data2, layout= {'legend_orientation': 'h','title': go.layout.Title( text="Travel Time Distribution per Configuration" , font=CHART_FONT),'template': TEMPLATE})

@printme
def update_time_graph(master_df, demo_case):

    master_df['bound_departure_time']
    config_df = pd.read_csv(os.path.join('demo_cases', demo_case, 'configuration.csv'))
    CURR = config_df['currency'][0]
    data1 = [
        dict(
            type = 'scatter',
            mode = 'markers',
            x = dff['bound_departure_time'].apply(lambda x: datetime.datetime.strptime( x.split('||')[0], '%H:%M:%S')  ),
            y = dff['bound_arrival_time'].apply(lambda x: datetime.datetime.strptime( x.split('||')[0], '%H:%M:%S')  ),
            #text = [f'{a}<br>Sells: {b}<br>Option Position: {i}<br>Price: {d} | Travel time: {c}' for a in dff['itinerary'] for b in dff['sells'] for c in dff['travel_time'] for d in  dff['price']  for i in  dff.index],
            text= [f"<b>{dff.loc[i]['itinerary'].split('||')[0]}</b>||{dff.loc[i]['itinerary'].split('||')[1]}<br>Position: <b>{i}</b> | Sells:{dff.loc[i]['sells']}<br>{CURR}{dff.loc[i]['price']} | {dff.loc[i]['travel_time']} minutes"
            for i in dff.index],
            name = provider,
            opacity=OPACITY,
            marker = dict(symbol='x',size = dff['sells'].astype(int)*2 + 60, sizemode = 'area', color=COLORS[provider],line = dict(color = 'rgb(255, 255, 255)',width = 2) ),
            hovertemplate = "%{text}<br>" + "Departing at: %{x}<br>Arriving at: %{y}",

        ) for dff, provider in [(master_df[master_df.provider == provider], provider) for provider in master_df.provider.unique()]
    ]

    layout1 = {
        'title': go.layout.Title( text="Outbound Time of Day Distribution" , font=CHART_FONT),
        'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="Bound Departure Time", font=CHART_FONT), tickformat = '%H:%M' ),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="Bound Arrival Time", font=CHART_FONT) , tickformat = '%H:%M'),
        'template': TEMPLATE,
    }
    data2 = [
        dict(
            type = 'scatter',
            mode = 'markers',
            x = dff['bound_departure_time'].apply(lambda x: datetime.datetime.strptime( x.split('||')[1], '%H:%M:%S')  ),
            y = dff['bound_arrival_time'].apply(lambda x: datetime.datetime.strptime( x.split('||')[1], '%H:%M:%S')  ),
            text= [f"<b>{dff.loc[i]['itinerary'].split('||')[0]}</b>||{dff.loc[i]['itinerary'].split('||')[1]}<br>Position: <b>{i}</b> | Sells:{dff.loc[i]['sells']}<br>{CURR}{dff.loc[i]['price']} | {dff.loc[i]['travel_time']} minutes"
            for i in dff.index],
            name = provider,
            opacity=OPACITY,
            marker = dict(symbol='x', size = dff['sells'].astype(int)*2 + 60, sizemode = 'area', color=COLORS[provider],line = dict(color = 'rgb(255, 255, 255)',width = 2) ),
            hovertemplate = "%{text}<br><br>" + " %{y:,.0f} | %{x} minutes",

        ) for dff, provider in [(master_df[master_df.provider == provider], provider) for provider in master_df.provider.unique()]
    ]

    layout2= {
            'title': go.layout.Title( text="Inbound Time of Day Distribution" , font=CHART_FONT),
            'xaxis': go.layout.XAxis(title=go.layout.xaxis.Title( text="Bound Departure Time", font=CHART_FONT), tickformat = '%H:%M' ),
            'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title( text="Bound Arrival Time", font=CHART_FONT) , tickformat = '%H:%M'),
            'template': TEMPLATE,
    }

    return go.Figure(data= data1, layout= layout1), go.Figure(data= data2, layout= layout2)

@printme
def update_results_tables(df, base_df, new_df):

    sells = pd.read_csv('sells/sells_LON_MIA_60.csv')
    itin_dict = dict(sells['flight'].value_counts())
    def get_itin_sells(itin):
            itin = itin.replace('||','|').replace('|', ' ')
            if itin +' ' in itin_dict:
                return itin_dict[itin+' ']
            else:
                return 0

    df['sells'] = df['itinerary'].apply(get_itin_sells)

    #print (df.groupby(['provider'])['price'].describe())

    #stats = dict(df.groupby(['provider'])['price'].describe())
    sells_dict = df.groupby(['provider']).sum()['sells']
    options_dict = df.groupby(['provider']).count()['itinerary']
    cheapest_dict = df.groupby(['provider']).min()['price']
    quickest_dict = df.groupby(['provider']).min()['travel_time']

    price_dict = df.groupby(['provider']).mean()['price'].round(1)
    ttime_dict = df.groupby(['provider']).mean()['travel_time'].round(1)
    airline_dict = df.groupby('provider').apply(lambda x: len(x['airlines'].unique()) )

    price_range = df.groupby(['provider'])['price'].apply(lambda x: (x.max() - x.min()) / x.min()  ).round(1)   #    price_range = df.groupby(['provider'])['price'].agg(['max','min']).diff(axis=1)
    price_std = df.groupby(['provider'])['price'].std().round(1)
    travel_time_std = df.groupby(['provider'])['travel_time'].std().round(1)


    distances = []
    for provider in df['provider'].unique():
        aux_df = df[df['provider'] == provider]
        distances.append(itineraries_distance(aux_df['itinerary'], sep='|'))


    dists = pd.DataFrame([distances], columns=df['provider'].unique(), index=['Sched Distance']).round(0)
    stats_df = pd.DataFrame( [options_dict, cheapest_dict, quickest_dict,price_dict,
        ttime_dict, airline_dict, sells_dict,price_range, price_std,travel_time_std ], index=['Number of Options','Cheapest','Quickest', 'Average Price', 'Average T Time',
        'CXR Count', 'Sells Count', 'Price Range','Price std', 'T. Time std'] )
    stats_df = stats_df.append([dists])
    stats_df = stats_df.rename_axis("Metric", axis='index').reset_index()
    stats_df = stats_df.rename(lambda x: x.split(' ')[0], axis='columns')

    min_airline_df = df.groupby(['airlines','provider']).min()[['price','travel_time']].round(0).unstack()
    min_airline_df.columns = ['_'.join(x) for x in min_airline_df.columns]

    #min_airline_df = min_airline_df.rename({ 'price_Alternate':'Cheapest Alternate','travel_time_Alternate':'Quickest Alternate','price_Baseline':'Quickest Baseline','travel_time_Baseline':'Quickest Baseline',
    #}, axis='columns')
    min_airline_df = min_airline_df.rename(decode_label, axis='columns')
    min_airline_df = min_airline_df.rename_axis("Airline", axis='index').reset_index()



    cols = ['itinerary', 'price', 'travel_time']


    return generate_table3(stats_df), generate_table3(min_airline_df), generate_table3(df[cols].round(0))

@printme
def heatmap(base_df, new_df):



    #return go.Figure(data={},layout={} )
    dist = itineraries_distance(base_df['itinerary'], new_df['itinerary'], sep='|'),
    layout = {
        'title': go.layout.Title( text=f"Schedule Distance | Total: {dist}" , font=CHART_FONT)
    }
    M = get_matrix_distance(base_df['itinerary'], new_df['itinerary'], sep='|')
    data = [go.Heatmap(z=M )]

    return go.Figure(data=data,layout=layout )


def update_kpis2(demo_case):
    df_path = os.path.join('demo_cases', demo_case, 'dataframes')

    fig = go.Figure()
    data = []
    for df_name in os.listdir(df_path):
        df = pd.read_csv(os.path.join(df_path, df_name))
        fig.add_trace(
            go.Box(
                y=df['travel_time'].astype(float),
                name=df_name
            )
        )
    #y = [0] + [kpi_df.iloc[i+1]['price_min'].astype(float) - kpi_df.iloc[i]['price_min'].astype(float) for i in range(0,kpi_df.shape[0]-1 ,1) ],
    #y = kpi_df['price_min'], #[0.0] +  [kpi_df[i+1]['price_min'] - kpi_df[i]['price_min'] for i in range(0,kpi_df.shape[0]-1,1) ],

            #base=kpi_df['price_min'],
            #marker_color='crimson',

    return fig



#@app.callback( [Output('dynamic_controls','children'),Output('output_plots','children') ], [Input('submit-button', 'n_clicks')], [State('demo_case','value')] )
def reload_layout(n_clicks,demo_case):
    return generate_controls( demo_case)[0], generate_output(demo_case)[0]




if __name__ == '__main__':
    app.run_server(dev_tools_hot_reload=False)

#    app.run_server()
