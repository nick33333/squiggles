from dash import Dash, dcc, html, Input, Output, callback
from tslearn.clustering import TimeSeriesKMeans
import plotly.express as px
import ipywidgets
import pandas as pd
import pickle
import sys
import os
import copy
# Import weird stuff
sys.path.insert(1, 'MSMC_clustering/')
from MSMC_clustering import *
from MSMC_plotting import *
from plotly_curve_heatmap import *


def load_model(taxanomic_class='bird',
               models_path='models/100_pts/',
               model_name='curves_80_20_split_birds_100pts_rtp.pkl',
               data_path='data/',
               dataset_name='curves_80_20_split_birds/train_data/',
               generation_lengths_path='generation_lengths/',
               
               **Msmc_clustering_kwargs):
    ### Loading in Model data ###
    dataset_path = data_path + dataset_name
    generation_lengths_path = data_path + generation_lengths_path
    model_save_path = models_path + model_name
    # time_window=False
    time_window = [1.17E4, 2.58E6]
    mu_dict = {'bird': 1.4e-9,
               'mammal': 2.2e-9}
    m_obj = Msmc_clustering(directory=dataset_path,
                            mu=mu_dict[taxanomic_class],
                            generation_time_path=generation_lengths_path,
                            manual_cluster_count=7,
                            algo='kmeans',
                            data_file_descriptor='.txt',
                            omit_front_prior=5,
                            omit_back_prior=5,
                            time_window=time_window,
                            time_field='left_time_boundary',
                            value_field='lambda',
                            interpolation_pts=100,
                            use_interpolation=True,
                            use_friendly_note=True,
                            use_real_time_and_c_rate_transform=True,
                            use_value_normalization=True,
                            use_time_log10_scaling=True,
                            use_plotting_on_log10_scale=False,
                            sep='\t')

    model_name = 'curves_80_20_split_birds_100pts_rtp.pkl'
    if model_name in os.listdir(models_path):
        print("Loading cluster :-)")
        # km_file = open(model_save_path, "rb")
        km = TimeSeriesKMeans.from_pickle(models_path + model_name)
    else:
        print("Gotta cluster :-(")
        m_obj.cluster_curves(plot_everything=False)
        km = m_obj.km
        m_obj.km.to_pickle(model_save_path)
    return m_obj, km

### Creating fig ###


model_pts2names = {p.split('_')[0] : os.listdir(f"models/{p}") for p in os.listdir("models")}
dataset_names = [f for f in os.listdir("data/") if "curve" in f and '.zip' not in f]
# model_names = [f[:-4] for f in os.listdir("models/") if '.pkl' in f]

# Hardcoded settings for demo
cluster_range = list(range(2,11))
K = 7
# Create model fig hardcode
k=7 

# Subplot settings hardcoded
Msmc_clustering, km = load_model()
print(type(km), km)

rows = given_col_find_row(k=k, cols=2)
fig = make_subplots(rows=rows, cols=2,
                    subplot_titles=([f"Cluster {label}" for label in range(1, k+1)]))
label2series_names, name2trace_index = compute_label2series_names_and_name2trace_index(Msmc_clustering=Msmc_clustering,
                                                                                       km=km)
label2dist_matrix, series_name2label = compute_intra_cluster_dtw_dist_matrix(Msmc_clustering=Msmc_clustering,
                                                                            label2series_names=label2series_names)
for name in Msmc_clustering.name2series:
    label = series_name2label[name]
    add_curve_to_subplot(fig=fig,
                         name=name,
                         cols=2,
                         Msmc_clustering=Msmc_clustering,
                         label=label,
                         marker_color='rgba(0, 180, 255, .3)')
fig.update_layout(title_text="Subplots with Annotations")
immutable_data_copy = copy.deepcopy(fig.data)  # Should only be assigned once (after fig is first init'd)


# Cluster heatmap hardcoded on 1 cluster
# f = go.FigureWidget()
# f.layout.hovermode = 'closest'
# f.layout.hoverdistance = -1 #ensures no "gaps" for selecting sparse data
# default_linewidth = 2
# default_color = 'rgba(0, 180, 255, .8)'
# highlighted_linewidth_delta = 2

# selected_cluster = 3
# selected_series_names = label2series_names[selected_cluster]
# selected_dist_matrix = label2dist_matrix[selected_cluster]

# for name in selected_series_names:
#     series = Msmc_clustering.name2series[name]
#     trace = go.Scatter(x=series[Msmc_clustering.time_field],
#                        y=series[Msmc_clustering.value_field],
#                        line={'color':default_color,
#                              'width': default_linewidth})
#     f.add_trace(trace)


# def update_trace(trace, points, selector):
#     # this list stores the points which were clicked on
#     # in all but one trace they are empty
#     if len(points.point_inds) == 0:
#         return
#     for i,_ in enumerate(f.data):
#         f.data[i]['line']['width'] = default_linewidth + highlighted_linewidth_delta * (i == points.trace_index)
#         f.data[i]['line']['color'] = 'rgba(0,0,0,1)'
# for i in range( len(f.data) ):
#     f.data[i].on_click(update_trace)
# End hardcoding




### Dashboard ###
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

app = Dash(__name__)
colors = {
    'background': '#0E1012',
    'text': '#FFFFFF'
}
colors = {
    'background': '#0E1012',
    'text': '#000000'
}
# print(df)
app.layout = html.Div([
    html.H1(
        children='''MSMC Clustering Tool''',
        style={
            'textAlign': 'center',
            'color': colors['text'],
        }
    ),
    
    html.Div([
        "Number of clusters",
        dcc.Dropdown(cluster_range,
                     7,
                     id='num-clusters'),
        
        "Points to interpolate data to",
        dcc.RadioItems(sorted(list(model_pts2names.keys()), key=lambda x:int(x)),
                       '100',
                       inline=True,
                       id='pts-radio'),
        
        "Available models",
        dcc.Dropdown(id='models-dropdown'),  # Can use this for model selection
        
        "Select dataset",
        dcc.Dropdown(dataset_names,
                     'curves_80_20_split_birds_train',
                     id='selected-dataset'),  # Can use this for data selection
        
        "Number of columns to plot",
        dcc.Dropdown(id='num-columns'),

        html.Br(),
        "Fig",
        html.Div(
            html.Div(
                dcc.Graph(figure=fig,
                          style={'width': '100%',
                                  'height': '100%'
                                  }
                        ),
                style={
                    "width": "100%",
                    'height': '100%'
                }
                ),
            style={
                    "width": "60%",
                    "height": "800px",
                    "display": "inline-block",
                    "border": "3px #5c5c5c solid",
                    "padding-top": "5px",
                    "padding-left": "1px",
                    "overflow": "hidden"
                }
        ),
        html.Div(
            html.Div([
                dcc.Dropdown(id='curve-clusters-dropdown'),
                # dcc.Graph(figure=f,
                #           style={'width': '100%',
                #                   'height': '100%'},
                #           id='selected-curve-cluster'
                #         ),
                dcc.Graph(style={'width': '100%',
                                 'height': '80%'},
                          id='selected-curve-cluster')
                ],
                style={
                    "width": "100%",
                    'height': '100%'
                }
                ),
            style={
                    "width": "30%",
                    "height": "800px",
                    "display": "inline-block",
                    "border": "3px #5c5c5c solid",
                    "padding-top": "5px",
                    "padding-left": "1px",
                    "overflow": "hidden"
                }
        )

        

    ]),
    

])


@callback(
    Output('models-dropdown', 'options'),
    Input('pts-radio', 'value'))
def set_models_options(selected_pts_number):
    return [{'label': i, 'value': i} for i in model_pts2names.get(selected_pts_number)]

@callback(
    Output('num-columns', 'options'),
    Input('num-clusters', 'value'))
def set_columns_options(selected_cluster_number):
    return list(range(min(cluster_range), selected_cluster_number+1))

# @callback(
#     Output('main-fig', 'figure'),
#     Input('selected-dataset', 'value'))
# def update_main_figure(cols):
#     # Hardcoded for demo

@callback(
    Output('curve-clusters-dropdown', 'options'),
    Input('num-clusters', 'value'))
def set_curve_clusters_options(selected_cluster_number):
    return list(range(1, selected_cluster_number+1))

@callback(
    Output('selected-curve-cluster', 'figure'),
    Input('curve-clusters-dropdown', 'value'))
def update_selected_cluster_fig(selected_cluster):
    print('selected_cluster',selected_cluster)
    f = go.FigureWidget()
    f.layout.hovermode = 'closest'
    f.layout.hoverdistance = -1 #ensures no "gaps" for selecting sparse data
    default_linewidth = 2
    default_color = 'rgba(0, 180, 255, .8)'
    highlighted_linewidth_delta = 2
    selected_series_names = label2series_names[selected_cluster-1]
    selected_dist_matrix = label2dist_matrix[selected_cluster-1]

    for name in selected_series_names:
        series = Msmc_clustering.name2series[name]
        trace = go.Scatter(x=series[Msmc_clustering.time_field],
                        y=series[Msmc_clustering.value_field],
                        line={'color':default_color,
                                'width': default_linewidth})
        f.add_trace(trace)
    def update_trace(trace, points, selector):
    # this list stores the points which were clicked on
    # in all but one trace they are empty
        if len(points.point_inds) == 0:
            return
        for i,_ in enumerate(f.data):
            f.data[i]['line']['width'] = default_linewidth + highlighted_linewidth_delta * (i == points.trace_index)
            f.data[i]['line']['color'] = 'rgba(0,0,0,1)'
    for i in range( len(f.data) ):
        f.data[i].on_click(update_trace)
    return f


if __name__ == '__main__':
    app.run(debug=True)