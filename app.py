from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
import pickle
from MSMC_clustering import *
from MSMC_plotting import *
from plotly_curve_heatmap import *

### Loading in Model data ###
data_path = '../data/'
data_path_train = data_path + 'curves_80_20_split_birds/train_data/'
generation_lengths_path = data_path + 'generation_lengths/'
model_save_path = '../models/'
time_window = [1.17E4, 2.58E6]
# time_window=False
mu_dict = {'bird':1.4e-9,
          'mammal':2.2e-9}
# time_series_path = '/scratch/nick/MSMC-Curve-Analysis/test_case_data/'
model_name = 'curves_80_20_split_birds_100pts_rtp.pkl' 
if model_name in os.listdir(model_save_path):
    print("Loading cluster :-)")
    m_obj = pickle.load(model_save_path + model_name)
else:
    print("Gotta cluster :-(")
    m_obj = Msmc_clustering(directory=data_path_train,
                            mu=1.4e-9,
                            generation_time_path=generation_lengths_path,
                            to_omit=[],
                            exclude_subdirs=['mammals_part_1'],
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
    m_obj.cluster_curves(plot_everything=False)
    m_obj.km.to_pickle(model_save_path + model_name)

### Creating fig ###
cols = 2
rows = given_col_find_row(k=m_obj.manual_cluster_count, cols=cols)
fig = make_subplots(rows=rows, cols=cols,
                   subplot_titles=([f"Cluster {label}" for label in range(1, k+1)]))
### Dashboard ###
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        df['year'].min(),
        df['year'].max(),
        step=None,
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        id='year-slider'
    )
])


@callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = df[df.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run(debug=True)