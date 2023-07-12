import sys
from flask import (
    Flask,
    render_template,
    request,
    make_response,
    redirect,
    session,
    url_for,
    request,
    flash
)
from flask_bootstrap import Bootstrap
import pickle
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import os
from tslearn.clustering import TimeSeriesKMeans
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import json

sys.path.insert(1, 'MSMC_clustering/')
from MSMC_clustering import Msmc_clustering
from MSMC_plotting import *
from MSMC_form_kwarg_handler import manual_modify_dict
import dash_bootstrap_components as dbc

from dash import Dash, html
'''
To use flask shell:
$ export FLASK_APP=index
$ flask shell
'''

UPLOAD_FOLDER = os.path.join('static', 'uploads')
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}


# init session data
# session['uploads'] = []
# session['form_results'] = ''
# session['fname'] = ''

# CONSTANTS
# Form inputs for Msmc_clustering kwargs which don't require user input
barred_args = ['self',
               'directory',
               'generation_time_path',
               'to_omit',
               'exclude_subdirs',
               'use_friendly_note',
               'readfile_kwargs',
               'tmp_data',
               'use_plotting_on_log10_scale']
# Lists to cast form inputs for Msmc_clustering kwargs into appropriate data
# types.
to_int_list = ['interpolation_pts',
               'manual_cluster_count',
               'omit_back_prior',
               'omit_front_prior']
to_float_list = ['mu']
to_bool_list = ['use_interpolation',
                'use_real_time_and_c_rate_transform',
                'use_time_log10_scaling',
                'use_value_normalization']
to_list_list = ['time_window']


# App

# App config
app = Flask(__name__)
# app = Dash(__name__)
bootstrap = Bootstrap(app)
app.secret_key = 'This is your secret key to utilize session in Flask'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Msmc_clustering_args = Msmc_clustering.__init__.__code__.co_varnames
Msmc_clustering_args = [i for i in Msmc_clustering_args if i not in
                        barred_args]
print(Msmc_clustering_args)
Msmc_clustering_form_defaults = {'data_file_descriptor': '.txt',
                            'mu': 1.4E-9,
                            'manual_cluster_count': 7,
                            'algo': 'kmeans',
                            'omit_front_prior': 5, 
                            'omit_back_prior': 5, 
                            'time_window': '', 
                            'time_field': 'left_time_boundary', 
                            'value_field': 'lambda', 
                            'interpolation_pts': 70, 
                            'interpolation_kind': 'linear', 
                            'use_interpolation': True, 
                            'use_real_time_and_c_rate_transform': True, 
                            'use_value_normalization': True, 
                            'use_time_log10_scaling': True}
app.config['Msmc_clustering_args'] = Msmc_clustering_args
app.config['Msmc_clustering_form_defaults'] = Msmc_clustering_form_defaults


# Helper functions
def find_upload_path(data_filename: "str") -> "str":
    '''
    Give data_filename and receive path to file. Path will
    vary depending on app.config['UPLOAD_FOLDER'] and name
    '''
    return os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename)


def clean_uploads(session):
    '''
    Given session, remove files uploaded during session.
    '''
    if session.get('uploads') is None:
        return
    elif isinstance(session.get('uploads'),
                    list) and len(session.get('uploads')) > 0:
        for data_filename in session.get('uploads'):
            path_to_file = find_upload_path(data_filename=data_filename)
            print(f"Removing {path_to_file}")
            os.remove(path_to_file)
        return


# App stuff


@app.route('/', methods=['GET', 'POST'])
def index():
    '''
    Home page gives options to upload data and choose clustering settings
    '''
    if request.method == 'POST':
        # Form handling
        session['form_results'] = request.form  # saves form data after upload
        # File handling
        # print(f"request.files {request.files}")
        if 'file(s)' not in request.files:
            flash('No file(s) given')
            return redirect(request.url)
        files = request.files.getlist('file(s)')  # retrieve list of uploaded fnames
        for file in files:
            if file:
                # print(f"filename: {file.filename}")
                data_filename = secure_filename(file.filename)
                # print(f"data_filename: {data_filename}")
                if session.get('uploads') is None:
                    session['uploads'] = [data_filename]
                else:
                    session['uploads'].append(data_filename)
                file.save(find_upload_path(data_filename=data_filename))
                session['path_to_last_upload'] = find_upload_path(data_filename=data_filename)
        return render_template('index2.html',
                               Msmc_clustering_args=Msmc_clustering_args,
                               Msmc_clustering_form_defaults=Msmc_clustering_form_defaults)
    # clean_uploads(session)
    return render_template("index.html",
                           Msmc_clustering_args=Msmc_clustering_args,
                           Msmc_clustering_form_defaults=Msmc_clustering_form_defaults)

@app.route('/show_data')
def showData():
    '''
    Instead of session.get in data_file_path, I could use a dropdown bar to
    assign data_file_path for selective plotting of uploaded data using
    session['uploads'] and find_upload_path.
    '''
    data_file_path = session.get('path_to_last_upload', None)  # Uploaded File Path
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')  # read csv
    uploaded_df_html = uploaded_df.to_html()  # Converting to html Table
    return render_template('show_csv_data.html',
                           fname=session['fname'],
                           data_var=uploaded_df_html)

@app.route('/plot_data')
def plotData():
    '''
    Instead of session.get in data_file_path, I could use a dropdown bar to
    assign data_file_path for selective plotting of uploaded data using
    session['uploads'] and find_upload_path.
    '''
    # Read from Uploaded File Path
    data_file_path = session.get('path_to_last_upload', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    fig = px.line(uploaded_df, x='time', y='NE', title='Muh plot')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)    
    return render_template('plot_csv_data.html', fname=session['fname'],
                           data_var=uploaded_df, graphJSON=graphJSON)

@app.route('/plot_clusters')
def plotClusters():

    # Load form results
    user_settings = session['form_results']
    print("plotClusters")
    # data_file_path = session.get('path_to_last_upload', None)
    # user_settings['directory']=data_file_path # Update user_settings file path 
    non_user_settings = user_settings.copy()  # Modify user settings but for base dataset
    user_settings['directory'] = 'static/uploads/'  # Lame way of loading user input (VERY BAD CUZ ALL DATA CAN BE FROM DIFFERENT SESSIONS AND FORMATS)
    common_settings  = {'generation_time_path': 'data/generation_lengths/',
                        'exclude_subdirs': [],
                        'use_plotting_on_log10_scale': False,
                        'sep': '\t',
                        'data_file_descriptor': '.csv'}
    user_defaults = common_settings.copy()
    non_user_settings_to_update = {'directory': 'data/curves_80_20_split_birds/train_data/',
                                   'time_field': 'left_time_boundary',
                                   'value_field': 'lambda',
                                   'data_file_descriptor': '.txt'}
    
    user_defaults.update(user_settings) # Load user settings for user dataset
    
    non_user_settings.update(common_settings)
    non_user_settings.update(non_user_settings_to_update)
    # Manual dict val casting :-p
    non_user_settings = manual_modify_dict(my_dict=non_user_settings,
                                           to_int_list=to_int_list,
                                           to_float_list=to_float_list,
                                           to_bool_list=to_bool_list,
                                           to_list_list=to_list_list)
    user_defaults = manual_modify_dict(my_dict=user_defaults,
                                       to_int_list=to_int_list,
                                       to_float_list=to_float_list,
                                       to_bool_list=to_bool_list,
                                       to_list_list=to_list_list)
    print("non_user_settings = ", non_user_settings)
    print("user_settings = ", user_defaults)

    m_obj_base = Msmc_clustering(**non_user_settings)
    '''
    Might fail to correctly read cols if static/upload/ dir is full of files
    from other ppl's sessions! Common cause of Usecols do not match error.
    '''
    m_obj_user = Msmc_clustering(**user_defaults)
    m_obj_base.cluster_curves(plot_everything=False)
    cols = 2
    k = m_obj_base.manual_cluster_count              
    rows = given_col_find_row(k=k, cols=cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=([f"Cluster {label}" for label in range(1, k+1)]))
    fig.update_layout(hovermode='closest')
    # Adding background training curves to curve cluster plot
    for name in m_obj_base.name2series:
        add_curve_to_subplot(fig=fig,
                             name=name,
                             cols=cols,
                             Msmc_clustering=m_obj_base,
                             km=None,
                             marker_color='rgba(0, 180, 255, .3)', # From here on, goScatter_kwargs
                             )
    # Add curve from user input
    for name in m_obj_user.name2series:
        add_curve_to_subplot(fig=fig,
                             name=name,
                             cols=cols,
                             Msmc_clustering=m_obj_user,
                             km=m_obj_base.km,
                             marker_color='rgba(180, 0, 255, .8)'  # From here on, goScatter_kwargs,
                             )
    fig.update_layout(width=1900,
                      height=1000,)
    
    # create our callback function
    scatter = fig.data[0]
    colors = ['#a3a7e4'] * (len(m_obj_base.name2series) + len(m_obj_user.name2series))
    scatter.marker.color = colors
    scatter.marker.size = [10] * (len(m_obj_base.name2series) + len(m_obj_user.name2series))    
    def update_point(trace, points, selector):
        c = list(scatter.marker.color)
        s = list(scatter.marker.size)
        for i in points.point_inds:
            c[i] = '#bae2be'
            s[i] = 20
            with fig.batch_update():
                scatter.marker.color = c
                scatter.marker.size = s
    scatter.on_click(update_point)
    graphJSON = json.dumps(scatter, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('plot_clusters.html',
                           fname=session['fname'],
                           graphJSON=graphJSON)



@app.route('/user/<name>')
def user(name):
    '''
    NOTES:
    - render_template integrates Jinja2 template engine with app.
    - render_template automatically looks in the "templates/" directory
    
    PARAMETERS:
    arg1: file name of template
    name: argument for actual value for name variable
    '''
    return render_template('user.html', name=name)


@app.errorhandler(404)
def page_not_found(e):
    # print(type(e)) # <class 'werkzeug.exceptions.NotFound'>
    # print(e)
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("3001"), debug=True)
