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
import json

sys.path.insert(1, 'MSMC_clustering/')
from MSMC_clustering import Msmc_clustering
from MSMC_plotting import *

'''
To use flask shell:
$ export FLASK_APP=index
$ flask shell
'''

UPLOAD_FOLDER = os.path.join('static', 'uploads')
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.secret_key = 'This is your secret key to utilize session in Flask'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Hardcoded stuff for now, should allow user to select these settings from a drop bar
app.config['animal'] = 'bird' # bird and mammal are options
app.config['preprocessing'] = "msmc_rtp_it" # msmc, msmc_it, and msmc_rtp_it (realtime processed and interpolated) should be my options so far
app.config['k'] = 7 # k= 2, ..., 15, ... models should be ready

# Will fill up this dict with some fat models using specified settings
km_dict = dict()
km_dict[app.config['k']] = TimeSeriesKMeans.from_pickle(f"models/{app.config['animal']}_{app.config['preprocessing']}_k{app.config['k']}_km.pkl")

# My testing model
app.config[f"k{app.config['k']}"] = km_dict[app.config['k']] # Should be a tslearn TimeSeriesKMeans obj
barred_args = ['self',
               'directory',
               'generation_time_path',
               'to_omit',
               'exclude_subdirs',
               'use_friendly_note',
               'readfile_kwargs',
               'tmp_data',
               'use_plotting_on_log10_scale']
Msmc_clustering_args = Msmc_clustering.__init__.__code__.co_varnames
Msmc_clustering_args = [i for i in Msmc_clustering_args if i not in barred_args]
app.config['Msmc_clustering_args'] = Msmc_clustering_args

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['form_results'] = request.form  # saves form data after upload
        f = request.files.get('file')  # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
        session['fname'] = data_filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        return render_template('index2.html', Msmc_clustering_args=Msmc_clustering_args)
    return render_template("index.html", Msmc_clustering_args=Msmc_clustering_args)

@app.route('/show_data')
def showData():
    data_file_path = session.get('uploaded_data_file_path', None)  # Uploaded File Path
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')  # read csv
    uploaded_df_html = uploaded_df.to_html()  # Converting to html Table
    
    return render_template('show_csv_data.html', fname=session['fname'], data_var=uploaded_df_html)

@app.route('/plot_data')
def plotData():
    # Read dataset
    form_data = request.form
    print(list(form_data.items()))
    # Make plotly subplot
    
    # Read from Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    fig = px.line(uploaded_df, x='time', y='NE', title='Muh plot')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)    
    
    return render_template('plot_csv_data.html', fname=session['fname'],
                           data_var=uploaded_df, graphJSON=graphJSON)

@app.route('/plot_clusters')
def plotClusters():
    def manual_modify_dict(my_dict,
                           to_int_list,
                           to_float_list,
                           to_bool_list,
                           to_list_list):
        my_dict = to_int(to_int_list, my_dict)
        my_dict = to_float(to_float_list, my_dict)
        my_dict = to_bool(to_bool_list, my_dict)
        my_dict = to_li(to_list_list, my_dict)
        return my_dict
        
    def to_int(to_int_list, my_dict):
        for thing in to_int_list:
            my_dict[thing] = int(my_dict[thing])
        return my_dict
    
    def to_float(to_float_list, my_dict):
        for thing in to_float_list:
            my_dict[thing] = float(my_dict[thing])
        return my_dict
            
    def to_bool(to_bool_list, my_dict):
        for thing in to_bool_list:
            my_dict[thing] = my_dict[thing] == 'True'
        return my_dict

    def to_li(to_list_list, my_dict):
        for thing in to_list_list:
            umm = thing.split(',')
            if len(umm) > 1:
                my_dict[thing] = [float(t) for t in umm]
                print(my_dict[thing])
        return my_dict

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
    # Load form results
    user_settings = session['form_results']
    # CURRENTLY WORKING ON HOW TO READ FREAKING FORM DATA
    print("plotClusters")
    data_file_path = session.get('uploaded_data_file_path', None)   
    # user_settings['directory']=data_file_path # Update user_settings file path 
    non_user_settings = user_settings.copy() # Modify user settings but for base dataset
    user_settings['directory'] = 'static/uploads/' # Lame way of loading user input (VERY BAD CUZ ALL DATA CAN BE FROM DIFFERENT SESSIONS AND FORMATS)
    common_settings  = {'generation_time_path':'data/generation_lengths/',
                        'exclude_subdirs':[],
                        'use_plotting_on_log10_scale': False,
                        'sep': '\t',
                        'data_file_descriptor': '.csv'}
    user_defaults = common_settings.copy()
    non_user_settings_to_update = {'directory':'data/msmc_curve_data_birds/',
                                   'time_field':'left_time_boundary',
                                   'value_field':'lambda',
                                   'data_file_descriptor':'.txt'}
    
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
    
    m_obj_user = Msmc_clustering(**user_defaults)
    m_obj_base.cluster_curves(plot_everything=False)
    cols = 2
    k=m_obj_base.manual_cluster_count                 
    rows = given_col_find_row(k=k, cols=cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=([f"Cluster {label}" for label in range(1, k+1)]))
    # Adding background training curves to curve cluster plot
    for name in m_obj_base.name2series:
        add_curve_to_subplot(fig=fig,
                             name=name,
                             cols=cols,
                             Msmc_clustering=m_obj_base,
                             km=None,
                             marker_color='rgba(0, 180, 255, .5)')
    # Add curve from user input
    for name in m_obj_user.name2series:
        add_curve_to_subplot(fig=fig,
                            name=name,
                            cols=cols,
                            Msmc_clustering=m_obj_user,
                            km=m_obj_base.km,
                            marker_color='rgba(180, 0, 255, .8)')
    fig.update_layout(width=1900,
                      height=1000,)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)    
    return render_template('plot_clusters.html', fname=session['fname'], graphJSON=graphJSON)



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
