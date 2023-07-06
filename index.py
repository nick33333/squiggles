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
os.chdir("MSMC_clustering")
from MSMC_clustering import Msmc_clustering
os.chdir("../")
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

@app.route('/', methods=['GET', 'POST'])
def index():
    barred_args = ['self',
                   'data_file_descriptor',
                   'directory',
                   'generation_time_path',
                   'to_omit',
                   'exclude_subdirs',
                   'time_field',
                   'value_field',
                   'use_friendly_note']
    Msmc_clustering_args = Msmc_clustering.__init__.__code__.co_varnames
    Msmc_clustering_args = [i for i in Msmc_clustering_args if i not in barred_args]
    if request.method == 'POST':
        # upload file flask
        f = request.files.get('file')
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
        session['fname'] = data_filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        return render_template('index2.html', Msmc_clustering_args=Msmc_clustering_args)
    return render_template("index.html", Msmc_clustering_args=Msmc_clustering_args)

@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    
    return render_template('show_csv_data.html', fname=session['fname'], data_var=uploaded_df_html)

@app.route('/plot_data')
def plotData():
    # Read dataset
    form_data = request.form
    print(form_data)
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
    # Read dataset
    
    # CURRENTLY WORKING ON HOW TO READ FREAKING FORM DATA
    form_data = request.form
    # Make plotly subplot
    
    # Read from Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    fig = px.line(uploaded_df, x='time', y='NE', title='Muh plot')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)    
    
    return render_template('plot_csv_data.html', fname=session['fname'],
                           data_var=uploaded_df, graphJSON=graphJSON)



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
