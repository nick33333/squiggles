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
msmc_obj_dict = dict()
msmc_obj_dict[app.config['k']] = pd.read_pickle(f"models/{app.config['animal']}_{app.config['preprocessing']}_k{app.config['k']}_blob.pkl")

# My testing model
app.config[f"k{app.config['k']}"] = msmc_obj_dict[app.config['k']][0][3][app.config['k']][999].km # Should be a tslearn TimeSeriesKMeans obj

@app.route("/") # app.route decorator adds url to app's URL map
def hello_world():
    '''
    URL map works by mapping URLs to view functions
    ex:
    >>> from index import app
    >>> app.url_map
    >>> app.url_map
    Map([<Rule '/' (HEAD, OPTIONS, GET) -> hello_world>,
    <Rule '/static/<filename>' (HEAD, OPTIONS, GET) -> static>,
    <Rule '/<name>' (HEAD, OPTIONS, GET) -> user>])
    >>>
    '''
    print('request:')
    print(request)
    print(request.files)
    print(session)
    return "<p>Hello, World!</p>"


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # upload file flask
        f = request.files.get('file')
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        return render_template('index2.html')
    return render_template("index.html")

@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html', data_var=uploaded_df_html)

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
    app.run(host="0.0.0.0", port=int("3000"), debug=True)
