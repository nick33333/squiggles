from flask import (
    Flask,
    render_template,
    request,
    make_response,
    redirect
)
# from flask.ext.bootstrap import Bootstrap
from flask_bootstrap import Bootstrap
import pickle
import numpy as np
# import pandas as pd
'''
To use flask shell:
$ export FLASK_APP=index
$ flask shell
'''

app = Flask(__name__)
bootstrap = Bootstrap(app)

# # request obj
# @app.route('/')
# def index():
#     user_agent = request.headers.get('User-Agent')
#     return '<p>Your browser is %s</p>' % user_agent

# # response obj
# @app.route('/')
# def index():
#     response = make_response('<h1>This document carries a cookie!</h1>')
#     response.set_cookie('answer', '42')
#     return response

# # redirect
# @app.route('/')
# def index():
#     return redirect('/nickdefault')
def add(a,b):
    return a+b

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
    return "<p>Hello, World!</p>"

@app.route('/index')
def index():
    return render_template('index.html')

# @app.route("/<name>")
# def user(name):
#     return f"<p>Hello {name}<p>"

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
