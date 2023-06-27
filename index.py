from flask import Flask, render_template
from flask import request
from flask import make_response
from flask import redirect

'''
To use flask shell:
$ export FLASK_APP=index
$ flask shell
'''

app = Flask(__name__)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("3000"), debug=True)
