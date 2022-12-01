from flask import Flask
from flask import jsonify
from flask import request
from flask import Blueprint
from flask import Flask
from flaskapi_home import home_bp
from flaskapi_contact import contact_bp

app = Flask(__name__)

@app.route('/person/')
def hello():
    return jsonify({'name':'Jimit',
                    'address':'India'})

@app.route('/numbers/')
def print_list():
    return jsonify(list(range(5)))

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

@app.route('/<int:number>/')
def incrementer(number):
    return "Incremented number is " + str(number+1)

@app.route('/home/')
def home():
    return "Home page"

@app.route('/contact')
def contact():
    return "Contact page"

@app.route('/teapot/')
def teapot():
    return "Would you like some tea?", 418

@app.before_request
def before():
    print("This is executed BEFORE each request.")

app.register_blueprint(home_bp, url_prefix='/home')
app.register_blueprint(contact_bp, url_prefix='/contact')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


