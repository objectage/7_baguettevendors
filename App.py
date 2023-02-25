from flask import Flask, make_response, request, render_template

app = Flask(__name__, template_folder = 'templates')

@app.route('/')
def Index():
    return render_template('landing.html')

@app.route('/login')
def Login():
    return render_template('sign_in.html')

@app.route('/signup')
def SignUp():
    return render_template('sign_up.html')

if __name__ == "__main__":
    app.run(debug = True)