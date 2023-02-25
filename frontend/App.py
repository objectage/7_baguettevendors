import csv
from flask import Flask, make_response, request, render_template, redirect, url_for

app = Flask(__name__, template_folder = 'templates')

users = []

@app.route('/')
def Index():
    return render_template('landing.html')

@app.route('/signin', methods = ['GET', 'POST'])
def SignIn():
    if request.method == "POST":
         return redirect(url_for('Chat'))
            
    return render_template('sign_in.html')

@app.route('/signup', methods = ['GET', 'POST'])
def SignUp():
    global users
    if request.method == "POST":
        t1 = [
            request.form.get("name"),
            request.form.get("email"),
            request.form.get("pass"),
        ]

        users.extend(t1)

        return redirect(url_for('DataEntry'))

    return render_template('sign_up.html')


@app.route('/input-data', methods = ['GET', 'POST'])
def DataEntry():
    global users
    if request.method == "POST":
        t2 = [
            request.form.get("N-value"),
            request.form.get("P-value"),
            request.form.get("K-value"),
            request.form.get("temp"),
            request.form.get("humidity"),
            request.form.get("pH"),
            request.form.get("rainfall"),
            request.form.get("crop")
        ]

        users.extend(t2)
        use = tuple(users)

        f = open("users.csv", "a")
        writer = csv.writer(f)
        writer.writerow(use)
        f.close()
        return redirect(url_for('Chat'))
    
    return render_template('data_entry.html')


@app.route('/chat')
def Chat():
    return render_template('chat.html')

if __name__ == "__main__":
    app.run()