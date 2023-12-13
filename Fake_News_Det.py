from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

def is_authenticated(username, password):
    return username == 'admin' and password == '123'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if is_authenticated(username, password):
        session['logged_in'] = True
        return redirect(url_for('fake_news_detector'))
    else:
        return render_template('login.html', error='Invalid credentials')

@app.route('/fake_news_detector')
def fake_news_detector():
    if 'logged_in' in session and session['logged_in']:
        return render_template('index.html')
    else:
        return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' in session and session['logged_in']:
        if request.method == 'POST':
            message = request.form['message']
            pred = fake_news_det(message)
            print(pred)
            return render_template('index.html', prediction=pred)
        else:
            return render_template('index.html', prediction="Something went wrong")
    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

