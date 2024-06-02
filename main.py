
from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
stopword=set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")




app = Flask(__name__)

# Load the trained classifier
with open("classifier2.pkl", "rb") as f:
    classifier = pickle.load(f)
    

# Load the CountVectorizer
with open("count_vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)
    
        
stopword = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")



# Define a function for text cleaning
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message = clean(message)
        message = cv.transform([message]).toarray()
        prediction = classifier.predict(message)
        if prediction[0] == "Hate Speech":
            result = 'Hate Speech'
        elif prediction[0] == "Offensive Language":
            result = 'Offensive Language'
        else:
            result = 'No Hate and Offensive Speech'
        return render_template('after.html', prediction=result)
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")
if __name__ == '__main__':
    app.run(debug=True)


