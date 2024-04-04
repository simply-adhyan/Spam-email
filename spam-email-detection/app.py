import numpy as np
from flask import Flask, redirect, request, render_template
import pickle
import re
app = Flask(__name__)
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

repetitive_chars = r"(.)\1{3,}"
unusual_symbols = r"[^a-zA-Z0-9\s\?(:),.']!+"
unrealistic_length = r".{200,}"


def preprocess_features(email_content):
  text = email_content.lower()
  stop_words = set(stopwords.words('english'))
  text = ' '.join([word for word in text.split() if word not in stop_words])
  if any(re.search(pattern, text) for pattern in [repetitive_chars, unusual_symbols, unrealistic_length]):
    return "error"
  return [text]
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('email_content')
    print(request.form.get('email_content'))
    if not user_input:
        return render_template('index.html', error_message="Please enter email content")
    features = preprocess_features(user_input)
    if features == "error":
      return render_template('index.html', prediction_text="The Email is Nonsensical (Error)")

    try:
        prediction = model.predict_proba(features)[0]
        print(prediction)
        result = "spam" if prediction[0] <= 0.95 else "ham"
    except Exception as e:
        return render_template('index.html', error_message="Error: {}".format(e))

    return render_template('index.html', prediction_text="The Email is {}".format(result))

if __name__ == "__main__":
    app.run(debug=True)
