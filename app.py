from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import sys

app = Flask(__name__)

# use pickle to load pre-trained model
# model = pickle.load(open('uni_rec.pkl', 'rb'))
model = pickle.load(open('models/uni_wizz_v0.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

href="{{ url_for('static', filename='style.css')}}"

@app.route('/predict', methods=['POST'])
def predict():
    raw_features = request.form.to_dict()
    final_features = pd.DataFrame(
        [raw_features.values()],
        columns=raw_features.keys())

    prediction = model.predict(final_features)[0]

    return render_template('index.html',\
        prediction_text='You should apply to ' + str(prediction) + '.')

if __name__ == "__main__":
    app.run(debug=True)
