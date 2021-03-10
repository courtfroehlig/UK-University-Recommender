sfrom flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

#use pickle to load pre-trained model
model = pickle.load(open('uni_rec.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    
    return render_template('index.html', prediction_text='You should apply to {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
    