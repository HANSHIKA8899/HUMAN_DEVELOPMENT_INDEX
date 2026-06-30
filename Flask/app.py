from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('HDI.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Predicted HDI: {prediction[0][0]:.4f}')
if __name__ == "__main__":
    app.run(debug=True)