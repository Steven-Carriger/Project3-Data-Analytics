import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn import *
import pickle

flask_app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# The route() decorator to tell Flask what URL should trigger our function.
# ‘/’ is the root of the website, such as www.westga.edu
@flask_app.route("/")   
def index():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])   
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features)
    features_scaled = scaler.fit_transform(features.reshape(1, -1))
    result = model.predict(features_scaled)
    if result > 0.0:
        result = 'Malignant'
    else:
        result = 'Benign'
    return render_template("index.html", predicted_text = result)

if __name__ =="__main__":
    flask_app.run(debug = True)
