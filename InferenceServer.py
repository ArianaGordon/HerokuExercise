from flask import Flask, request
from pickle import load
import numpy as np
import pandas as pd
import json
import os

# Create flask application
app = Flask(__name__)


@app.route('/predict_single')
def predict_single():
    with open('gnb_heart.pkl', 'rb') as input_file:
        gnb = load(input_file)
    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')
    trestbps = request.args.get('trestbps')
    chol = request.args.get('chol')
    fbs = request.args.get('fbs')
    restecg = request.args.get('restecg')
    thalach = request.args.get('thalach')
    exang = request.args.get('exang')
    oldpeak = request.args.get('oldpeak')
    slope = request.args.get('slope')
    ca = request.args.get('ca')
    thal = request.args.get('thal')
    array = np.reshape([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal], (1, -1))
    result = gnb.predict(array)
    return str(result)


@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    # Load data from json object
    data = request.json
    df = pd.read_json(data)

    # Open file containing model
    with open('gnb_heart.pkl', 'rb') as input_file:
        gnb = load(input_file)

    result = gnb.predict(df).tolist()
    with open('predictions.json', 'w') as json_file:
        json.dump(result, json_file, separators=(',', ':'))
    return str(result)


if __name__ == '__main__':
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0', port=int(port))
