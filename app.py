from flask import Flask, request, jsonify
import pickle
import numpy as np
model = pickle.load(open('cropPredict_ML_Model2.pkl','rb'))

app=Flask(__name__)
@app.route('/')
def home():
    return "Hello anu"
@app.route('/predict',methods=['POST'])
def predict():
    N = request.form.get('N')
    P = request.form.get('P')
    K = request.form.get('K')

    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    input_query = np.array([[int(N),int(P),int(K),float(temperature),float(humidity),float(ph),float(rainfall)]])
    result = model.predict(input_query)[0]
    print(jsonify(result))
    return jsonify({'crop_name':result})
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
