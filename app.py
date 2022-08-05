import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    Pregnancies = float(request.form['pregnancies'])
    DiabetesPedigreeFunction = float(request.form['dpf'])
    Age = float(request.form['age'])
    Glucose = float(request.form['glucose'])
    BloodPressure = float(request.form['bloodpressure'])
    SkinThickness = float(request.form['skinthickness'])
    Insulin = float(request.form['insulin'])
    BMI = float(request.form['bmi'])
    
    data = np.array([[Pregnancies, DiabetesPedigreeFunction, Age, Glucose, 
                      BloodPressure, SkinThickness, Insulin, BMI]])
    
    prediction = model.predict(data)



    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)