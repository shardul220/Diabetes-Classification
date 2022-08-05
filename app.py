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
    
    Pregnancies = int(request.form['pregnancies'])
    DiabetesPedigreeFunction = float(request.form['dpf'])
    Age = int(request.form['age'])
    Glucose = int(request.form['glucose'])
    BloodPressure = int(request.form['bloodpressure'])
    SkinThickness = int(request.form['skinthickness'])
    Insulin = int(request.form['insulin'])
    BMI = float(request.form['bmi'])
    
    data = np.array([[Pregnancies, Glucose, 
                      BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    prediction = model.predict(data)



    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)