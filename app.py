from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model_py import prepare_data, train_and_evaluate

app = Flask(__name__)

# Variables globales
scaler = None
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/click-prediction')
def click_prediction():
    return render_template('click_prediction.html')

@app.route('/face-recognition')
def face_recognition():
    return render_template('face_recognition.html')

@app.route('/train_python', methods=['POST'])
def train_python():
    global scaler, model
    df = pd.read_csv('ad_mini.csv')
    df_for_scaler = df.copy()
    df_for_scaler['Gender'] = df_for_scaler['Gender'].map({'Male': 0, 'Female': 1})
    features = df_for_scaler.drop('Clicked on Ad', axis=1)

    scaler = StandardScaler()
    scaler.fit(features)

    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)
    model, accuracy = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
    print(f"Modèle Python entraîné avec une précision de : {accuracy}")
    return jsonify({"message": "Modèle Python chargé avec succès.", "accuracy": accuracy})

@app.route('/predict', methods=['POST'])
def predict():
    global scaler, model
    # Récupérer les données du formulaire
    time_spent = float(request.form['time_spent'])
    age = float(request.form['age'])
    area_income = float(request.form['area_income'])
    internet_usage = float(request.form['internet_usage'])
    gender = request.form['gender']
    gender_val = 0 if gender.lower() == 'male' else 1

    # Créer le tableau d'entrée et appliquer le scaler
    input_data = np.array([[time_spent, age, area_income, internet_usage, gender_val]])
    input_data_scaled = scaler.transform(input_data)
    
    # Prédiction
    prediction = model.predict(input_data_scaled)
    click = 1 if prediction[0][0] > 0.5 else 0
    
    return render_template('result.html', prediction=click)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)