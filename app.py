from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load('modelo_bike.pkl')
scaler = joblib.load('scaler_bike.pkl')
columnas = joblib.load('columnas_bike.pkl')

@app.route('/')
def home():
    return 'API de predicción de bicicletas funcionando'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # armar el arreglo en el mismo orden que las columnas guardadas
    entrada = [data[col] for col in columnas]

    entrada_escalada = scaler.transform([entrada])

    prediccion = modelo.predict(entrada_escalada)

    return jsonify({'prediccion_cnt': float(prediccion[0])})

if __name__ == '__main__':
    app.run(debug=True)