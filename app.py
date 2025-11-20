from flask import Flask, request, jsonify
import joblib
import numpy as np
import json

app = Flask(__name__)

# Cargar el modelo y el scaler al iniciar la app
modelo = joblib.load('modelo_regresion_logistica.pkl')
scaler = joblib.load('scaler.pkl')

# Cargar la información del modelo
with open('modelo_regresion_logistica_info.json', 'r') as f:
    info_modelo = json.load(f)

@app.route('/')
def home():
    return jsonify({
        "mensaje": "API del Modelo de Hepatitis - Regresión Logística",
        "endpoints": {
            "/": "Información de la API",
            "/info": "Información del modelo",
            "/predecir": "Realizar predicción (POST)"
        }
    })

@app.route('/info')
def info():
    return jsonify(info_modelo)

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Recibir los datos del request
        datos = request.get_json()
        
        # Convertir a array numpy
        caracteristicas = np.array([datos['caracteristicas']])
        
        # Escalar los datos
        caracteristicas_scaled = scaler.transform(caracteristicas)
        
        # Hacer la predicción
        prediccion = modelo.predict(caracteristicas_scaled)
        probabilidad = modelo.predict_proba(caracteristicas_scaled)
        
        return jsonify({
            "prediccion": int(prediccion[0]),
            "probabilidad_clase_0": float(probabilidad[0][0]),
            "probabilidad_clase_1": float(probabilidad[0][1]),
            "mensaje": "Predicción realizada exitosamente"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
