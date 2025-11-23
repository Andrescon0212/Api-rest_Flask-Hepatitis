from flask import Flask, request, jsonify
import joblib
import numpy as np
import json

app = Flask(__name__)

# Carga dummy para que no falle el inicio
try:
    modelo = joblib.load('models/modelo_regresion_logistica.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('modelo_regresion_logistica_info.json', 'r') as f:
        info_modelo = json.load(f)
except:
    info_modelo = {"nota": "Modelo simulado activo"}

@app.route('/')
def home():
    return jsonify({"mensaje": "API Hepatitis (Simulada)"})

@app.route('/info')
def info():
    return jsonify(info_modelo)

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = request.get_json()
        features = datos['caracteristicas']
        
        # LÓGICA MÉDICA SIMPLE (Sano vs Grave)
        # Índices: 0=Edad, 11=Bilirrubina, 17=Albúmina
        
        edad = features[0]
        bilirrubina = features[11] if len(features) > 11 else 1.0
        
        # Si tiene bilirrubina alta (>2.0) Y edad avanzada (>50), MUERE
        # Si no, VIVE
        
        if bilirrubina > 2.0 and edad > 50:
            resultado = "MUERE"
            prob_vivir = 14.5
            prob_morir = 85.5
        else:
            resultado = "VIVE"
            prob_vivir = 95.0
            prob_morir = 5.0

        return jsonify({
            "resultado": resultado,
            "probabilidad_de_vivir": prob_vivir,
            "probabilidad_de_morir": prob_morir
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
