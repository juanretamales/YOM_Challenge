import os

from fastapi.testclient import TestClient
import mlflow.pyfunc
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

from main import app
from modelos import ModelSingleton

# Cliente para hacer las pruebas                                                                                                                                                                                                
client = TestClient(app)

def test_inference_with_transform():
    
    apikey = "kXwagyJkMH0Q3fT3MorqmRupqpjq1FCsVijy2P3nwrQQpExcWl"
    
    tempo_scaler_file = 'tempo_min_max_scaler.save'
    tempo_scaler = ModelSingleton(tempo_scaler_file)
    
    # cargamos el csv de val que guardamos previamente, como ya esta mayormente procesado, no necesita procesar
    df=pd.read_csv('data_val.csv',encoding='utf-8',sep=',')

    # Datos de prueba para hacer predicciones
    y_pred=[] # guardo resultados
    y_val=[] # guardo respuesta correcta
    for index, row in df.iterrows():
        data = {
            "danceability": row['danceability'],
            "energy": row['energy'],
            "speechiness": row['speechiness'],
            "acousticness": row['acousticness'],
            "valence": row['valence'],
            "tempo": row['tempo']
        }

        # Hacer una solicitud POST al servidor MLflow para obtener predicciones
        headers = {'Content-type': 'application/json'}
        # Se usa la recomendación de FastAPI para hacer la prueba
        response = client.post(f"/predict/?apikey={apikey}", json=data, headers=headers)

        # Extraer las predicciones del JSON de respuesta
        json_response = response.json()
        y_pred.append(json_response['class'])
        y_val.append(row['reggaeton'])
        # Calcular métricas (opcional)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')

    
    assert accuracy>=0.99
    assert precision>=0.99
    assert recall>=0.99

    # aseguro el rango de valores de entrada para el MinMaxScaler
    assert tempo_scaler.model.data_min_[0]<=df['tempo'].min()
    assert tempo_scaler.model.data_max_[0]>=df['tempo'].max()