import os
import sys
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
import mlflow.pyfunc
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json
from pytest_check import check

from main import app
from model_instances import ModelSingleton

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
        y_pred.append(json_response['id'])
        y_val.append(row['reggaeton'])
        # Calcular métricas (opcional)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')

    # Se uso 0.99 como valor de acuerdo a la data registrada en el entrenamiento
    # Se podría intentar obtener de MLFlow pero por ahora se dejo con estos valores
    check.greater_equal(accuracy, 0.99, "Comparar accuracy entre predicciones y el valor en entrenamiento")
    check.greater_equal(precision, 0.99, "Comparar precision entre predicciones y el valor en entrenamiento")
    check.greater_equal(recall, 0.99, "Comparar recall entre predicciones y el valor en entrenamiento")
    
    # Revisamos si cambio la data de entrada para el MinMaxScaler
    check.less_equal(
        tempo_scaler.model.data_min_[0], 
        df['tempo'].min(),
        "Comparar el menor valor de los datos de entrada entre modelo y datos de prueba"
    )
    check.greater_equal(
        tempo_scaler.model.data_max_[0], 
        df['tempo'].max(),
        "Comparar el mayor valor de los datos de entrada entre modelo y datos de prueba"
    )