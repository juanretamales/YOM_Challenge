import os

from fastapi.testclient import TestClient
import mlflow 
import mlflow.sklearn
import mlflow.pyfunc
import json
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from sklearn.ensemble import GradientBoostingClassifier
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from datetime import datetime
from pytest_check import check

from main import app
from model_instances import ModelSingleton

# Cliente para hacer las pruebas                                                                                                                                                                                                
client = TestClient(app)

def test_inference_with_transform():
    version="2"
    experiment_name = f"Spike_Challenge_V{version}"

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name, tags={"version": "v1"}), 

    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Setup de MLflow
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
    
    apikey = "kXwagyJkMH0Q3fT3MorqmRupqpjq1FCsVijy2P3nwrQQpExcWl"
    
    current_tempo_scaler_file = 'tempo_min_max_scaler.save'
    current_tempo_scaler = ModelSingleton(current_tempo_scaler_file)
    
    # cargamos el csv con la nueva data para revisar si cambiaron los inputs
    # Para ahorrar tiempo se usa un CSV filtrado y procesado, pero se podria usar uno sin procesar
    df=pd.read_csv('nueva_data.csv',encoding='utf-8',sep=',')

    # cuando aun es dataframe, separo para obtener el de validación para hacer pruebas posteriores
    df_val = df.sample(int(len(df)*0.2))
    df_train = df[~df.index.isin(df_val.index)]

    # # Drop unnecessary columns
    # columns_to_use = [ 'danceability', 'energy', 'speechiness', 'acousticness','valence','tempo_scale', 'reggaeton']
    # df_train = df_train[columns_to_use]

    # columns_to_use = [ 'danceability', 'energy', 'speechiness', 'acousticness','valence','tempo_scale', 'reggaeton']
    # df_val = df_val[columns_to_use]

    # Primero obtenemos la información del modelo actual
    y_current_model_pred=[] # guardo resultados
    y_current_model_val=[] # guardo respuesta correcta
    for index, row in df_val.iterrows():
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
        y_current_model_pred.append(json_response['id'])
        y_current_model_val.append(row['reggaeton'])
    
    current_model_accuracy = accuracy_score(y_current_model_val, y_current_model_pred)
    current_model_precision = precision_score(y_current_model_val, y_current_model_pred, average='weighted')
    current_model_recall = recall_score(y_current_model_val, y_current_model_pred, average='weighted')
    print(f'[Nuevo Modelo]Accuracy: {current_model_accuracy}\nPrecision: {current_model_precision}\nRecall: {current_model_recall}')


    # sin embargo, para el MinMaxScaler, se mantuvo ya que es mas facil que cambie el rango de entrada
    with check:
        current_tempo_scaler.model.data_min_[0]<=df['tempo'].min()
    with check:
        current_tempo_scaler.model.data_max_[0]>=df['tempo'].max()

    # Ahora se entrena un nuevo modelo con el dataset leido
    new_tempo_scaler = MinMaxScaler()
    df_train.loc[:, 'tempo'] = new_tempo_scaler.fit_transform(df_train[['tempo']].values)

    # Drop unnecessary columns
    columns_to_use = [ 'danceability', 'energy', 'speechiness', 'acousticness','valence','tempo', 'reggaeton']
    df_train = df_train[columns_to_use]

    # separo entre caracteristivas y objetivo
    X = df_train.loc[:, df_train.columns != 'reggaeton'].values
    y = df_train['reggaeton'].values.ravel()

    # Hago split entre train y test, validación se usara otro archivo para probar
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size= 0.2,
        random_state= 1234
        )

    # Los parametros fueron cambiadas a su equivalente para las nuevas funciónes debido a la perdida del archivo requirements.txt, actualizando presort que ahora es automatica, y max_features que ahora es sqrt
    classifier_GB=GradientBoostingClassifier(
        random_state=0,subsample=0.8,n_estimators=300,warm_start=True,
        max_features='sqrt', min_samples_split=106,min_samples_leaf=177)

    # Devuelvo a DataFrame para prevenir el Warning de usar feature names y mejorar la legibilidad y mantenimiento del código
    df_train = pd.DataFrame(data=X_train, columns=[ 'danceability', 'energy', 'speechiness', 'acousticness','valence','tempo'])

    # Entrenar el modelo con los nuevos datos de entrenamiento
    classifier_GB.fit(df_train, y_train)

    # para efectos de esta prueba, se guardara el experimento en MLflow pero, podria no hacerlo
    with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=f"RunTest_{datetime.now()}"):


        # Logueo los mejores resultados
        mlflow.log_params({
            "n_estimators": classifier_GB.n_estimators,
            "max_depth": classifier_GB.max_depth,
            "learning_rate": classifier_GB.learning_rate,
            "train_length": len(X_train),
            "test_length": len(X_test),
            "MinMaxScaler_min": new_tempo_scaler.data_min_[0],
            "MinMaxScaler_max": new_tempo_scaler.data_max_[0]
        })
        
        # Logueo los resultados de la prueba# Devuelvo a DataFrame para prevenir el Warning de usar feature names y mejorar la legibilidad y mantenimiento del código
        df_test = pd.DataFrame(data=X_test, columns=[ 'danceability', 'energy', 'speechiness', 'acousticness','valence','tempo'])

        # Obtengo las predicciones
        y_pred = classifier_GB.predict(df_test)

        # Calculo el acuraccy y el AUC
        new_model_accuracy = accuracy_score(y_test, y_pred)
        new_model_aprecision = precision_score(y_test, y_pred, average='weighted')
        new_model_recall = recall_score(y_test, y_pred, average='weighted')
        print(f'[Nuevo Modelo]Accuracy: {new_model_accuracy}\nPrecision: {new_model_aprecision}\nRecall: {new_model_recall}')

        # Log de parámetros
        metrics ={
            'new_accuracy': new_model_accuracy,
            'new_precision':  new_model_aprecision, 
            'new_recall':  new_model_recall,
            "new_MinMaxScaler_min": new_tempo_scaler.data_min_[0],
            "new_MinMaxScaler_max": new_tempo_scaler.data_max_[0],
            'current_accuracy': current_model_accuracy,
            'current_precision': current_model_precision,
            'current_recall': current_model_recall,
            "current_MinMaxScaler_min": current_tempo_scaler.model.data_min_[0],
            "current_MinMaxScaler_max": current_tempo_scaler.model.data_max_[0]
            }

        mlflow.log_metrics(metrics)

        # Definir el esquema manualmente
        input_schema = Schema([
            ColSpec("float", "danceability"),
            ColSpec("float", "energy"),
            ColSpec("float", "speechiness"),
            ColSpec("float", "acousticness"),
            ColSpec("float", "valence"),
            ColSpec("float", "tempo")
        ])

        output_schema = Schema([ColSpec("integer")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = [X_train[0]]

        # Log model & artifacts
        mlflow.sklearn.log_model(classifier_GB, f"Reggaeton_Classifier_V{version}",signature=signature, input_example=input_example)

    # comparamos mmetricas
    with check:
        new_model_accuracy>=current_model_accuracy
    with check:
        new_model_aprecision>=current_model_precision
    with check:
        new_model_recall>=current_model_recall
    
    # estas metricas se revisen para que la prueba falle y el personal correspondiente verifique si efectivamente hay que reemplazar el antiguo MinMaxScaler con el nuevo
    with check:
        new_tempo_scaler.data_min_[0]<=current_tempo_scaler.model.data_min_[0]
    with check:
        new_tempo_scaler.data_max_[0]>=current_tempo_scaler.model.data_max_[0]



    # se podria hacer mas pruebas sin usar transform como esta en ipynb pero 
    # se omitio en esta ocación por tiempo y dar prioridad a objetivos mas importantes