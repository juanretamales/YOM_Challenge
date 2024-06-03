from typing import List
import os
import datetime as dt
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import joblib
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Crear la aplicación FastAPI
description_file = ""

class Settings(BaseSettings):
    app_name: str = "Reggaeton Classifier API"
    description: str = description_file
    version: str = "1.0.0"

settings = Settings()
app = FastAPI(title=settings.app_name,
                description=settings.description,
                docs_url='/docs',
                version=settings.version,
                )
# dependiendo de la APP se vera el tema de CORS, por ahora esta desabilitado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# Configurar MLFlow
MLFLOW_URI = os.environ["MLFLOW_URI"] if "MLFLOW_URI" in os.environ else "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(MLFLOW_URI)

flag_run=True
msg=f"Start on {dt.datetime.now()}"


# Cargar el modelo MLFlow
try:
    model = mlflow.pyfunc.load_model("models:/Reggaeton_Classifier_V1.0/latest")
except:
    flag_run=False
    msg="Can't load MLflow Model"
    

# Initialize MinMaxScaler for loudness
loudness_scaler_file = 'loudness_min_max_scaler.save'
if os.path.exists(loudness_scaler_file):
    loudness_scaler = joblib.load(loudness_scaler_file)
else:
    flag_run=False
    msg="Can't load MLflow Model"

# Initialize MinMaxScaler for tempo
tempo_scaler_file = 'tempo_min_max_scaler.save'
if os.path.exists(tempo_scaler_file):
    tempo_scaler = joblib.load(tempo_scaler_file)
else:
    flag_run=False
    msg="Can't load MLflow Model"

# Configurar SlowAPI para limitar el tráfico
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Feature(BaseModel):
    danceability:float
    energy:float
    speechiness:float
    acousticness:float
    valence:float
    tempo:float
    # loudness:float

@limiter.limit("10/minute")
@app.get("/")
async def root(request: Request):
    """Endpoint for test the connection with this API"""
    response = {"app": f"{settings.app_name} v{settings.version}", 
    "status": "Online" if flag_run else "Offline",
    "message":msg}
    if flag_run:
        return JSONResponse(content=response, status_code=200)
    else:
        return JSONResponse(content=response, status_code=500)


@limiter.limit("60/minute")
@app.post("/predict/")
async def predict(request: Request, feature:Feature, transform:bool = True):
    """Endpoint for use the Model of MLFlow, transform on False only for testing purporse"""
    try:
        # Camino 1, usar array para evitar uso de dataframe
        # Primero escalamos la data recibida
        if transform:
            scaled_tempo=tempo_scaler.transform([[feature.tempo]])[0][0]
            scaled_loudness=loudness_scaler.transform([[feature.loudness]])[0][0]
        else: # si es falso, es que ya esta transformado
            scaled_tempo=feature.tempo
            scaled_loudness=feature.loudness

        # Ahora juntamos
        data = [
            feature.danceability,
            feature.energy,
            feature.speechiness,
            feature.acousticness,
            feature.valence,
            scaled_tempo,
            scaled_loudness
        ]
        # Camino 2, replicar camino con Dataframe
        # import pandas as pd
        # df = pd.DataFrame([feature.dict()])
        # df.loc[:, 'loudness_scale'] = loudness_scaler.transform(df[['loudness']].values)
        # df.loc[:, 'tempo_scale'] = tempo_scaler.transform(df[['tempo']].values)
        # del df['loudness']
        # del df['tempo']
        # Obtenemos la predicción 
        predictions = model.predict([data])[0]
        id2name={
            0:"Is not Reggaeton",
            1:"Is Reggaeton"
        }
        return {
            "predictions": id2name[predictions], "class": int(predictions)
        }
    except Exception as e:
        raise JSONResponse({"status": "Online" if flag_run else "Offline","detail":str(e)},status_code=500)

