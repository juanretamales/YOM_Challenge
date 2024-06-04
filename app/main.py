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
import numpy as np
from modelos import ModelSingleton
from base_models import Feature, FeatureWithTrue

# Crear la aplicación FastAPI
## Crear la documentación con un archivo si existe
description_file = ""
if os.path.isfile('redoc.md'):
    with open('redoc.md') as f:
        description_file = f.read()

class Settings(BaseSettings):
    app_name: str = "Reggaeton Classifier API"
    description: str = description_file
    version: str = "1.0.0"
    msg:str = f"Start on {dt.datetime.now()}"
    flag_run:bool = True

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

MODEL_URI = os.environ["MODEL_URI"] if "MODEL_URI" in os.environ else "models:/Reggaeton_Classifier_V1.0/latest"

# Configurar SlowAPI para limitar el tráfico
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@limiter.limit("10/minute")
@app.get("/")
async def root(request: Request):
    """Endpoint for test the connection with this API"""
    response = {"app": f"{settings.app_name} v{settings.version}", 
    "status": "Online" if settings.flag_run else "Offline",
    "message": settings.msg}
    if settings.flag_run:
        return JSONResponse(content=response, status_code=200)
    else:
        return JSONResponse(content=response, status_code=500)


@limiter.limit("60/minute")
@app.post("/predict/")
async def predict(request: Request, feature:Feature, transform:bool = True, apikey:str=""):
    """Endpoint for use the Model of MLFlow, transform on False only for testing purporse"""
    # For falta de tiempo, solo se usara una apikey fija, se recomienda usar token o un administrador
    if apikey != "kXwagyJkMH0Q3fT3MorqmRupqpjq1FCsVijy2P3nwrQQpExcWl":
        return JSONResponse({"status": "Online" if settings.flag_run else "Offline","detail":"The credencials is not valid"}, status_code=401)
    try:
        model_instance = ModelSingleton(MODEL_URI)

        # Primero escalamos la data recibida
        if transform:
            tempo_scaler_file = 'tempo_min_max_scaler.save'
            tempo_scaler = ModelSingleton(tempo_scaler_file)
            scaled_tempo=tempo_scaler.model.transform([[feature.tempo]])[0][0]
            feature.tempo = float(scaled_tempo)

        # Ahora transformamos a float32 para evitar problemas al usar el modelo
        data = feature.dict()
        for key, val in data.items():
            data[key]=np.float32(val)

        predictions = model_instance.model.predict([data])[0]
        id2name={
            0:"Is not Reggaeton",
            1:"Is Reggaeton"
        }
        return {
            "predictions": id2name[predictions], "class": int(predictions)
        }
    except Exception as e:
        settings.flag_run=False
        settings.msg="Can't load MLflow Model"
        return JSONResponse({"status": "Online" if settings.flag_run else "Offline","detail":str(e)}, status_code=500)

# @app.post("/retrain-model")
# async def retrain_model(new_dataset: list, secure_password:str=''):
#     """Endpoint for use the Model of MLFlow, transform on False only for testing purporse"""
    
#     # Registrar nuevo modelo en MLflow
#     new_model = SimpleRegressionModel()
#     register_model(new_model, model_path)
    
#     # Evaluar drift de datos
#     evaluate_data_drift(mlflow.active_run().info.run_id, new_data)
    
#     return {"message": "Model retrained and evaluated for data drift."}