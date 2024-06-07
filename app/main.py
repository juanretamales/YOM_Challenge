from typing import List
import os
import datetime as dt
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import joblib
import logging
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np
from model_instances import ModelSingleton
from base_models import Feature, RootResponse, PredictionResponse


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Crear la documentación con un archivo si existe
description_file = ""
if os.path.isfile('redoc.md'):
    with open('redoc.md') as f:
        description_file = f.read()

class Settings(BaseSettings):
    app_name: str = "Reggaeton Classifier API"
    description: str = description_file
    version: str = "1.0.0"
    msg: str = f"Start on {dt.datetime.now()}"
    flag_run: bool = True

settings = Settings()

# Crear la aplicación FastAPI
app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    docs_url='/docs',
    version=settings.version,
)

# Dependiendo de la aplicación se verá el tema de CORS, por ahora está deshabilitada la protección
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Configurar MLFlow
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://127.0.0.1:5000/")
mlflow.set_tracking_uri(MLFLOW_URI)
logger.info(f"MLFlow Tracking URI: {MLFLOW_URI}")

MODEL_URI = os.getenv("MODEL_URI", "models:/Reggaeton_Classifier_V1.0/latest")
logger.info(f"MLFlow MODEL URI: {MODEL_URI}")

# Configurar SlowAPI para limitar el tráfico
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.on_event("startup")
async def startup_event():
    """
    Precarga los modelos para evitar que se carguen en la primera petición.

    Esta función se ejecuta al iniciar la aplicación y se encarga de cargar:
    1. Un modelo de MLflow desde una URI especificada.
    2. Un MinMaxScaler desde un archivo especificado.
    
    Además, mide y registra el tiempo total que se tarda en cargar ambos modelos.
    """
    start_time = dt.datetime.now()
    logger.info("Iniciando la carga de los modelos...")

    try:
        # Cargar el modelo de MLflow
        model_instance = ModelSingleton(MODEL_URI)
        logger.info(f"Modelo cargado correctamente desde {MODEL_URI}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo desde {MODEL_URI}: {e}")
        settings.flag_run = False
        settings.msg = "Can't load MLflow Model"

    try:
        # Cargar el MinMaxScaler
        tempo_scaler = ModelSingleton('tempo_min_max_scaler.save')
        logger.info("MinMaxScaler cargado correctamente desde 'tempo_min_max_scaler.save'")
    except Exception as e:
        logger.error(f"Error al cargar el MinMaxScaler: {e}")
        settings.flag_run = False
        settings.msg = "Can't load MLflow Model"

    finish_time = dt.datetime.now()
    time_took = (finish_time - start_time).total_seconds()

    logger.info(f"Se demoró {round(time_took, 2)} segundos en startup_event")

@limiter.limit("10/minute")
@app.get("/", response_model=RootResponse, status_code=status.HTTP_200_OK)
async def root(request: Request) -> RootResponse:
    """Endpoint for testing the connection with this API"""
    response = RootResponse(
        app=f"{settings.app_name} v{settings.version}",
        status="Online" if settings.flag_run else "Offline",
        message=settings.msg
    )
    # Si logro cargar el modelo, se muestra el mensaje de conexión correcta
    if settings.flag_run:
        return JSONResponse(content=response.model_dump(), status_code=200)
    else:
        return JSONResponse(content=response.model_dump(), status_code=500)


@limiter.limit("60/minute")
@app.post("/predict/", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: Request, feature: Feature, transform: bool = True, apikey: str = "") -> PredictionResponse:
    """Endpoint for using the MLFlow model, set transform to False only for testing purposes"""
    
    # Se captura el tiempo de inicio para calcular el tiempo total
    start_time = dt.datetime.now()

    # Por falta de tiempo, solo se usará una apikey fija, se recomienda añadir 
    # un sistema de tokens o administrador de credenciales para evitar el uso 
    # no deseado de la API.
    if apikey != "kXwagyJkMH0Q3fT3MorqmRupqpjq1FCsVijy2P3nwrQQpExcWl":
        # Se captura el tiempo de finalización para calcular el tiempo total
        finish_time = dt.datetime.now()
        time_took = (finish_time - start_time).total_seconds()
        # Creo un objeto de respuesta con el mensaje de error
        response = PredictionResponse(
            predictions=None, 
            id=None, 
            message="Your API credentials are incorrect", 
            time_took=time_took
        )
        return JSONResponse(response.model_dump(), status_code=401)
    try:
        model_instance = ModelSingleton(MODEL_URI)

        # Primero escalamos los datos recibidos
        if transform:
            tempo_scaler_file = 'tempo_min_max_scaler.save'
            tempo_scaler = ModelSingleton(tempo_scaler_file)
            scaled_tempo = tempo_scaler.model.transform([[feature.tempo]])[0][0]
            feature.tempo = float(scaled_tempo)

        # Ahora transformamos a float32 para evitar problemas al usar el modelo
        data = feature.dict()
        for key, val in data.items():
            data[key] = np.float32(val)

        predictions = model_instance.model.predict([data])[0]
        id2name = {
            0: "Is not Reggaeton",
            1: "Is Reggaeton"
        }
        # Se captura el tiempo de finalización para calcular el tiempo total
        finish_time = dt.datetime.now()
        time_took = (finish_time - start_time).total_seconds()
        # Creo un objeto de respuesta con mensaje de respuesta correcta
        response = PredictionResponse(
            predictions=id2name[predictions], 
            id=int(predictions), 
            message='Ok', 
            time_took=time_took
        )
        return JSONResponse(response.model_dump(), status_code=200)  # Cambié el código de estado a 200
    except Exception as e:
        settings.flag_run = False
        settings.msg = f"Found a unexpected error, {str(r)}"
        # Se captura el tiempo de finalización para calcular el tiempo total
        finish_time = dt.datetime.now()
        time_took = (finish_time - start_time).total_seconds()
        # Creo un objeto de respuesta con el mensaje de error
        response = PredictionResponse(
            predictions=None, 
            id=None, 
            message=str(e), 
            time_took=time_took
        )
        return JSONResponse(response.model_dump(), status_code=500)
