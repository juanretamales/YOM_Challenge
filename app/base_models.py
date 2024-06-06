from pydantic import BaseModel
from typing import Optional

class Feature(BaseModel):
    danceability:float
    energy:float
    speechiness:float
    acousticness:float
    valence:float
    tempo:float

    # Se usa de esta forma para que lo tome FastAPI
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "danceability": 0.512,
                    "energy": 0.541,
                    "speechiness": 0.141,
                    "acousticness": 0.781,
                    "valence": 0.941,
                    "tempo": 125.513
                }
            ]
        }
    }

class RootResponse(BaseModel):
    app:str
    status:str
    message:str

    # Se usa de esta forma para que lo tome FastAPI
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "app": "Reggaeton Classifier API v1.0.0",
                    "status": "Online",
                    "message": "Start on 2024-06-05 12:26:36.471321"
                }
            ]
        }
    }

class PredictionResponse (BaseModel):
    predictions: Optional[str] = None
    id: Optional[int] = None
    message: Optional[str] = None
    time_took: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predictions": "Is not Reggaeton",
                    "id": 0,
                    "message": "Ok",
                    "time_took": 0.001000
                }
            ]
        }
    }