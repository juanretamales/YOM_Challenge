from pydantic import BaseModel

class Feature(BaseModel):
    danceability:float
    energy:float
    speechiness:float
    acousticness:float
    valence:float
    tempo:float
    # loudness:float

class FeatureWithTrue(BaseModel):
    danceability:float
    energy:float
    speechiness:float
    acousticness:float
    valence:float
    tempo:float
    reggaeton:float

