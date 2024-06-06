import os

from fastapi.testclient import TestClient
import mlflow.pyfunc
import pandas as pd

from main import app

# Cliente para hacer las pruebas                                                                                                                                                                                                
client = TestClient(app)


def test_root():
    """Probar si la api pudo prender
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Online"

# def test_inference_with_transform():
#     response = client.get("/")
#     assert response.status_code == 200
#     assert response.json()["status"] == "Online"