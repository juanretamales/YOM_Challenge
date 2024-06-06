import os
import sys
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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