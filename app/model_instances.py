import os
import joblib
import mlflow.pyfunc
from threading import Lock

class ModelSingleton:
    """
    Clase Singleton para cargar y mantener modelos en memoria.
    
    Esta clase asegura que sólo una instancia de cada modelo sea creada y reutilizada,
    ya sea que el modelo provenga de MLflow o de una ruta de archivo específica.
    """
    _instances = {}
    _lock = Lock()

    def __new__(cls, model_path):
        """
        Método de creación de una nueva instancia, siguiendo el patrón Singleton por modelo.
        
        Args:
            model_path (str): La ruta del modelo o el URI de MLflow.
            
        Returns:
            ModelSingleton: La instancia única de la clase para el modelo especificado.
        """
        with cls._lock:
            if model_path not in cls._instances:
                cls._instances[model_path] = super().__new__(cls)
                cls._instances[model_path]._model = cls._load_model(model_path)
        return cls._instances[model_path]

    @staticmethod
    def _load_model(model_path):
        """
        Carga el modelo desde MLflow o desde una ruta de archivo local.
        
        Args:
            model_path (str): La ruta del modelo o el URI de MLflow.
        
        Returns:
            Model: El modelo cargado.
            
        Raises:
            ValueError: Si el archivo no existe en la ruta especificada.
        """
        if 'models:/' in model_path:  # Es un modelo de MLflow
            return mlflow.pyfunc.load_model(model_path)
        else:  # Es un modelo en un archivo local
            if os.path.isfile(model_path):
                return joblib.load(model_path)
            else:
                raise ValueError(f"El archivo no existe en {model_path}")

    @property
    def model(self):
        """
        Propiedad para acceder al modelo cargado.
        
        Returns:
            Model: El modelo cargado.
        """
        return self._model
