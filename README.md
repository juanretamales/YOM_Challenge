# YOM Challenge

El objetivo de esta prueba es entender tus habilidades para tomar un modelo hecho en local por un Data Scientist y dejarlo en un ambiente simulado de producción.

Los aspectos que evaluaremos son:
- Calidad del código (refactorización, orden, lógica, uso de patrones de diseño, etc)
- Capacidad de implementar herramientas que permitan aplicar los principios de ML-Ops al proyecto en producción (recomendamos Neptune o ML Flow)
- Uso de alguna herramienta que permita contener el ambiente creado para la ejecución y hacerlo reproducible fácilmente para alguien que quiera ejecutarlo
- Documentación clara y fácil de leer

Tienes bonus si:
- Específicas cómo vas a abordar la interacción del modelo con el mundo real y nos explicas cómo vas a prevenir la degradación del modelo
- Creas mecanismos de monitoreo que permitan alertar cuando exista un deterioro del modelo
- Usas herramientas de automatización de creación de infraestructura

La prueba consiste en tomar [este challenge realizado para una prueba técnica](https://github.com/iair/Spike_Challenge) hace varios años y modificarlo para que cumpla con los requerimientos presentados anteriormente.
[El objetivo del challenge inicial está especificado aquí](https://github.com/iair/Spike_Challenge/blob/master/Desaf%C3%ADo%20Spotify%20Reggaeton.pdf), pero recuerda que tu objetivo no es resolver el problema de modelamiento sino que tomar lo que hizo este DS junior y dejarlo en algo que pueda ser pasado a producción.

# 1. Contexto

Se busca poner en producción un modelo de sklearn que predice si una canción es de regaeton o no.

## 1.1 Patrón

El patrón Singleton y el patrón Flyweight tienen propósitos diferentes y se usan en contextos distintos. A continuación, comparo ambos patrones para ver cuál se adapta mejor a tu caso:

### 1.1.1 Singleton:

- Propósito: Garantizar que una clase tenga una única instancia y proporcionar un punto de acceso global a esa instancia.
- Uso: Es adecuado para gestionar recursos que deben ser únicos en el sistema, como la carga de un modelo de machine learning que solo necesita ser cargado una vez y reutilizado en todas las solicitudes.

### 1.1.2 Flyweight:

- Propósito: Reducir el uso de memoria compartiendo la mayor cantidad de datos posible con objetos similares.
- Uso: Es adecuado cuando se necesita manejar una gran cantidad de objetos similares que comparten datos comunes, lo cual no es exactamente el caso aquí.

### 1.1.3 En resumen
Para desplegar un modelo de machine learning, el Singleton parece más apropiado por las siguientes razones:

- Única Instancia del Modelo: El modelo de machine learning debe cargarse una sola vez y reutilizarse, evitando recargas innecesarias que pueden ser costosas en términos de tiempo y recursos.
- Acceso Global: El Singleton proporciona un punto de acceso global a la instancia del modelo, lo cual es ideal para servir predicciones a múltiples solicitudes concurrentes.
- Mantenimiento: El uso de Singleton puede ser más apropiado en aplicaciones más grandes y complejas donde el control y la encapsulación son cruciales. 

## 1.2 Arquitectura

```mermaid
flowchart LR
%% Nodes
    A("<i class="fa-solid fa-globe"></i> Internet")
    B("fa:fa-code FastAPI")
    C("fa:fa-shapes MLflow")

%% Edge connections between nodes
    A -- Consume la API --> B
    B -- Obtiene el modelo --> C

%% Individual node styling. Try the visual editor toolbar for easier styling!
    style A color:#FFFFFF, fill:#AA00FF, stroke:#AA00FF
    style B color:#FFFFFF, stroke:#00C853, fill:#00C853
    style C color:#FFFFFF, stroke:#2962FF, fill:#2962FF
```

### 1.2.1 MLflow para Tracking
Se decidió usar MLflow solo para tracking y no para el despliegue debido a:

- Propósito Específico: MLflow está diseñado específicamente para el seguimiento y gestión del ciclo de vida de los experimentos de machine learning, incluyendo la experimentación, la reproducción de resultados y la comparación de modelos.
- Versionado y Registro de Modelos: MLflow facilita el registro, versionado y almacenamiento de modelos, además de capturar métricas, parámetros y artefactos asociados a cada experimento.
- Interfaz de Usuario: Ofrece una interfaz web intuitiva que permite visualizar y comparar experimentos de manera sencilla.    
- Uso Interno: MLflow está diseñado principalmente para ser utilizado en entornos internos durante el desarrollo y experimentación de modelos. Por lo tanto, no está optimizado para el control de acceso granular y la exposición a clientes externos.
- Enfoque en Tracking: Mientras que MLflow es excelente para gestionar el ciclo de vida de los experimentos, no está optimizado para manejar solicitudes de alta frecuencia y baja latencia, como las que se esperan de una API en producción.
- Carga y Recursos: Su infraestructura no está diseñada para soportar cargas intensivas de tráfico externo.

### 1.2.2 FastAPI para Despliegue

Se decidio usar FastAPI para el despliegue debido a:

- Desempeño y Escalabilidad: FastAPI es conocido por su alto rendimiento y capacidad de manejar muchas solicitudes concurrentes, lo que es crucial para la fase de despliegue de un modelo.
- Flexibilidad: Permite crear APIs RESTful rápidas y eficientes, integrándose fácilmente con otras tecnologías y servicios.
- Facilidad de Uso: Ofrece una sintaxis sencilla y soporta la generación automática de documentación de API, lo 
que facilita el mantenimiento y la colaboración.    
- Autenticación y Autorización: FastAPI permite la implementación de diversas estrategias de autenticación (OAuth2, JWT, etc.) y autorización, proporcionando un control de acceso robusto y seguro.
- Configuración de Seguridad: Es más adecuado para configurar certificados SSL/TLS y gestionar conexiones seguras, algo crucial cuando se expone una API a usuarios externos.
- Asincronía y Concurrencia: FastAPI utiliza ASGI (Asynchronous Server Gateway Interface) para manejar solicitudes de manera asíncrona, permitiendo un manejo eficiente de múltiples conexiones concurrentes.
- Despliegue y Escalado: Es más sencillo desplegar FastAPI en servicios de orquestación y escalado automático como ElasticBeanstalk, lo que garantiza alta disponibilidad y balanceo de carga.
- Se uso la biblioteca Slowapi para gestionar el uso de la API y prevenir posibles ataques.

# 2. Instalación

## Requisitos Previos
1. Tener Docker y Docker Compose instalados en tu sistema.
2. Clonar el repositorio de tu proyecto.

## Pasos de Instalación
1. Iniciar MLflow con Docker Compose
   1. Navega al directorio raíz de tu proyecto donde se encuentra el archivo docker-compose.yml.
   2. Ejecuta el siguiente comando para iniciar MLflow en segundo plano: ```docker-compose up mlflow -d```
2. Entrenar el Modelo y Enviar a MLflow
   1. Abre el archivo train_model_and_send_mlflow.ipynb que se encuentra en la carpeta app.
   2. Sigue las instrucciones dentro del notebook para entrenar tu modelo. Esto generará el archivo tempo_min_max_scaler.save y creará un experimento dentro de MLflow.
3. Registrar el Modelo en MLflow
   1. Abre MLflow en tu navegador accediendo a http://127.0.0.1:5000.
   2. Dentro de la interfaz de MLflow, registra el modelo bajo el nombre Reggaeton_Classifier_V1.0. Si decides cambiar el nombre del modelo, asegúrate de actualizar los códigos correspondientes en los archivos de tu proyecto para reflejar este cambio. Si sabes lo que haces, puedes solo intentar cambiar la variable de entorno MODEL_URI en el archivo .env o docker-compose.yml.
4. Iniciar la Aplicación FastAPI
   1. Puedes iniciar tu aplicación de FastAPI de dos maneras: usando Docker o de manera local.
   2. Por falta de tiempo, se uso una apikey fija la cual es ```kXwagyJkMH0Q3fT3MorqmRupqpjq1FCsVijy2P3nwrQQpExcWl``` y esta en en el endpoint predict, ya que se pretende evitar el uso no deseado de la API.

Nota: si usas MLflow en otro servidor, debes cambiar la variable de entorno MLFLOW_URI en el archivo .env o docker-compose.yml.

### Opción 1: Usar Docker
1. En el directorio raíz de tu proyecto, ejecuta el siguiente comando para iniciar la API en el puerto 8001: ```docker-compose up api -d```
2. Una vez que la aplicación esté en funcionamiento, puedes acceder a ella navegando a http://127.0.0.1:8001 en tu navegador.
### Opción 2: Iniciar de Manera Local
1. Crea una variable de entorno si es necesario. Por ejemplo, puedes crear un archivo .env en el directorio raíz de tu proyecto y definir tus variables de entorno allí.
2. Instala las dependencias de tu proyecto ejecutando: ```pip install -r requirements.txt```
3. Inicia la aplicación FastAPI ejecutando: ```uvicorn app.main:app --host 0.0.0.0 --port 8001```
4. Una vez que la aplicación esté en funcionamiento, puedes acceder a ella navegando a http://127.0.0.1:8001 en tu navegador.

# 3. Pruebas
Para iniciar todas las pruebas, abrir terminal dentro de app y ejecutar el siguiente comando:
```bash
pytest
```
Los archivos como modelos y dataset, estan en el archivo raiz para evitar problemas al momento de leer archivos.

1. Archivo: test/test_main.py
Este archivo probará el funcionamiento general de tu API.

1. Archivo: test/test_inference.py
Este archivo probará la inferencia del modelo.

1. Archivo: test/test_degrade_drift.py
Este archivo probará la degradación del modelo con un dataset como nueva_data.csv.

# 4. Proximos pasos
- Implementar un Reverse Proxy (Puede ser con NGINX y CertBot) para que la API tengo uso de SSL y que se pueda acceder desde cualquier lugar con conexion segura.
- Decidir si usar MLflow para almacenar los modelos o usar un servicio de almacenamiento de modelos como Hugging Face o S3, este ultimo para no requerir el servidor de MLFlow para iniciar la API.
- Añadir mas pruebas, como IsolationForest que se propuso pero no se implementó dentro del flujo de entrenamiento.
- Ver la posibilidad de que el MinMaxScaler tambien este junto al modelo en MLFlow para que este versionado.
- Añadir un sistema de tokens o administrador de credenciales para evitar el uso no deseado de la API.

# 5. Referencias
- https://medium.com/@dast04/running-airflow-with-docker-compose-2023-for-machine-learning-a78eeadc00cd
- https://github.com/sachua/mlflow-docker-compose/tree/master/mlflow
- https://anderfernandez.com/blog/tutorial-mlflow-completo/