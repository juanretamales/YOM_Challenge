FROM python:3.10-slim-buster

# SET THE WORKING DIRECTORY
WORKDIR /app

# COPY THE CODE FROM THE APP FOLDER TO THE WORKING DIRECTORY
COPY . /app

# Update the system
RUN apt-get update

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the server on port 15400
EXPOSE 15400

# Command to run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "15400","--timeout-keep-alive","5"]