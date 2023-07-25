# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add current directory files to /app in the container
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip &&\
    pip install numpy pandas scikit-learn mlflow boto3 tensorflow

# Run cat-dog.py when the container launches
ENTRYPOINT ["python", "mnist-digit-k8s.py"]
