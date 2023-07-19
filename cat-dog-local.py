import os
import warnings
import sys

import mlflow
from mlflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

# Input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

def load_data():
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test), input_shape

def build_model(input_shape, loss='sparse_categorical_crossentropy', optimizer='adam'):
    # Build model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # Set the tracking URI to point to your MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set the endpoint URL to point to your MinIO server
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://localhost:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    os.environ['AWS_CA_BUNDLE'] = ''

    # Load data
    (x_train, y_train), (x_test, y_test), input_shape = load_data()

    # Number of epochs to train the model
    n_epochs = 5 if len(sys.argv) > 1 else 5

    # Loss function and optimizer for the model
    loss = sys.argv[2] if len(sys.argv) > 2 else 'sparse_categorical_crossentropy'
    optimizer = sys.argv[3] if len(sys.argv) > 3 else 'adam'

    with mlflow.start_run():
        # Build and train model
        model = build_model(input_shape, loss=loss, optimizer=optimizer)
        model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=n_epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        # Evaluate model
        score = model.evaluate(x_test, y_test, verbose=0)

        # Log metrics
        mlflow.log_metric("test_loss", score[0])
        mlflow.log_metric("test_accuracy", score[1])

        # Log model
        mlflow.keras.log_model(model, "model", registered_model_name="MnistModel-local")
