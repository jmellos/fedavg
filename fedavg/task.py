"""fedavg: A Flower / TensorFlow app."""

import os

import keras
from keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configura o logger para mostrar apenas mensagens de erro ou acima
logging.basicConfig(level=logging.ERROR)



# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(input_size, hidden_size, output_size, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=(None, input_size)),  # Define the input shape using an Input layer
        layers.LSTM(hidden_size, return_sequences=False),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(output_size)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Map partition_id to the corresponding client CSV file
    client_files = {
        0: "Residential_4.csv",
        1: "Residential_8.csv",
        2: "Residential_9.csv",
        3: "Residential_10.csv",
        4: "Residential_13.csv",
    }

    # Ensure the partition_id is valid
    if partition_id not in client_files:
        raise ValueError(f"Invalid partition_id: {partition_id}. Valid IDs are {list(client_files.keys())}")

    # Load the CSV file for the corresponding client
    csv_file_path = client_files[partition_id]
    data = pd.read_csv(csv_file_path)

    # Preprocess the data for LSTM
    sequence_length = 24  # Example: Use 24-hour sequences
    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data['energy_kWh'][i:i+sequence_length].values)
        y.append(data['energy_kWh'][i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to be suitable for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test
