import argparse
from datetime import datetime
import os
from pathlib import Path

import tensorflow as tf
from datasets import Dataset

import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, log_dir):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.log_dir = log_dir

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Create a TensoBoard callback with a unique log directory for each client
        log_dir_client = f"{self.log_dir}/client_{os.getpid()}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_client, histogram_freq=1)

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
            callbacks=[tensorboard_callback],  # Add the TensorBoard callback
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition of dataset to be used. "
        "Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. "
        "Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    # Load and compile Keras model
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    '''
    # Model with only GRU layer
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(units=128, activation='relu', input_shape=(1, 15)),
        tf.keras.layers.Dense(units=2, activation='linear')  
    ])

    # Model with only Bidirectional LSTM layer
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=False), input_shape=(1, 15)),
        tf.keras.layers.Dense(units=2, activation='linear')  
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=512, return_sequences=True, input_shape=(1, 15)),
        tf.keras.layers.LSTM(units=128, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='linear')
    ])
 '''

    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True), input_shape=(1, 23)),
        tf.keras.layers.GRU(units=128, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='linear')  
    ]) 

    model.compile("adam", "mean_squared_error", metrics=["accuracy"])
    #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])


    # Load a subset of dataset to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    if args.toy:
        x_train, y_train = x_train[:10], y_train[:10]
        x_test, y_test = x_test[:10], y_test[:10]

    # Start Flower client
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    client = CifarClient(model, x_train, y_train, x_test, y_test, log_dir)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    # Read the entire dataset
    df = pd.read_csv("preprocessed_data.csv")

    # Create features, labels, and client_ids from your preprocessed dataset
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[[ 'resource_request_cpus', 'resource_request_memory',  'poly_maximum_usage_cpus random_sample_usage_cpus', 
                                                'maximum_usage_cpus',  'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
                                                'maximum_usage_memory',  'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
                                                'random_sample_usage_cpus', 'assigned_memory',  'poly_maximum_usage_cpus', 'memory_demand_rolling_std', 
                                                'start_hour', 'start_dayofweek', 'duration_seconds', 'sample_rate', 'cycles_per_instruction', 
                                                'memory_accesses_per_instruction', 'page_cache_memory', 'priority',
                                            ]])
    labels = df[['average_usage_cpus', 'average_usage_memory']]


    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Convert NumPy arrays back to TensorFlow tensors
    x_train = tf.constant(x_train, dtype=tf.float32)
    x_test = tf.constant(x_test, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)

    # Reshape the input features to add a third dimension for time steps
    x_train_reshaped = tf.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test_reshaped = tf.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    return (
        x_train_reshaped[idx * 30000 : (idx + 1) * 30000],
        y_train[idx * 30000 : (idx + 1) * 30000],
    ), (
        x_test_reshaped[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )


if __name__ == "__main__":
    main()
