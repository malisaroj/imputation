from typing import Dict, Optional, Tuple
from pathlib import Path
import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
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

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for 100 rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )

    # Save the trained model after the training is completed
    model_save_path = Path(".cache") / "trained_model.h5"

    # Check if the model file already exists, and replace it if necessary
    if model_save_path.exists():
        print("A trained model already exists. Replacing it.")
        try:
            os.remove(model_save_path)
        except PermissionError as e:
            print(f"Error removing existing model file: {e}")
            # Handle the error as needed, e.g., by renaming the existing file
            # or prompting the user for action.
            # Example: os.rename(model_save_path, 'backup_model')
    else:
        print("No existing model file found.")

    # Save the new model
    model.save(os.path.join(model_save_path, "trained_model.h5"))

    # Plot the metrics
    plot_metrics(eval_loss, eval_accuracy)


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

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


    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]


    # Calculate the dynamic end index based on the length of the training data
    # Use the last 5k training examples as a validation set

    dynamic_end_index = len(x_train_reshaped) - 5000

    x_val, y_val = x_train_reshaped[dynamic_end_index:], y_train[dynamic_end_index:]  # make the last part of the code dynamic


    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        eval_loss.append(loss)
        eval_accuracy.append(accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 20,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 20
    return {"val_steps": val_steps}

def plot_metrics(eval_loss, eval_accuracy):
    rounds = np.arange(1, len(eval_loss) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, eval_loss)
    plt.xlabel("Round")
    plt.ylabel("Evaluation Loss")
    plt.title("Evaluation Loss over Rounds")

    plt.subplot(1, 2, 2)
    plt.plot(rounds, eval_accuracy)
    plt.xlabel("Round")
    plt.ylabel("Evaluation Accuracy")
    plt.title("Evaluation Accuracy over Rounds")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize lists to store loss and accuracy
    eval_loss = []
    eval_accuracy = []
    main()
