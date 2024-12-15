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
from keras.utils import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Step 3: Define the BiLSTM Autoencoder Model
    input_dim = 4  # Number of features

    # Create the model
    model = keras.Sequential([
        layers.Input(shape=(1, input_dim)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.RepeatVector(1),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.TimeDistributed(layers.Dense(input_dim))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

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
        server_address="0.0.0.0:8181",
        config=fl.server.ServerConfig(num_rounds=200),
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

    # Load the dataset
    df = pd.read_excel("datasets\\mis_iris_data.xlsx")
    new_data = df.copy()
    df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]  # Simplify column names
    # #Step 1, 
    # # Fill NaN in the first row of each column with the first non-NaN value from below
    # for col in df.columns:
    #     if pd.isna(df.loc[0, col]):
    #         first_non_nan = df[col].loc[1:].first_valid_index()  # Finds the first non-NaN index below the first row
    #         if first_non_nan is not None:
    #             df.loc[0, col] = df.loc[first_non_nan, col]
    # # Fill NaN in the last row of each column with the last non-NaN value from above
    # for col in df.columns:
    #     if pd.isna(df.loc[df.index[-1], col]):
    #         last_non_nan = df[col].iloc[:-1].last_valid_index()  # Finds the last non-NaN index above the last row
    #         if last_non_nan is not None:
    #             df.loc[df.index[-1], col] = df.loc[last_non_nan, col]

    # new_data = df.copy()
    def compute_min_gap(column_data, row_idx):
        """
        Finds the smallest gap (in row indices) from `row_idx` to a non-NaN value in `column_data`.
        Returns the index of the closest non-NaN value relative to `row_idx`.
        """
        # Get non-NaN values
        non_nan_indices = column_data.index[column_data.notna()].tolist()
        
        if not non_nan_indices:
            return np.nan

        # Find the closest index
        min_gap_index = min(non_nan_indices, key=lambda x: abs(x - row_idx))
        
        return min_gap_index - row_idx

    # Identify rows that are entirely NaN
    empty_row_index = df[df.isna().all(axis=1)].index.tolist()

    # Initialize a DataFrame for storing gaps
    gap = pd.DataFrame(index=empty_row_index, columns=df.columns)

    # Compute the gap for each empty row
    for row_idx in empty_row_index:
        for col in df.columns:
            gap.loc[row_idx, col] = compute_min_gap(df[col], row_idx)

    # Ensure all columns in the DataFrame are numeric before performing operations like idxmin
    abs_gap = gap.abs().apply(pd.to_numeric, errors='coerce')
    min_gap_indices = abs_gap.idxmin(axis=1)

    # Fill empty rows based on the calculated minimum gap
    for row_idx in empty_row_index:
        col_idx = min_gap_indices[row_idx]  # Column with the smallest gap for this empty row
        line_idx = row_idx + gap.loc[row_idx, col_idx]

        # Check if line_index is within bounds before assignment
        if 0 <= line_idx < df.shape[0]:
            df.at[row_idx, col_idx] = df.at[line_idx, col_idx]

    # Step 1: Compute pairwise correlations with missing data
    correlations = df.corr(method='pearson', min_periods=1)

    # Step 2: Sort each feature's correlations individually (highest to lowest)
    sorted_feature_correlations = {col: correlations[col].abs().sort_values(ascending=False).index.tolist() 
                                for col in correlations.columns}

    # Step 3: Reorder the DataFrame columns for each feature based on sorted correlations
    reordered_data = {feature: df[sorted_columns] for feature, sorted_columns in sorted_feature_correlations.items()}   

    # Step 4: Compute mean of reordered data (used later in data fusion step)
    mean_values = {col: df[col].mean(skipna=True) for col in df.columns}
    # mean_values = {feature: reordered_df.mean() for feature, reordered_df in reordered_data.items()}

    # Step 4: Impute missing values in the first and last rows within each DataFrame in reordered_data
    for feature, reordered_df in reordered_data.items():
        # Use the average for the feature as the target imputation value
        target_mean = mean_values[feature]

        # Iterate over each column in the reordered DataFrame for this feature
        for col in reordered_df.columns:
            # Get the mean for the current reference column
            reference_mean = mean_values[col]

            # Check the first row for NaNs in the current column
            if pd.isna(reordered_df.at[0, col]):
                if pd.notna(reordered_df.at[0, feature]):
                    reordered_df.at[0, col] = (reordered_df.at[0, feature] / target_mean) * reference_mean
                else:
                    # Fallback to nearest non-NaN value if the direct reference is NaN
                    nearest_non_nan = reordered_df[feature].dropna().iloc[0] if not reordered_df[feature].dropna().empty else np.nan
                    if nearest_non_nan is not np.nan:
                        reordered_df.at[0, col] = (nearest_non_nan / target_mean) * reference_mean

            # Check the last row for NaNs in the current column
            last_row = reordered_df.index[-1]
            if pd.isna(reordered_df.at[last_row, col]):
                if pd.notna(reordered_df.at[last_row, feature]):
                    reordered_df.at[last_row, col] = (reordered_df.at[last_row, feature] / target_mean) * reference_mean
                else:
                    # Fallback to nearest non-NaN value if the direct reference is NaN
                    nearest_non_nan = reordered_df[feature].dropna().iloc[-1] if not reordered_df[feature].dropna().empty else np.nan
                    if nearest_non_nan is not np.nan:
                        reordered_df.at[last_row, col] = (nearest_non_nan / target_mean) * reference_mean

        # Update reordered_data with the fully imputed DataFrame for the feature
        reordered_data[feature] = reordered_df
    # Step 5: Data fusion function
    def data_fusion(reordered_df, mean_values):
        fused_data = reordered_df.copy()
        for i in range(len(fused_data)):
            dependent_value = np.nan
            for col in fused_data.columns:
                if not np.isnan(fused_data.loc[i, col]):
                    dependent_value = fused_data.loc[i, col]
                    break
            if not np.isnan(dependent_value):
                for col in fused_data.columns:
                    if col != fused_data.columns[0] and not np.isnan(fused_data.loc[i, col]):
                        fused_data.loc[i, col] = (
                            dependent_value / fused_data.loc[i, col]
                        ) * (mean_values[col] / mean_values[fused_data.columns[0]])
        return fused_data
    mean_values = {feature: reordered_df.mean() for feature, reordered_df in reordered_data.items()}
    # Step 6: Apply data fusion
    fused_results = {feature: data_fusion(reordered_df, mean_values[feature])
                    for feature, reordered_df in reordered_data.items()}
    # Step 7: Optimized Linear Interpolation
    def fast_linear_interpolation(data):
        interpolated_data = data.copy()
        for col in range(interpolated_data.shape[1]):
            column_data = interpolated_data[:, col]
            nans = np.isnan(column_data)
            valid = ~nans
            valid_indices = np.where(valid)[0]
            
            if nans.any() and valid.any():
                interpolated_data[nans, col] = np.interp(
                    np.flatnonzero(nans),
                    valid_indices,
                    column_data[valid]
                )
        
        return interpolated_data

    # Step 8: Apply optimized linear interpolation to the fused results
    interpolated_fused_results = {
        feature: pd.DataFrame(fast_linear_interpolation(fused_results[feature].values), 
                            columns=fused_results[feature].columns)
        for feature in fused_results
    }

    # Step 9: Define function to compute adjustment matrix based on correlations
    def compute_adjustment_matrix(correlations):
        def get_adjustment_value(p):
            if 0 <= abs(p) <= 0.2:
                return 0.5
            elif 0.2 < abs(p) <= 0.4:
                return 0.6
            elif 0.4 < abs(p) <= 0.6:
                return 0.7
            elif 0.6 < abs(p) <= 0.8:
                return 0.8
            elif 0.8 < abs(p) <= 1.0:
                return 0.9
            return 1.0

        num_features = correlations.shape[0]
        adjustment_matrix = np.zeros((num_features, num_features))

        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    adjustment_matrix[i, j] = get_adjustment_value(correlations.iloc[i, j])
                else:
                    adjustment_matrix[i, j] = 1  # Identity for self-correlation

        return adjustment_matrix

    # Compute the adjustment matrix
    adjustment_matrix = compute_adjustment_matrix(correlations)

    # Step 10: Define the data recovery function with value capping
    def recover_data(fused_data, reordered_df, mean_values, adjustment_matrix, original_min, original_max):
        row, column = fused_data.shape
        recovered_data = np.full((row, column), np.nan)
        
        for i in range(row):
            for j in range(column):
                if not np.isnan(reordered_df.values[i][j]):
                    recovered_data[i][j] = reordered_df.values[i][j]
                else:
                    for k in range(column):
                        if k != j and not np.isnan(reordered_df.values[i][k]):
                            recovered_value = (
                                fused_data[i][j] * reordered_df.values[i][k] * 
                                mean_values[j] / mean_values[k] * adjustment_matrix[j][k]
                            )
                            recovered_data[i][j] = np.clip(recovered_value, original_min[j], original_max[j])
                            break
        return recovered_data

    # Step 11: Compute original min and max values for each feature
    original_min = df.min().values
    original_max = df.max().values

    # Step 12: Apply data recovery with value capping
    recovered_results = {
        feature: pd.DataFrame(
            recover_data(interpolated_fused_results[feature].values, reordered_df, 
                        mean_values[feature], adjustment_matrix, original_min, original_max), 
            columns=reordered_df.columns
        )
        for feature, reordered_df in reordered_data.items()
    }

    # Step 1: Scale each feature in recovered_results using MinMaxScaler
    scaled_recovered_results = {}
    scalers = {}  # Dictionary to store scalers for each feature to use on test data later

    for feature, recovered_df in recovered_results.items():
        scaler = MinMaxScaler()
        scaled_recovered_results[feature] = pd.DataFrame(
            scaler.fit_transform(recovered_df),
            columns=recovered_df.columns,
            index=recovered_df.index
        )
        scalers[feature] = scaler  # Store the scaler for this feature


    # Step 13: Split each DataFrame in reordered_data into training and testing based on NaNs
    train_test_data = {}

    for feature, reordered_df in reordered_data.items():
        train_test_data[feature] = {}

        for col in reordered_df.columns:
            # Training Data: Rows with observed (non-NaN) values in the current column
            train_data = reordered_df[reordered_df[col].notna()]
            train_indices = train_data.index
            
            # Testing Data: Rows with NaN values in the current column
            test_data = reordered_df[reordered_df[col].isna()]
            test_indices = test_data.index
            
            train_test_data[feature][col] = {
                'train': train_data.reset_index(drop=True),
                'train_indices': train_indices,
                'test': test_data.reset_index(drop=True),
                'test_indices': test_indices
            }

    # Initialize a dictionary to store the final train and test data from recovered_results
    final_train_test_data = {}

    for feature, feature_data in train_test_data.items():
        final_train_test_data[feature] = {}

        # Retrieve the recovered DataFrame for the current feature
        recovered_df = recovered_results[feature]

        for col, split_indices in feature_data.items():
            # Extract train and test indices for the current column from train_test_data
            train_indices = split_indices['train_indices']
            test_indices = split_indices['test_indices']
            
            # Select the training and testing data from recovered_df using the exact indices from train_test_data
            train_data = recovered_df.loc[train_indices]
            test_data = recovered_df.loc[test_indices]
            
            # Store the train and test data for each column in final_train_test_data without altering indices
            final_train_test_data[feature][col] = {
                'train': train_data,
                'train_indices': train_indices,
                'test': test_data,
                'test_indices': test_indices
            }

    # Step 2: Combine all scaled training data from final_train_test_data into a single DataFrame
    all_train_data = []

    for feature, feature_data in final_train_test_data.items():
        for col, data in feature_data.items():
            # Scale the training data using the corresponding scaler
            train_data = scalers[feature].transform(data['train'])
            all_train_data.append(pd.DataFrame(train_data, columns=data['train'].columns))

    # Concatenate all training data for BiLSTM training
    combined_train_data = pd.concat(all_train_data, axis=0).reset_index(drop=True)

    # Reshape for LSTM input [samples, time steps, features]
    X_train = combined_train_data.values.reshape((combined_train_data.shape[0], 1, combined_train_data.shape[1]))

    # Step 5: Prepare the test data for BiLSTM prediction
    all_test_data = []

    for feature, feature_data in final_train_test_data.items():
        for col, data in feature_data.items():
            # Scale the test data using the corresponding scaler
            test_data = scalers[feature].transform(data['test'])
            all_test_data.append(pd.DataFrame(test_data, columns=data['test'].columns))

    # Concatenate all scaled test data for BiLSTM prediction
    combined_test_data = pd.concat(all_test_data, axis=0).reset_index(drop=True)

    # Reshape for LSTM input [samples, time steps, features]
    X_test = combined_test_data.values.reshape((combined_test_data.shape[0], 1, combined_test_data.shape[1]))


    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]


    # Calculate the dynamic end index based on the length of the training data

    dynamic_end_index = len(X_test) - 10

    x_val, y_val = X_test[dynamic_end_index:], X_test[dynamic_end_index:]  # make the last part of the code dynamic
 



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
