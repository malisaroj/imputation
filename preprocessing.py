from pandas import json_normalize
import ast
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import os
from sklearn.preprocessing import LabelEncoder


# Load your dataset (assuming the dataset is in a pandas DataFrame)
df = pd.read_csv("datasets/borg_traces_data.csv")

column_to_encode = 'user'

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the column with label encoding
df[column_to_encode] = label_encoder.fit_transform(df[column_to_encode])


columns_to_normalize = ['resource_request', 'average_usage', 'maximum_usage', 'random_sample_usage']

# Iterate over each column and apply normalization
for column in columns_to_normalize:
    # Assuming the column contains dictionary-like strings
    df_normalized = json_normalize(df[column].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None))
    
    # Concatenate the normalized DataFrame with the original DataFrame
    df = pd.concat([df, df_normalized.add_prefix(f"{column}_")], axis=1)

    # Drop the original column
    df.drop(column, axis=1, inplace=True)


columns_to_drop = ['time', 'instance_events_type', 'scheduling_class', 'priority',
                   'constraint', 'collections_events_type', 'collection_name', 
                   'collection_logical_name', 'start_after_collection_ids', 'vertical_scaling', 
                   'scheduler', 'cpu_usage_distribution', 'tail_cpu_usage_distribution', 
                   'cluster', 'event', 'failed', 'random_sample_usage_memory', 'collection_id',
                   'alloc_collection_id', 'collection_type','start_time', 'end_time', 'sample_rate'
                   , 'cycles_per_instruction', 'memory_accesses_per_instruction', 'page_cache_memory',
                   'instance_index', 'machine_id',]


# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True)
df = df.drop(df.columns[df.columns.str.contains('Unnamed', case=False, regex=True)][0], axis=1)

# Adding more feature engineering as needed based on  data
# Cross-feature interactions
df['interaction_feature'] = df['maximum_usage_cpus'] * df['random_sample_usage_cpus']

# Creating lag features for memory_demand
df['memory_demand_lag_1'] = df['resource_request_memory'].shift(1)

# Creating rolling window statistics for memory_demand
df['memory_demand_rolling_mean'] = df['resource_request_memory'].rolling(window=3).mean()
df['memory_demand_rolling_std'] = df['resource_request_memory'].rolling(window=3).std()

# Check for empty values
empty_values = df.isnull().sum()
print("Empty Values:\n", empty_values)

# Check for zero values
zero_values = (df == 0).sum()
print("\nZero Values:\n", zero_values)

df.fillna(df.mean(), inplace=True)

# Now, check again for empty or zero values
updated_empty_values = df.isnull().sum()
updated_zero_values = (df == 0).sum()

print("\nUpdated Empty Values:\n", updated_empty_values)
print("\nUpdated Zero Values:\n", updated_zero_values)

# Specify the file name
file_name = "preprocessed_data.csv"

# Check if the file exists
if os.path.exists(file_name):
    # If it exists, replace it
    os.remove(file_name)
    
# Save the final DataFrame to a CSV file
df.to_csv(file_name, index=False)
# Print the first few rows of the DataFrame
print(df.head())


