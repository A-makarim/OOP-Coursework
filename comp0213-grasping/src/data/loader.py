def load_grasp_data(data_dir):
    import os
    import pandas as pd

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")

    # Load the dataset
    grasp_data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            data = pd.read_csv(file_path)
            grasp_data.append(data)

    # Concatenate all data into a single DataFrame
    if grasp_data:
        return pd.concat(grasp_data, ignore_index=True)
    else:
        raise ValueError("No grasp data files found in the specified directory.")

def preprocess_grasp_data(grasp_data):
    # Placeholder for preprocessing steps
    # This function can include normalization, feature extraction, etc.
    return grasp_data

def load_and_preprocess_data(data_dir):
    raw_data = load_grasp_data(data_dir)
    processed_data = preprocess_grasp_data(raw_data)
    return processed_data