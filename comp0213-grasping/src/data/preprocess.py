def preprocess_data(raw_data):
    """
    Preprocess the raw grasp data for model training.

    Parameters:
    - raw_data: A DataFrame containing the raw grasp data.

    Returns:
    - processed_data: A DataFrame containing the normalized and feature-extracted data.
    """
    # Normalize the data
    normalized_data = (raw_data - raw_data.mean()) / raw_data.std()

    # Feature extraction (example: extracting relevant features)
    processed_data = normalized_data[['feature1', 'feature2', 'feature3']]  # Adjust based on actual features

    return processed_data


def split_data(processed_data, test_size=0.2):
    """
    Split the processed data into training and testing sets.

    Parameters:
    - processed_data: A DataFrame containing the processed data.
    - test_size: Proportion of the dataset to include in the test split.

    Returns:
    - train_data: DataFrame for training.
    - test_data: DataFrame for testing.
    """
    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(processed_data, test_size=test_size, random_state=42)
    return train_data, test_data


def save_processed_data(processed_data, file_path):
    """
    Save the processed data to a specified file path.

    Parameters:
    - processed_data: A DataFrame containing the processed data.
    - file_path: The path where the processed data will be saved.
    """
    processed_data.to_csv(file_path, index=False)