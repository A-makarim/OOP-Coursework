def visualize_data(data):
    # Function to visualize the given data
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title('Data Visualization')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid()
    plt.show()

def log_message(message, log_file='project.log'):
    # Function to log messages to a specified log file
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def save_model_checkpoint(model, filepath):
    # Function to save the model checkpoint
    import joblib
    joblib.dump(model, filepath)

def load_model_checkpoint(filepath):
    # Function to load the model checkpoint
    import joblib
    return joblib.load(filepath)