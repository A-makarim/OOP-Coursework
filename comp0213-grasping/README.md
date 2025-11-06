# COMP0213 Object Oriented Programming for Robotics and AI Coursework Project

## Project Overview
This project focuses on grasping and grasp planning in robotics using machine learning models and the PyBullet simulation environment. The goal is to develop a system that can effectively plan and execute grasps using a robotic gripper.

## Project Structure
The project is organized into several directories and files, each serving a specific purpose:

- **src/**: Contains the main source code for the project.
  - **main.py**: Entry point for the application, initializing the simulation environment and orchestrating the grasp planning process.
  - **envs/**: Implementation of the PyBullet simulation environment.
    - **pybullet_env.py**: Setup and configuration for the robotic gripper and objects.
  - **agents/**: Defines the grasp planning agent.
    - **planner.py**: Methods for generating grasp candidates and evaluating their success.
  - **models/**: Contains the machine learning model for classifying grasp success.
    - **model.py**: Model architecture and training methods.
  - **data/**: Handles data loading and preprocessing.
    - **loader.py**: Loads the grasp dataset.
    - **preprocess.py**: Preprocessing functions for raw data.
  - **training/**: Manages the training process of the classifier.
    - **train.py**: Handles hyperparameter tuning and validation.
  - **utils/**: Utility functions for various tasks.
    - **helpers.py**: Data visualization and logging functions.

- **notebooks/**: Contains Jupyter notebooks for experimentation.
  - **experiments.ipynb**: Exploratory data analysis and grasp planning strategies.

- **data/**: Directories for storing datasets.
  - **raw/**: Raw grasp data collected during simulations.
  - **processed/**: Processed dataset ready for training and evaluation.

- **models/**: Directory for model checkpoints.
  - **checkpoints/**: Saves model checkpoints during training.

- **tests/**: Contains unit tests for the project.
  - **test_env.py**: Unit tests for the PyBullet environment.
  - **test_planner.py**: Unit tests for the grasp planning agent.

- **scripts/**: Scripts for running simulations.
  - **run_simulation.py**: Executes the grasp planning and evaluation pipeline.

- **requirements.txt**: Lists the dependencies required for the project.

- **pyproject.toml**: Project metadata and configuration for package management.

- **.gitignore**: Specifies files and directories to be ignored by version control.

## Requirements
To run this project, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- PyBullet

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd comp0213-grasping
   ```

2. Install the required dependencies.

3. Run the simulation:
   ```bash
   python src/main.py
   ```

4. For experimentation, open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/experiments.ipynb
   ```

## Contribution
Feel free to contribute to this project by submitting issues or pull requests. Your feedback and contributions are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for more details.