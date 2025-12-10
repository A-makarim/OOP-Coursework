# Robotic Grasp Prediction System# How to Use This Repository



A PyBullet-based simulation system for robotic grasp prediction using machine learning. This project simulates grasping tasks with different gripper-object combinations, trains Random Forest classifiers, and evaluates grasp success predictions.This project simulates robotic grasps in PyBullet, collects data, trains a classifier (Random Forest), and evaluates grasp success. It’s organized for quick experiments and clear, object‑oriented structure.



## FeaturesEverything is working. noise and random.uniform and config parameters can be refiend more

Training datasets needs to be generateds, option(2)

- Simulation of robotic grasping with PR2 and SDH grippersThen test the classifier, option(3)

- Support for cuboid and cylinder objectsGet plots for courseworkreport, option(4) 

- Automated data collection with configurable noise parameters

- Machine learning-based grasp success prediction#

- Comprehensive evaluation metrics and visualization

- Voice-assisted operation using Google Gemini API## Setup (Windows)

```powershell

## System Requirements# 1) Get the code

git clone https://github.com/A-makarim/Object-Oriented-Programming-Pybullet-Coursework-.git

- Python 3.8 or highercd Object-Oriented-Programming-Pybullet-Coursework-

- Windows, macOS, or Linux

- PyBullet physics engine# 2) Create and activate a virtual environment

- scikit-learn for machine learningpython -m venv venv

- Google Gemini API key (optional, for voice assistant)venv\Scripts\activate



## Installation# 3) Install dependencies

pip install -r requirements.txt

### 1. Clone the Repository```



```bash## Project Structure

git clone https://github.com/A-makarim/Object-Oriented-Programming-Pybullet-Coursework-.git```

cd Object-Oriented-Programming-Pybullet-Coursework-main.py                 # Menu-driven workflow: generate → train → test → visualize

```train_model.py          # Model training helpers

evaluate.py             # Evaluation and metrics

### 2. Create Virtual Environmentvisualize.py            # Visualization module

robots/                 # Gripper implementations (PR2, SDH)

**Windows:**  gripper.py

```powershell  gripper_factory.py

python -m venv venvobjects/                # Object implementations (Cuboid, Cylinder)

venv\Scripts\activate  base_object.py

```  cuboid.py

  cylinder.py

**macOS/Linux:**  object_factory.py

```bashdata/                   # CSV datasets (training/test and with predictions)

python3 -m venv venvmodels/                 # Trained models (.pkl) and URDFs

source venv/bin/activateimages/                 # Generated plots

```requirements.txt

```## Run the Workflow

### 3. Install DependenciesLaunch the interactive menu and follow the on-screen prompts:

```powershell

```bashpython main.py

pip install -r requirements.txt```

```You’ll:

1) Generate training data (choose gripper and object) → CSV saved in `data/`

## Project Structure2) Train the classifier → model file (.pkl) saved in `models/`

3) Test the classifier → predictions appended to a new CSV in `data/`

```4) Visualize results → plots saved in `images/`

OOPProject/

├── main.py                     # Main workflow script with interactive menu## Data Files

├── train_model.py              # Model training module- Training CSV examples: `data/grasp_data_cuboid.csv`, `data/grasp_data_cylinder.csv`

├── evaluate.py                 # Evaluation and metrics calculation- After testing, updated CSVs with predictions: `data/updated_*_with_predictions.csv`

├── voice_assistant_gemini.py   # Voice-assisted operation

├── batch_generate.py           # Batch data generationColumns (training):

├── regenerate_all_figures.py   # Figure regeneration for reports```

├── robots/                     # Gripper implementationsPosition X, Position Y, Position Z,

│   ├── gripper.py             # Base gripper classes (PR2, SDH)Orientation Roll, Orientation Pitch, Orientation Yaw,

│   └── gripper_factory.py     # Gripper factory patternInitial Z, Final Z, Delta Z, Success

├── objects/                    # Object implementations
│   ├── base_object.py         # Abstract object base
│   ├── cuboid.py              # Cuboid object
│   ├── cylinder.py            # Cylinder object
│   └── object_factory.py      # Object factory pattern
├── data/                       # Generated datasets
│   ├── grasp_data_*.csv       # Training data
│   ├── test_results_*.csv     # Test predictions
│   ├── statistics.json        # Performance metrics
│   └── summary_table.csv      # Summary statistics
├── models/                     # Trained models
│   ├── grasp_model_*.pkl      # Random Forest models
│   ├── scaler_*.pkl           # Feature scalers
│   └── sdh.urdf               # SDH gripper URDF
├── latex_report/               # LaTeX report and figures
│   └── figures/               # Generated visualizations
└── requirements.txt            # Python dependencies
```

## Usage

### Standard Operation (Interactive Menu)

Run the main script to access the interactive menu:

```bash
python main.py
```

The menu provides the following options:

**1. Generate Training Data**
- Select gripper type (PR2 or SDH)
- Select object type (Cuboid or Cylinder)
- Specify number of samples to generate
- Data saved to `data/grasp_data_{gripper}_{object}.csv`

**2. Train Model**
- Trains Random Forest classifier on collected data
- Performs 5-fold cross-validation
- Saves model to `models/grasp_model_{gripper}_{object}.pkl`
- Saves feature scaler to `models/scaler_{gripper}_{object}.pkl`

**3. Test Model**
- Generates test data and makes predictions
- Evaluates model performance (accuracy, precision, recall, F1)
- Saves results to `data/test_results_{gripper}_{object}.csv`
- Updates `data/statistics.json` with metrics

**4. Visualize Results**
- Generates confusion matrices
- Creates ROC curves
- Produces training data analysis plots
- Saves visualizations to `latex_report/figures/`

**5. Generate All Data (Batch Mode)**
- Automatically generates data for all four configurations:
  - PR2-Cuboid
  - PR2-Cylinder
  - SDH-Cuboid
  - SDH-Cylinder
- Useful for comprehensive dataset creation

### Voice-Assisted Operation

The voice assistant provides natural language interaction with the system using Google Gemini API.

#### Setup Google Gemini API

1. **Obtain API Key:**
   - Visit https://makersuite.google.com/app/apikey
   - Sign in with your Google account
   - Create a new API key
   - Copy the generated key

2. **Configure API Key:**
   
   Create a file named `gemini_config.json` in the project root:
   
   ```json
   {
     "api_key": "YOUR_API_KEY_HERE"
   }
   ```
   
   Replace `YOUR_API_KEY_HERE` with your actual Gemini API key.

3. **Run Voice Assistant:**
   
   ```bash
   python voice_assistant_gemini.py
   ```

#### Voice Assistant Commands

The assistant understands natural language commands such as:

- "Generate 500 samples for PR2 gripper with cuboid object"
- "Train a model for SDH and cylinder configuration"
- "Test the model and show me the accuracy"
- "Create visualizations for all configurations"
- "What is the current success rate for PR2-Cuboid?"
- "Show me statistics for all models"

The assistant will:
- Parse your request
- Execute the appropriate operations
- Provide feedback on results
- Handle errors gracefully

### Batch Data Generation

For generating complete datasets programmatically:

```bash
python batch_generate.py
```

This script:
- Generates training data for all four configurations
- Uses optimized parameters from `parameters.yaml`
- Saves data to respective CSV files
- Provides progress feedback

### Regenerating Visualizations

To regenerate all figures for the report:

```bash
python regenerate_all_figures.py
```

This generates:
- Training data analysis plots (4 per configuration)
- Test results and confusion matrices (4 per configuration)
- ROC curves with AUC scores (4 per configuration)
- Feature importance rankings (4 per configuration)
- Combined comparison figures (6 total)

All figures are saved to `latex_report/figures/` directory.

## Configuration

### Noise Parameters

Noise parameters for each gripper are defined in `main.py` within the `CONFIG` dictionary:

```python
CONFIG = {
    'pr2': {
        'radius_variation': (-0.1, 0.1),      # ±10cm radial noise
        'y_offset': (-0.05, 0.05),            # ±5cm Y-axis offset
        'z_variation': (-0.2, 0.2),           # ±20cm Z-axis noise
        'roll_range': (-math.pi, math.pi),    # Full roll range
        'close_start': 0.5                    # Closing start ratio
    },
    'sdh': {
        'radius_variation': (-0.05, 0.05),    # ±5cm radial noise
        'y_offset': (-0.03, 0.03),            # ±3cm Y-axis offset
        'z_variation': (-0.05, 0.05),         # ±5cm Z-axis noise
        'roll_range': (-math.pi, math.pi),    # full roll range
        'close_start': 0.7                    # Closing start ratio
    }
}
```

### Model Parameters

Random Forest classifier parameters in `train_model.py`:

```python
model = RandomForestClassifier(
    n_estimators=200,          # Number of trees
    max_depth=15,              # Maximum tree depth
    min_samples_split=5,       # Minimum samples to split
    min_samples_leaf=2,        # Minimum samples per leaf
    class_weight='balanced',   # Handle class imbalance
    random_state=42            # Reproducibility
)
```

## Data Format

### Training Data CSV

Columns:
- `Position X`, `Position Y`, `Position Z`: Gripper position (meters)
- `Orientation Roll`, `Orientation Pitch`, `Orientation Yaw`: Gripper orientation (radians)
- `Initial Z`: Starting Z position (meters)
- `Final Z`: Final Z position after lift (meters)
- `Delta Z`: Change in Z position (meters)
- `Success`: Grasp outcome (1 = success, 0 = failure)

### Test Results CSV

Same columns as training data, plus:
- `Predicted Success`: Model prediction (1 or 0)
- `Prediction Confidence`: Probability of success (0.0 to 1.0)

### Statistics JSON

Contains per-configuration metrics:
- Training samples and success rate
- Test accuracy, precision, recall, F1 score
- Confusion matrix (TP, TN, FP, FN)

## Performance Metrics

Current system performance (9,770 training samples, 600 test samples):

| Configuration | Training Samples | Success Rate | Test Accuracy | F1 Score |
|--------------|------------------|--------------|---------------|----------|
| PR2-Cuboid   | 2,120           | 49.5%        | 82.7%         | 0.812    |
| PR2-Cylinder | 1,643           | 42.7%        | 83.3%         | 0.800    |
| SDH-Cuboid   | 1,495           | 43.9%        | 86.0%         | 0.844    |
| SDH-Cylinder | 4,512           | 41.5%        | 87.3%         | 0.835    |
| **Average**  | **9,770**       | **44.4%**    | **84.8%**     | **0.823**|

## Troubleshooting

### PyBullet Connection Issues

If you encounter "Not connected to physics server" errors:
- Ensure PyBullet is properly installed: `pip install pybullet`
- Check that no other PyBullet instances are running
- Restart your Python environment

### Model Loading Errors

If models fail to load:
- Verify `.pkl` files exist in `models/` directory
- Ensure models were trained with the same scikit-learn version
- Retrain models using option 2 in the main menu

### Voice Assistant Not Responding

If the voice assistant fails:
- Verify `gemini_config.json` exists with valid API key
- Check internet connection for API access
- Ensure Google Gemini API quota is not exceeded
- Check API key permissions at https://makersuite.google.com

### Import Errors

If you encounter missing module errors:
- Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify Python version is 3.8 or higher: `python --version`

## Development

### Adding New Grippers

1. Create gripper class inheriting from `BaseGripper` in `robots/gripper.py`
2. Implement `open_gripper()` and `close_gripper()` methods
3. Add gripper to factory in `robots/gripper_factory.py`
4. Update CONFIG dictionary in `main.py` with noise parameters

### Adding New Objects

1. Create object class inheriting from `BaseObject` in `objects/`
2. Implement `_create_object()` method
3. Add object to factory in `objects/object_factory.py`
4. Update menu options in `main.py`

## Citation

If you use this code in your research, please cite:

```
Robotic Grasp Prediction System
Object-Oriented Programming Coursework
GitHub: https://github.com/A-makarim/Object-Oriented-Programming-Pybullet-Coursework-
```

## License

This project is developed for educational purposes as part of an Object-Oriented Programming coursework.

## Contact

For questions or issues, please open an issue on the GitHub repository.
