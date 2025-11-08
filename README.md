# How to Use This Repository

This project simulates robotic grasps in PyBullet, collects data, trains a classifier (Random Forest), and evaluates grasp success. It’s organized for quick experiments and clear, object‑oriented structure.

## Prerequisites
- Windows 10/11
- Python 3.8+ installed and on PATH
- Git (optional, for cloning)

## Setup (Windows)
```powershell
# 1) Get the code
git clone https://github.com/A-makarim/Object-Oriented-Programming-Pybullet-Coursework-.git
cd Object-Oriented-Programming-Pybullet-Coursework-

# 2) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
main.py                 # Menu-driven workflow: generate → train → test → visualize
train_model.py          # Model training helpers
evaluate.py             # Evaluation and metrics
simulation_utils.py     # Shared simulation utilities
robots/                 # Gripper/task/robot simulation components
	gripper.py
	GripperTask.py
	RobotSimulator.py
	RobotSimulator.bat
	simulation_utils.py
data/                   # CSV datasets (training/test and with predictions)
models/                 # Trained models (.pkl) and URDFs
images/                 # Generated plots
plots/                  # Plot scripts (optional, advanced)
parameters.yaml         # Tunable settings
requirements.txt
```

## Run the Workflow
Launch the interactive menu and follow the on-screen prompts:
```powershell
python main.py
```
You’ll typically:
1) Generate training data (choose gripper and object) → CSV saved in `data/`
2) Train the classifier → model file (.pkl) saved in `models/`
3) Test the classifier → predictions appended to a new CSV in `data/`
4) Visualize results → plots saved in `images/` (or use scripts in `plots/`)

## Data Files
- Training CSV examples: `data/grasp_data_cuboid.csv`, `data/grasp_data_cylinder.csv`
- After testing, updated CSVs with predictions: `data/updated_*_with_predictions.csv`

Columns (training):
```
Position X, Position Y, Position Z,
Orientation Roll, Orientation Pitch, Orientation Yaw,
Initial Z, Final Z, Delta Z, Success
```
Columns (test extras):
```
Predicted Success, Actual Success, Match, Confidence
```

## Tips for Effective Use
- Generate at least ~120–140 samples per gripper/object for more stable accuracy.
- Keep CSVs in `data/` and models in `models/` (the code already uses safe, project‑relative paths).
- If you change defaults (e.g., sample count), check `parameters.yaml` or menu prompts.

## Visualizations
- Menu option will create plots and place them in `images/`.
- For more control, run scripts in `plots/` (e.g., KDEs, summary plots) after data exists.

## Extending the Project
- New gripper logic: add/modify classes in `robots/` (e.g., `gripper.py`, `GripperTask.py`).
- New object types/URDFs: place assets under `models/` and reference them from the simulation utilities.
- Model tweaks: adjust training parameters in `train_model.py` (e.g., number of trees) and retrain.

## Troubleshooting
- PyBullet GUI issues: update graphics drivers or try running without GUI if supported by your environment.
- Low accuracy: generate more data or balance classes; verify pose ranges make sense.
- Model not found: run the training step before testing (model saved under `models/`).
- CSV not updating: ensure you’re running from the project folder and files aren’t open in another program.

## Example: End‑to‑End Session
```powershell
# Activate environment
venv\Scripts\activate

# Start menu
python main.py

# 1) Generate data for PR2 + cuboid
# 2) Train model on cuboid data
# 3) Test model and review accuracy
# 4) Visualize results (check images/)
```

That’s it—use the menu for an end‑to‑end loop or dig into the scripts for finer control.
