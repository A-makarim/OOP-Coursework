"""
main.py - Grasp Planning Pipeline using Object-Oriented Programming

This module implements a complete grasp planning system with proper OOP principles:

OOP Design Patterns Used:
1. **Abstract Base Classes (ABC)**: 
   - BaseGripper: Abstract class defining gripper interface
   - BaseObject: Abstract class defining graspable object interface
   
2. **Inheritance**:
   - PR2Gripper, SDHGripper inherit from BaseGripper
   - CuboidObject, CylinderObject inherit from BaseObject
   
3. **Factory Pattern**:
   - GripperFactory: Creates gripper instances
   - ObjectFactory: Creates object instances
   
4. **Encapsulation**:
   - Configuration parameters stored in GRIPPER_CONFIG dictionary
   - Object properties hidden behind methods (get_height(), get_grasp_center())
   
5. **Polymorphism**:
   - Different grippers implement open_gripper() and close_gripper() differently
   - Different objects implement create_shape() and get_grasp_center() differently

Class Hierarchy:
- BaseGripper (ABC)
  ├── PR2Gripper (two-finger gripper)
  ├── SDHGripper (three-finger gripper)
  └── CustomGripper (placeholder)

- BaseObject (ABC)
  ├── CuboidObject (rectangular box)
  └── CylinderObject (circular cylinder)

Workflow:
1) User selects gripper type and object shape
2) Factory creates appropriate instances
3) Grasp poses are sampled using spherical/cylindrical sampling
4) Grippers approach with smooth trajectory and progressive finger closing
5) Success is evaluated by lifting test
6) Data saved to CSV for ML classifier training
"""

import os
import pybullet as p
import pybullet_data
import time
import math
import random
import pandas as pd
import joblib
import train_model
import visualize
from robots.gripper_factory import GripperFactory
from objects.object_factory import ObjectFactory
from evaluate import GripperEvaluator

# Get the directory where this script is located (OOPProject folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Gripper-specific configuration parameters
GRIPPER_CONFIG = {
    "pr2": {
        "open_pos": 0.5,
        "close_pos": 0.0,

        "cuboid_radius": 0.28,
        "cylinder_radius": 0.29,
        "radius_variation": (-0.1, 0.1),
        "y_offset": (-0.05, 0.05),
        "z_base_offset": +0,
        "z_variation": (-0.2, 0.2),
        "roll_range": (-math.pi, math.pi),
        "approach_distance": 0.1
    },
    "sdh": {
        "open_pos": -0.5,
        "close_pos": 0.0,   
        "cuboid_radius": 0.16,        # Closer approach for SDH (3-finger gripper)
        "cylinder_radius": 0.16,      # Closer approach for cylinder
        "radius_variation": (-0.05, 0.05),  # Smaller variation for tighter control
        "y_offset": (-0.03, 0.03),    # Smaller Y offset
        "z_base_offset": + 0,
        "z_variation": (-0.2, 0.2), # Slightly smaller Z variation
        "roll_range": (-math.pi, math.pi),     # Slightly smaller roll range
        "approach_distance": 0.1 # SDH needs to approach from 15cm away to avoid collisions
    }
}


def safe_step_simulation(num_steps=50, delay=0.01):
    """
    Safely step the simulation for num_steps, each with delay seconds,
    avoiding 'Not connected to physics server' errors if user closes PyBullet.
    """
    for _ in range(num_steps):
        if not p.isConnected():
            print("[INFO] PyBullet disconnected, stopping further steps.")
            return
        p.stepSimulation()
        time.sleep(delay)


class CustomObject:
    """
    Class to handle creation of both cuboid and cylinder objects
    in PyBullet for data generation.
    """

    def __init__(self, object_type="cuboid", size=None, height=None, radius=None):
        self.object_type = object_type.lower()
        self.size = size if size else [0.05, 0.05, 0.8]
        self.height = height if height else 0.8
        self.radius = radius if radius else 0.06
        self.object_id = None
        self._create_object()

    def _create_object(self):
        if self.object_type == "cuboid":
            self._create_cuboid()
        elif self.object_type == "cylinder":
            self._create_cylinder()
        else:
            raise ValueError("Invalid object type. Supported: 'cuboid', 'cylinder'.")

    def _create_cuboid(self):
        half_extents = [s / 2 for s in self.size]
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1, 1, 1, 1]
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents
        )
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, self.size[2] / 2],
            useMaximalCoordinates=False
        )

    def _create_cylinder(self):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.height,
            rgbaColor=[1, 1, 1, 1]
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height
        )
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, self.height / 2],
            useMaximalCoordinates=False
        )

    def get_id(self):
        return self.object_id


def generate_random_pose(object_position, height, step, total_steps, object_type, gripper_type="pr2"):
    config = GRIPPER_CONFIG.get(gripper_type)
    midplane_line = [
        object_position[0],
        object_position[1],
        object_position[2]
    ]
    height = midplane_line[2]


    # Select radius based on object type and gripper configuration
    if object_type == "cuboid":
        radius = config["cuboid_radius"]
    else: 
        radius = config["cylinder_radius"]



    # Apply radius variation
    radius_variation = random.uniform(*config["radius_variation"]) #unwrapping tupple
    radius += radius_variation

    print(f"Height: {height} Radius: {radius}")

    # Calculate gripper position


    angle = (2 * math.pi) * (step / total_steps)
    
    gripper_x = midplane_line[0] + radius * math.cos(angle) # random already applied in radius
    gripper_y = midplane_line[1] + radius * math.sin(angle) + random.uniform(*config["y_offset"])
    gripper_z = midplane_line[2] + config["z_base_offset"]+ random.uniform(*config["z_variation"]) 
    position = [gripper_x, gripper_y, gripper_z]


    # Calculate direction vector pointing toward object center
    direction_vec = [
        midplane_line[0] - position[0],
        midplane_line[1] - position[1],
        midplane_line[2] - position[2]     # should be zero for now
    ]

    mag = math.sqrt(sum(i**2 for i in direction_vec))   # magnitude, length
    if mag < 1e-8:
        direction_vec = [1.0, 0.0, 0.0]
        mag = 1.0
    else:
        direction_vec = [i / mag for i in direction_vec]   # unit vector

    # print(f" object: {midplane_line}, gripper: {position} dir: {direction_vec} mag: {mag} ")
    print(f" object: {[round(x, 2) for x in midplane_line]}, gripper: {[round(x, 2) for x in position]} dir: {[round(x, 2) for x in direction_vec]} mag: {mag:.2f} ")

    # Calculate orientation
    yaw = math.atan2(direction_vec[1], direction_vec[0])
    pitch = -math.asin(direction_vec[2])
    roll = random.uniform(*config["roll_range"])

    orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    if gripper_type == "sdh":

        # STEP 1 — rotate SDH so it faces the object
        face_object = p.getQuaternionFromEuler([0, math.pi/2, 0])
        orientation = p.multiplyTransforms([0,0,0], orientation,
                                        [0,0,0], face_object)[1]

        # STEP 2 — local roll of +90° (roll about the direction fingers point)
        roll_local = p.getQuaternionFromEuler([0, 0, math.pi/2])

        # Apply local roll in the SDH frame → multiply on the RIGHT
        orientation = p.multiplyTransforms([0,0,0], orientation,
                                        [0,0,0], roll_local)[1]


    return position, orientation


def generate_data_for_shape(object_type="cuboid", num_grasps=50, gripper_type="pr2"):
    """
    Generate new data for the given shape (cuboid/cylinder) with specified gripper.
    Writes to data/grasp_data_{gripper}_{shape}.csv, delegates CSV saving
    and success logic to GripperEvaluator from evaluate.py.
    """

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    # Use ObjectFactory to create object (OOP design pattern)
    graspable_object = ObjectFactory.create_object(object_type) # CuboidObject() or CylinderObject()
    object_id = graspable_object.get_id()
    object_pos = graspable_object.get_grasp_center()

    p.changeDynamics(object_id, -1, mass=0.2, lateralFriction=1.2, spinningFriction=0.1)
    
    # Use GripperFactory to create gripper (OOP design pattern)
    # this is spawning
    gripper = GripperFactory.create_gripper(
        gripper_type, 
        [0, 0, graspable_object.get_height() + 1], 
        [0, 0, 0, 1]
    )

    data_folder = os.path.join(SCRIPT_DIR, "data") # Script_DIR is defined at the top
    os.makedirs(data_folder, exist_ok=True) # ensure data folder exists
    csv_file = os.path.join(data_folder, f"grasp_data_{gripper_type}_{object_type}.csv") # CSV file path for saving grasp data
    evaluator = GripperEvaluator(csv_filename=csv_file) # delegate CSV saving to evaluator
    
    # Get gripper configuration
    config = GRIPPER_CONFIG.get(gripper_type)

    for step in range(num_grasps):
        print(f"[INFO] Generating grasp {step+1}/{num_grasps} for {object_type}...")
        
        # Reset object position
        p.resetBasePositionAndOrientation(object_id, object_pos, [0, 0, 0, 1]) # reset to initial pos
        safe_step_simulation(30)
        
        # Open gripper fingers
        gripper.open_gripper()
        safe_step_simulation(30)

        # Generate target grasp pose
        position, orientation_quat = generate_random_pose(
            object_pos, graspable_object.get_height(), step, num_grasps, object_type, gripper_type
        )
        
        # If gripper needs approach phase (to avoid collisions)
        if config["approach_distance"] > 0:
            # Calculate direction vector from gripper to object center
            obj_center = graspable_object.get_grasp_center()
            approach_dir = [
                obj_center[0] - position[0],   # from pose est
                obj_center[1] - position[1],
                obj_center[2] - position[2]
            ]
            # this is without the approach distance
            



            mag = math.sqrt(sum(d**2 for d in approach_dir))
            if mag > 1e-8:
                approach_dir = [d / mag for d in approach_dir]
            else:
                approach_dir = [1, 0, 0]
            
            # Calculate approach start position (move back by approach_distance)
            approach_start = [
                position[0] - approach_dir[0] * config["approach_distance"],
                position[1] - approach_dir[1] * config["approach_distance"],
                position[2] - approach_dir[2] * config["approach_distance"]
            ]
            
            # Move to approach start position
            gripper.set_position(approach_start, orientation_quat)  # intially here
            safe_step_simulation(20)
            
            # Gradual approach with finger closing
            num_approach_steps = 50
            for i in range(num_approach_steps):
                # Interpolate position from approach_start to final position
                t = (i + 1) / num_approach_steps # t
                current_pos = [
                    approach_start[0] + t * (position[0] - approach_start[0]),
                    approach_start[1] + t * (position[1] - approach_start[1]),
                    approach_start[2] + t * (position[2] - approach_start[2])
                ]
                gripper.set_position(current_pos, orientation_quat)
                
                # Start closing fingers halfway through approach
                if i >= num_approach_steps // 2:
                    # Gradually close fingers
                        close_progress = (i - num_approach_steps // 2) / (num_approach_steps // 2)

                        open_pos  = config["open_pos"]
                        close_pos = config["close_pos"]

                        # interpolate from open → close based on close_progress
                        target_pos = open_pos + close_progress * (close_pos - open_pos)

                        for joint_index in gripper.active_joints:
                            p.setJointMotorControl2(
                                bodyIndex=gripper.gripper,
                                jointIndex=joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_pos,
                                force=300
        )

                p.stepSimulation()
                time.sleep(0.01)
            
            safe_step_simulation(30)
        else:
            # PR2: Direct approach without gradual motion
            gripper.set_position(position, orientation_quat)
            safe_step_simulation(30)
            # For SDH we may want a partial close (leave a small gap).
            if gripper_type == 'sdh':
                gripper.close_gripper(fraction=0.6)  # adjust fraction as needed (0-1)
            else:
                gripper.close_gripper()
            safe_step_simulation(30)

        init_obj_pos, _ = p.getBasePositionAndOrientation(object_id)

        # Lift the object
        lift_target_z = position[2] + 0.3
        gripper.move_up_smoothly(target_z=lift_target_z, steps=100, delay=0.005)
        safe_step_simulation(100)

        success_code, delta_z, final_pos = evaluator.evaluate_grasp(object_id, init_obj_pos)

        orientation_euler = p.getEulerFromQuaternion(orientation_quat)

        row = [
            position[0],
            position[1],
            position[2],
            orientation_euler[0],
            orientation_euler[1],
            orientation_euler[2],
            init_obj_pos[2],     # the "Initial Z" from the moment we captured
            final_pos[2],        # final Z from evaluate_grasp
            delta_z,
            success_code
        ]
        print([round(i, 2) for i in row])
        evaluator.save_to_csv(row)

        gripper.open_gripper()
        safe_step_simulation(30)

        spawn_z = graspable_object.get_height() + 1.0
        gripper.set_position([0, 0, spawn_z], [0, 0, 0, 1])
        safe_step_simulation(30)


    p.disconnect()
    print(f"[INFO] Generated {num_grasps} grasps for {gripper_type} gripper on '{object_type}' -> {csv_file}")


def test_classifier(object_type, num_tests=10, gripper_type="pr2"):
    """
    Test the trained classifier on a separate test set.
    Generates new grasp poses, predicts success using trained model,
    executes the grasp, and compares prediction vs actual outcome.
    Saves results to test_results_{gripper}_{shape}.csv
    """
    models_folder = os.path.join(SCRIPT_DIR, "models")
    model_file = os.path.join(models_folder, f"{gripper_type}_{object_type}_grasp_model.pkl")
    
    if not os.path.exists(model_file):
        print(f"[ERROR] Model file {model_file} not found. Please train the model first.")
        return
    
    print(f"[INFO] Loading model from {model_file}")
    model = joblib.load(model_file)
    
    # Setup simulation
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Create object and gripper
    graspable_object = ObjectFactory.create_object(object_type)
    object_id = graspable_object.get_id()
    object_pos = graspable_object.get_grasp_center()
    
    p.changeDynamics(object_id, -1, mass=0.2, lateralFriction=1.2, spinningFriction=0.1)
    gripper = GripperFactory.create_gripper(
        gripper_type,
        [0, 0, graspable_object.get_height() + 0.2],
        [0, 0, 0, 1]
    )
    
    # Setup test results file
    data_folder = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(data_folder, exist_ok=True)
    test_results_file = os.path.join(data_folder, f"test_results_{gripper_type}_{object_type}.csv")
    
    # Check if file exists for appending
    file_exists = os.path.exists(test_results_file)
    
    # Get configuration
    config = GRIPPER_CONFIG.get(gripper_type, GRIPPER_CONFIG["pr2"])
    
    results = []
    correct_predictions = 0
    
    print(f"\n[INFO] Starting {num_tests} test grasps...")
    print(f"[INFO] Each grasp will be: Predicted → Executed → Compared")
    
    for test_num in range(num_tests):
        print(f"\n--- Test Grasp {test_num + 1}/{num_tests} ---")
        
        # Reset object
        p.resetBasePositionAndOrientation(object_id, object_pos, [0, 0, 0, 1])
        safe_step_simulation(30)
        
        # Open gripper
        gripper.open_gripper()
        safe_step_simulation(30)
        
        # Generate random test pose
        position, orientation_quat = generate_random_pose(
            object_pos, graspable_object.get_height(), test_num, num_tests, object_type, gripper_type
        )
        
        orientation_euler = p.getEulerFromQuaternion(orientation_quat)
        
        # PREDICT using trained model
        features = [[
            position[0], position[1], position[2],
            orientation_euler[0], orientation_euler[1], orientation_euler[2]
        ]]
        prediction_proba = model.predict_proba(features)[0]
        # determine index of positive class (1) robustly
        if hasattr(model, "classes_"):
            try:
                pos_index = list(model.classes_).index(1)
            except ValueError:
                pos_index = 1 if len(prediction_proba) > 1 else 0
        else:
            pos_index = 1 if len(prediction_proba) > 1 else 0

        success_proba = prediction_proba[pos_index]
        THRESHOLD = 0.7
        predicted_success = 1 if success_proba >= THRESHOLD else 0

        print(f"  Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        print(f"  Orientation: ({orientation_euler[0]:.3f}, {orientation_euler[1]:.3f}, {orientation_euler[2]:.3f})")
        print(f"  PREDICTED: {'SUCCESS' if predicted_success == 1 else 'FAILURE'} (confidence: {success_proba:.2%})")
        
        # EXECUTE the grasp (with approach phase if needed)
        if config["approach_distance"] > 0:
            # Calculate approach
            obj_center = graspable_object.get_grasp_center()
            approach_dir = [
                obj_center[0] - position[0],
                obj_center[1] - position[1],
                obj_center[2] - position[2]
            ]
            mag = math.sqrt(sum(d**2 for d in approach_dir))
            if mag > 1e-8:
                approach_dir = [d / mag for d in approach_dir]
            else:
                approach_dir = [1, 0, 0]
            
            approach_start = [
                position[0] - approach_dir[0] * config["approach_distance"],
                position[1] - approach_dir[1] * config["approach_distance"],
                position[2] - approach_dir[2] * config["approach_distance"]
            ]
            
            gripper.set_position(approach_start, orientation_quat)
            safe_step_simulation(20)
            
            # Gradual approach
            num_approach_steps = 50
            for i in range(num_approach_steps):
                t = (i + 1) / num_approach_steps
                current_pos = [
                    approach_start[0] + t * (position[0] - approach_start[0]),
                    approach_start[1] + t * (position[1] - approach_start[1]),
                    approach_start[2] + t * (position[2] - approach_start[2])
                ]
                gripper.set_position(current_pos, orientation_quat)
                
                if i >= num_approach_steps // 2:
                    close_progress = (i - num_approach_steps // 2) / (num_approach_steps // 2)
                    for joint_index in gripper.active_joints:
                        target_pos = -0.5 + close_progress * 1.5
                        p.setJointMotorControl2(
                            bodyIndex=gripper.gripper,
                            jointIndex=joint_index,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_pos,
                            force=300
                        )
                
                p.stepSimulation()
                time.sleep(0.01)
            
            safe_step_simulation(30)
        else:
            gripper.set_position(position, orientation_quat)
            safe_step_simulation(30)
            if gripper_type == 'sdh':
                gripper.close_gripper(fraction=0.6)
            else:
                gripper.close_gripper()
            safe_step_simulation(30)
        
        init_obj_pos, _ = p.getBasePositionAndOrientation(object_id)
        
        # Lift
        lift_target_z = position[2] + 0.3
        gripper.move_up_smoothly(target_z=lift_target_z, steps=100, delay=0.005)
        safe_step_simulation(100)
        
        # Evaluate actual outcome
        final_pos, _ = p.getBasePositionAndOrientation(object_id)
        delta_z = final_pos[2] - init_obj_pos[2]
        actual_success = 1 if delta_z > 0.1 else 0
        
        print(f"  ACTUAL: {'SUCCESS' if actual_success == 1 else 'FAILURE'} (lifted: {delta_z:.3f}m)")
        
        # Compare
        match = (predicted_success == actual_success)
        if match:
            correct_predictions += 1
            print(f"  [CORRECT] Prediction matched actual outcome")
        else:
            print(f"  [INCORRECT] Prediction did not match actual outcome")
        
        # Create result row
        result_row = {
            'Position X': position[0],
            'Position Y': position[1],
            'Position Z': position[2],
            'Orientation Roll': orientation_euler[0],
            'Orientation Pitch': orientation_euler[1],
            'Orientation Yaw': orientation_euler[2],
            'Predicted Success': predicted_success,
            'Actual Success': actual_success,
            'Match': match,
            'Confidence': success_proba,
            'Delta Z': delta_z
        }
        
        # Save immediately to CSV (append mode)
        result_df = pd.DataFrame([result_row])
        if test_num == 0 and not file_exists:
            # First test and file doesn't exist: write with header
            result_df.to_csv(test_results_file, mode='w', index=False, header=True)
        else:
            # Append without header
            result_df.to_csv(test_results_file, mode='a', index=False, header=False)
        
        results.append(result_row)
        
        gripper.open_gripper()
        safe_step_simulation(30)
    
    p.disconnect()
    
    # Print summary
    accuracy = (correct_predictions / num_tests) * 100
    print(f"\n" + "="*60)
    print(f"TEST RESULTS SUMMARY")
    print(f"="*60)
    print(f"Total Tests: {num_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {num_tests - correct_predictions}")
    print(f"Prediction Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {test_results_file}")
    print(f"="*60)


def visualize_training_data(object_type, gripper_type="pr2"):
    """Visualize training data with meaningful plots."""
    visualize.visualize_training_data(object_type, gripper_type)


def visualize_test_results(object_type, gripper_type="pr2"):
    """Visualize test results and predictions."""
    visualize.visualize_test_results(object_type, gripper_type)


def main():
    """
    Main menu system for grasp planning pipeline.
    Supports:
    1. Generate Training Data (accumulates in CSV)
    2. Train Classifier (creates models)
    3. Test Classifier (separate test set with predictions)
    4. Visualize Results
    """
    data_folder = os.path.join(SCRIPT_DIR, "data")
    models_folder = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)

    while True:
        print("\n" + "="*60)
        print("GRASP PLANNING SYSTEM - Main Menu")
        print("="*60)
        print("1. Generate Training Data (accumulates to existing data)")
        print("2. Train Classifier Model")
        print("3. Test Classifier (separate test set)")
        print("4. Visualize Results")
        print("5. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            # Generate training data
            print("\n--- Generate Training Data ---")
            gripper = input("Enter gripper type (pr2/sdh): ").strip().lower()
            if gripper not in ["pr2", "sdh"]:
                print("[ERROR] Invalid gripper type.")
                continue
            
            shape = input("Enter object shape (cuboid/cylinder): ").strip().lower()
            if shape not in ["cuboid", "cylinder"]:
                print("[ERROR] Invalid shape.")
                continue
            
            try:
                num_grasps = int(input("Enter number of grasps to generate: "))
            except ValueError:
                print("[ERROR] Invalid number.")
                continue
            
            # Check current data count
            csv_file = os.path.join(data_folder, f"grasp_data_{gripper}_{shape}.csv")
            if os.path.exists(csv_file):
                existing_data = pd.read_csv(csv_file)
                current_count = len(existing_data)
                print(f"[INFO] Current dataset has {current_count} samples")
                print(f"[INFO] After this run, you will have {current_count + num_grasps} samples")
            
            generate_data_for_shape(shape, num_grasps, gripper)
            
        elif choice == '2':
            # Train classifier
            print("\n--- Train Classifier Model ---")
            gripper = input("Enter gripper type (pr2/sdh): ").strip().lower()
            if gripper not in ["pr2", "sdh"]:
                print("[ERROR] Invalid gripper type.")
                continue
            
            shape = input("Enter object shape (cuboid/cylinder): ").strip().lower()
            if shape not in ["cuboid", "cylinder"]:
                print("[ERROR] Invalid shape.")
                continue
            
            train_model.train_model(shape, gripper)
            
        elif choice == '3':
            # Test classifier
            print("\n--- Test Classifier (Separate Test Set) ---")
            gripper = input("Enter gripper type (pr2/sdh): ").strip().lower()
            if gripper not in ["pr2", "sdh"]:
                print("[ERROR] Invalid gripper type.")
                continue
            
            shape = input("Enter object shape (cuboid/cylinder): ").strip().lower()
            if shape not in ["cuboid", "cylinder"]:
                print("[ERROR] Invalid shape.")
                continue
            
            try:
                num_tests = int(input("Enter number of test grasps (minimum 10): "))
                if num_tests < 10:
                    print("[WARNING] Coursework requires minimum 10 test attempts.")
                    num_tests = max(10, num_tests)
            except ValueError:
                print("[ERROR] Invalid number.")
                continue
            
            test_classifier(shape, num_tests, gripper)
            
        elif choice == '4':
            # Visualize results
            print("\n--- Visualize Results ---")
            print("1. Visualize Training Data")
            print("2. Visualize Test Results")
            viz_choice = input("Enter choice (1-2): ").strip()
            
            gripper = input("Enter gripper type (pr2/sdh): ").strip().lower()
            if gripper not in ["pr2", "sdh"]:
                print("[ERROR] Invalid gripper type.")
                continue
            
            shape = input("Enter object shape (cuboid/cylinder): ").strip().lower()
            if shape not in ["cuboid", "cylinder"]:
                print("[ERROR] Invalid shape.")
                continue
            
            if viz_choice == '1':
                visualize_training_data(shape, gripper)
            elif viz_choice == '2':
                visualize_test_results(shape, gripper)
            
        elif choice == '5':
            print("\n[INFO] Exiting. Goodbyeeeee!")
            break
        else:
            print("[ERROR] Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
