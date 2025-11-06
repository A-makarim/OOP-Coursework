# main.py

import sys
from envs.pybullet_env import PyBulletEnv
from agents.planner import GraspPlanner
from data.loader import DataLoader
from training.train import Trainer

def main():
    # Initialize the simulation environment
    env = PyBulletEnv()
    env.setup()

    # Load the grasp dataset
    data_loader = DataLoader()
    train_data, test_data = data_loader.load_data()

    # Initialize the grasp planning agent
    planner = GraspPlanner()

    # Train the model
    trainer = Trainer()
    trainer.train(train_data)

    # Run the simulation and grasp planning
    for object in env.get_objects():
        grasp_candidate = planner.generate_grasp_candidates(object)
        success = planner.evaluate_grasp(grasp_candidate)
        if success:
            print(f"Successful grasp for object: {object}")

if __name__ == "__main__":
    main()