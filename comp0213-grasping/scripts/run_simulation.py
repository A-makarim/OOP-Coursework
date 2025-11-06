# /comp0213-grasping/comp0213-grasping/scripts/run_simulation.py

import pybullet as p
import time
from src.envs.pybullet_env import PyBulletEnv
from src.agents.planner import GraspPlanner
from src.data.loader import DataLoader
from src.training.train import Trainer

def main():
    # Initialize the PyBullet simulation environment
    env = PyBulletEnv()
    env.setup()

    # Load the grasp dataset
    data_loader = DataLoader()
    dataset = data_loader.load_data()

    # Initialize the grasp planning agent
    planner = GraspPlanner()

    # Run the simulation for grasp planning
    for i in range(len(dataset)):
        grasp_candidates = planner.generate_grasp_candidates(dataset[i])
        success = planner.evaluate_grasps(grasp_candidates)

        # Log the results
        print(f"Grasp attempt {i + 1}: Success - {success}")

        # Step the simulation
        env.step()
        time.sleep(0.1)  # Adjust the sleep time as necessary for simulation speed

    # Clean up the environment
    env.cleanup()

if __name__ == "__main__":
    main()