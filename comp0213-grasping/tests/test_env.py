# test_env.py

import unittest
from src.envs.pybullet_env import PyBulletEnv

class TestPyBulletEnv(unittest.TestCase):

    def setUp(self):
        """Set up the PyBullet environment for testing."""
        self.env = PyBulletEnv()

    def test_environment_initialization(self):
        """Test if the environment initializes correctly."""
        self.assertIsNotNone(self.env)

    def test_step_function(self):
        """Test the step function of the environment."""
        action = self.env.action_space.sample()  # Sample a random action
        state, reward, done, info = self.env.step(action)
        self.assertIsNotNone(state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_reset_function(self):
        """Test the reset function of the environment."""
        initial_state = self.env.reset()
        self.assertIsNotNone(initial_state)

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()

if __name__ == '__main__':
    unittest.main()