import unittest
from src.agents.planner import GraspPlanner

class TestGraspPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = GraspPlanner()

    def test_generate_grasp_candidates(self):
        # Test the generation of grasp candidates
        candidates = self.planner.generate_grasp_candidates()
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)

    def test_evaluate_grasp_success(self):
        # Test the evaluation of grasp success
        candidate = self.planner.generate_grasp_candidates()[0]
        success = self.planner.evaluate_grasp_success(candidate)
        self.assertIn(success, [True, False])

    def test_grasp_planning_integration(self):
        # Test the integration of grasp planning process
        candidates = self.planner.generate_grasp_candidates()
        best_candidate = self.planner.select_best_grasp(candidates)
        self.assertIsNotNone(best_candidate)

if __name__ == '__main__':
    unittest.main()