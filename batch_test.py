"""
batch_test.py - Automated batch testing for all gripper-object combinations

Run this script to test trained classifiers on all 4 combinations:
- PR2 gripper × cuboid
- PR2 gripper × cylinder
- SDH gripper × cuboid
- SDH gripper × cylinder

The script will run test_classifier for each combination sequentially and append
to existing test results CSV files, allowing you to accumulate test data over multiple runs.

Usage:
    python batch_test.py --tests 150
    python batch_test.py --tests 50 --combinations pr2-cuboid sdh-cylinder
    python batch_test.py --tests 100
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Import from main.py
from main import test_classifier


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_batch_testing(tests_per_case, combinations=None):
    """
    Run classifier testing for all specified gripper-object combinations.
    
    Args:
        tests_per_case: Number of test grasps to run per combination
        combinations: List of combinations to test (e.g., ['pr2-cuboid', 'sdh-cylinder'])
                     If None, tests all 4 combinations
    """
    # Define all possible combinations
    all_combinations = [
        ("pr2", "cuboid"),
        ("pr2", "cylinder"),
        ("sdh", "cuboid"),
        ("sdh", "cylinder")
    ]
    
    # Filter combinations if specified
    if combinations:
        combo_dict = {f"{g}-{o}": (g, o) for g, o in all_combinations}
        selected = [combo_dict[c] for c in combinations if c in combo_dict]
        if not selected:
            print("[ERROR] No valid combinations specified. Available: pr2-cuboid, pr2-cylinder, sdh-cuboid, sdh-cylinder")
            return
    else:
        selected = all_combinations
    
    # Print configuration
    print_banner("BATCH CLASSIFIER TESTING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tests per combination: {tests_per_case}")
    print(f"Total combinations: {len(selected)}")
    print(f"Total tests this run: {tests_per_case * len(selected)}")
    print(f"\nCombinations to test:")
    for gripper, shape in selected:
        print(f"  - {gripper.upper()} gripper × {shape}")
    print()
    
    # Track statistics
    start_time = time.time()
    results = []
    
    # Run each combination
    for idx, (gripper_type, object_type) in enumerate(selected, 1):
        combo_name = f"{gripper_type.upper()} × {object_type}"
        print_banner(f"Combination {idx}/{len(selected)}: {combo_name}")
        
        combo_start = time.time()
        
        try:
            # Check if model exists
            models_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            model_file = os.path.join(models_folder, f"{gripper_type}_{object_type}_grasp_model.pkl")
            
            if not os.path.exists(model_file):
                print(f"[WARNING] Model file not found: {model_file}")
                print(f"[INFO] Skipping {combo_name}. Please train the model first.")
                results.append({
                    'combination': combo_name,
                    'tests': 0,
                    'duration': 0,
                    'status': 'SKIPPED (no model)'
                })
                continue
            
            # Call the test function from main.py
            print(f"[INFO] Starting {tests_per_case} test grasps...")
            test_classifier(
                object_type=object_type,
                num_tests=tests_per_case,
                gripper_type=gripper_type
            )
            
            combo_duration = time.time() - combo_start
            results.append({
                'combination': combo_name,
                'tests': tests_per_case,
                'duration': combo_duration,
                'status': 'SUCCESS'
            })
            
            print(f"\n[SUCCESS] Completed {combo_name} in {combo_duration:.1f} seconds")
            print(f"           ({combo_duration/tests_per_case:.2f} seconds per test)")
            
        except KeyboardInterrupt:
            print("\n[WARNING] User interrupted batch testing.")
            results.append({
                'combination': combo_name,
                'tests': 0,
                'duration': time.time() - combo_start,
                'status': 'INTERRUPTED'
            })
            break
            
        except Exception as e:
            combo_duration = time.time() - combo_start
            print(f"\n[ERROR] Failed to test {combo_name}: {str(e)}")
            results.append({
                'combination': combo_name,
                'tests': 0,
                'duration': combo_duration,
                'status': f'FAILED: {str(e)}'
            })
    
    # Print final summary
    total_duration = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_tests = sum(r['tests'] for r in results)
    
    print_banner("BATCH TESTING SUMMARY")
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Successful combinations: {successful}/{len(selected)}")
    print(f"Total tests run: {total_tests}")
    if total_tests > 0:
        print(f"Average time per test: {total_duration/total_tests:.2f} seconds")
    print("\nDetailed Results:")
    for result in results:
        status_icon = "✓" if result['status'] == 'SUCCESS' else ("⊗" if 'SKIPPED' in result['status'] else "✗")
        print(f"  {status_icon} {result['combination']:20} | "
              f"Tests: {result['tests']:4} | "
              f"Duration: {result['duration']:6.1f}s | "
              f"Status: {result['status']}")
    
    print("\n[INFO] Test results location: ./data/")
    print("[INFO] Files: test_results_{gripper}_{shape}.csv")
    print("=" * 70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch test trained classifiers for all gripper-object combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test 150 grasps per combination (600 total)
  python batch_test.py --tests 150
  
  # Test 50 grasps per combination
  python batch_test.py --tests 50
  
  # Test only specific combinations
  python batch_test.py --tests 100 --combinations pr2-cuboid sdh-cylinder
  
  # Quick test run (10 tests each)
  python batch_test.py --tests 10
        """
    )
    
    parser.add_argument(
        '--tests',
        type=int,
        required=True,
        help='Number of test grasps per combination (minimum 10 recommended)'
    )
    
    parser.add_argument(
        '--combinations',
        nargs='+',
        choices=['pr2-cuboid', 'pr2-cylinder', 'sdh-cuboid', 'sdh-cylinder'],
        help='Specific combinations to test (default: all 4)'
    )
    
    args = parser.parse_args()
    
    # Validate tests
    if args.tests < 1:
        print("[ERROR] Number of tests must be at least 1")
        sys.exit(1)
    
    if args.tests < 10:
        print("[WARNING] Fewer than 10 tests may not provide reliable statistics.")
    
    # Run batch testing
    try:
        run_batch_testing(
            tests_per_case=args.tests,
            combinations=args.combinations
        )
    except KeyboardInterrupt:
        print("\n[INFO] Batch testing interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
