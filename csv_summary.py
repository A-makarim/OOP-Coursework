"""
csv_summary.py - Detailed CSV Statistics for All Project Files

Scans all CSV files in the project and provides comprehensive statistics:
- Line counts (total rows excluding header)
- Success/Failure analysis
- Prediction accuracy (TP, FP, TN, FN)
- Per-file detailed breakdown
- Aggregate totals across all files

Usage:
    python csv_summary.py
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Get project root directory
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"


def analyze_csv(filepath):
    """Analyze a single CSV file and return detailed statistics."""
    stats = {
        'filename': filepath.name,
        'path': str(filepath.relative_to(PROJECT_ROOT)),
        'total_rows': 0,
        'columns': [],
        'has_success': False,
        'successes': 0,
        'failures': 0,
        'success_rate': 0.0,
        'has_predictions': False,
        'predicted_successes': 0,
        'predicted_failures': 0,
        'actual_successes': 0,
        'actual_failures': 0,
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'correct_predictions': 0,
        'prediction_accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
    }
    
    try:
        df = pd.read_csv(filepath)
        stats['total_rows'] = len(df)
        stats['columns'] = list(df.columns)
        
        # Check for Success column (training/test data)
        if 'Success' in df.columns:
            stats['has_success'] = True
            stats['successes'] = int((df['Success'] == 1).sum())
            stats['failures'] = int((df['Success'] == 0).sum())
            if stats['total_rows'] > 0:
                stats['success_rate'] = (stats['successes'] / stats['total_rows']) * 100
        
        # Check for prediction columns (test results)
        has_predicted = 'Predicted Success' in df.columns
        has_actual = 'Actual Success' in df.columns
        
        if has_predicted and has_actual:
            stats['has_predictions'] = True
            
            # Get prediction and actual values
            predicted = df['Predicted Success']
            actual = df['Actual Success']
            
            stats['predicted_successes'] = int((predicted == 1).sum())
            stats['predicted_failures'] = int((predicted == 0).sum())
            stats['actual_successes'] = int((actual == 1).sum())
            stats['actual_failures'] = int((actual == 0).sum())
            
            # Calculate confusion matrix values
            stats['true_positives'] = int(((predicted == 1) & (actual == 1)).sum())
            stats['true_negatives'] = int(((predicted == 0) & (actual == 0)).sum())
            stats['false_positives'] = int(((predicted == 1) & (actual == 0)).sum())
            stats['false_negatives'] = int(((predicted == 0) & (actual == 1)).sum())
            
            # Calculate accuracy metrics
            stats['correct_predictions'] = stats['true_positives'] + stats['true_negatives']
            if stats['total_rows'] > 0:
                stats['prediction_accuracy'] = (stats['correct_predictions'] / stats['total_rows']) * 100
            
            # Calculate precision, recall, F1
            if (stats['true_positives'] + stats['false_positives']) > 0:
                stats['precision'] = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
            if (stats['true_positives'] + stats['false_negatives']) > 0:
                stats['recall'] = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
            if (stats['precision'] + stats['recall']) > 0:
                stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
    
    except Exception as e:
        stats['error'] = str(e)
    
    return stats


def print_file_stats(stats, index):
    """Print detailed statistics for a single file."""
    print(f"\n{'='*80}")
    print(f"FILE #{index}: {stats['filename']}")
    print(f"{'='*80}")
    print(f"Path: {stats['path']}")
    print(f"Total Rows: {stats['total_rows']:,}")
    
    if stats.get('error'):
        print(f"ERROR: {stats['error']}")
        return
    
    print(f"Columns: {len(stats['columns'])}")
    
    # Success/Failure stats
    if stats['has_success']:
        print(f"\n--- SUCCESS ANALYSIS ---")
        print(f"Successes:     {stats['successes']:>6,} ({stats['success_rate']:>6.2f}%)")
        print(f"Failures:      {stats['failures']:>6,} ({100-stats['success_rate']:>6.2f}%)")
    
    # Prediction stats
    if stats['has_predictions']:
        print(f"\n--- PREDICTION ANALYSIS ---")
        print(f"Predicted Success:  {stats['predicted_successes']:>5}")
        print(f"Predicted Failure:  {stats['predicted_failures']:>5}")
        print(f"Actual Success:     {stats['actual_successes']:>5}")
        print(f"Actual Failure:     {stats['actual_failures']:>5}")
        
        print(f"\n--- CONFUSION MATRIX ---")
        print(f"True Positives  (TP): {stats['true_positives']:>5}")
        print(f"True Negatives  (TN): {stats['true_negatives']:>5}")
        print(f"False Positives (FP): {stats['false_positives']:>5}")
        print(f"False Negatives (FN): {stats['false_negatives']:>5}")
        
        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"Correct Predictions: {stats['correct_predictions']:>5} / {stats['total_rows']}")
        print(f"Accuracy:   {stats['prediction_accuracy']:>6.2f}%")
        print(f"Precision:  {stats['precision']:>6.2%}")
        print(f"Recall:     {stats['recall']:>6.2%}")
        print(f"F1-Score:   {stats['f1_score']:>6.2%}")


def print_aggregate_stats(all_stats):
    """Print aggregate statistics across all files."""
    print(f"\n\n{'#'*80}")
    print(f"{'AGGREGATE STATISTICS ACROSS ALL FILES':^80}")
    print(f"{'#'*80}\n")
    
    totals = {
        'files': len(all_stats),
        'total_rows': sum(s['total_rows'] for s in all_stats),
        'successes': sum(s['successes'] for s in all_stats if s['has_success']),
        'failures': sum(s['failures'] for s in all_stats if s['has_success']),
        'tp': sum(s['true_positives'] for s in all_stats if s['has_predictions']),
        'tn': sum(s['true_negatives'] for s in all_stats if s['has_predictions']),
        'fp': sum(s['false_positives'] for s in all_stats if s['has_predictions']),
        'fn': sum(s['false_negatives'] for s in all_stats if s['has_predictions']),
    }
    
    print(f"Total CSV Files Analyzed: {totals['files']}")
    print(f"Total Data Rows:          {totals['total_rows']:,}")
    
    if totals['successes'] + totals['failures'] > 0:
        success_rate = (totals['successes'] / (totals['successes'] + totals['failures'])) * 100
        print(f"\n--- OVERALL SUCCESS/FAILURE ---")
        print(f"Total Successes: {totals['successes']:>6,} ({success_rate:.2f}%)")
        print(f"Total Failures:  {totals['failures']:>6,} ({100-success_rate:.2f}%)")
    
    if totals['tp'] + totals['tn'] + totals['fp'] + totals['fn'] > 0:
        total_predictions = totals['tp'] + totals['tn'] + totals['fp'] + totals['fn']
        correct = totals['tp'] + totals['tn']
        accuracy = (correct / total_predictions) * 100
        
        print(f"\n--- OVERALL PREDICTIONS ---")
        print(f"Total Predictions:       {total_predictions:>6,}")
        print(f"Correct Predictions:     {correct:>6,}")
        print(f"Overall Accuracy:        {accuracy:>6.2f}%")
        
        print(f"\n--- AGGREGATE CONFUSION MATRIX ---")
        print(f"True Positives  (TP):    {totals['tp']:>6,}")
        print(f"True Negatives  (TN):    {totals['tn']:>6,}")
        print(f"False Positives (FP):    {totals['fp']:>6,}")
        print(f"False Negatives (FN):    {totals['fn']:>6,}")
        
        if (totals['tp'] + totals['fp']) > 0:
            precision = totals['tp'] / (totals['tp'] + totals['fp'])
            print(f"\nOverall Precision:       {precision:>6.2%}")
        
        if (totals['tp'] + totals['fn']) > 0:
            recall = totals['tp'] / (totals['tp'] + totals['fn'])
            print(f"Overall Recall:          {recall:>6.2%}")
    
    # File type breakdown
    print(f"\n--- FILE BREAKDOWN ---")
    training_files = [s for s in all_stats if 'grasp_data' in s['filename'] and 'updated' not in s['filename']]
    test_files = [s for s in all_stats if 'test_results' in s['filename']]
    updated_files = [s for s in all_stats if 'updated_grasp_data' in s['filename']]
    
    print(f"Training Data Files:     {len(training_files)}")
    print(f"Test Results Files:      {len(test_files)}")
    print(f"Updated Training Files:  {len(updated_files)}")
    print(f"Other CSV Files:         {totals['files'] - len(training_files) - len(test_files) - len(updated_files)}")


def save_summary_table(all_stats):
    """Save a summary table to CSV."""
    output_file = DATA_DIR / "csv_detailed_summary.csv"
    
    # Create summary dataframe
    summary_data = []
    for stats in all_stats:
        row = {
            'Filename': stats['filename'],
            'Path': stats['path'],
            'Total_Rows': stats['total_rows'],
            'Successes': stats['successes'] if stats['has_success'] else '',
            'Failures': stats['failures'] if stats['has_success'] else '',
            'Success_Rate_%': f"{stats['success_rate']:.2f}" if stats['has_success'] else '',
            'TP': stats['true_positives'] if stats['has_predictions'] else '',
            'TN': stats['true_negatives'] if stats['has_predictions'] else '',
            'FP': stats['false_positives'] if stats['has_predictions'] else '',
            'FN': stats['false_negatives'] if stats['has_predictions'] else '',
            'Accuracy_%': f"{stats['prediction_accuracy']:.2f}" if stats['has_predictions'] else '',
            'Precision': f"{stats['precision']:.4f}" if stats['has_predictions'] else '',
            'Recall': f"{stats['recall']:.4f}" if stats['has_predictions'] else '',
            'F1_Score': f"{stats['f1_score']:.4f}" if stats['has_predictions'] else '',
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"\n\nDetailed summary table saved to: {output_file}")


def main():
    """Main function to scan and analyze all CSV files."""
    print("="*80)
    print(f"{'CSV FILE ANALYSIS - DETAILED REPORT':^80}")
    print("="*80)
    print(f"\nScanning directory: {PROJECT_ROOT}")
    
    # Find all CSV files
    csv_files = sorted(PROJECT_ROOT.rglob('*.csv'))
    
    # Filter out summary files and venv
    csv_files = [f for f in csv_files if 'venv' not in str(f) and 'csv_summary' not in f.name]
    
    print(f"Found {len(csv_files)} CSV file(s)\n")
    
    # Analyze each file
    all_stats = []
    for idx, csv_file in enumerate(csv_files, 1):
        stats = analyze_csv(csv_file)
        all_stats.append(stats)
        print_file_stats(stats, idx)
    
    # Print aggregate statistics
    if all_stats:
        print_aggregate_stats(all_stats)
        save_summary_table(all_stats)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
