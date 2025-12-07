"""
visualize.py - Visualization module for grasp planning data

Creates meaningful plots including:
- 3D success heatmaps
- Position distribution plots
- Orientation analysis
- Confusion matrices for test results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Get the directory where this script is located (OOPProject folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def visualize_training_data(object_type, gripper_type="pr2"):
    """
    Visualize training data with multiple plots:
    1. 3D scatter plot of grasp positions colored by success
    2. 2D heatmaps of success rates
    3. Orientation distribution
    4. Success rate statistics
    """
    data_folder = os.path.join(SCRIPT_DIR, "data")
    csv_file = os.path.join(data_folder, f"grasp_data_{gripper_type}_{object_type}.csv")
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] Training data file {csv_file} not found.")
        return
    
    print(f"[INFO] Loading training data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    if len(df) == 0:
        print("[ERROR] No data found in file.")
        return
    
    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Successful grasps: {df['Success'].sum()}")
    print(f"[INFO] Failed grasps: {len(df) - df['Success'].sum()}")
    print(f"[INFO] Success rate: {(df['Success'].sum() / len(df)) * 100:.2f}%")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Training Data Analysis: {gripper_type.upper()} - {object_type.capitalize()}', 
                 fontsize=16, fontweight='bold')
    
    # 1. 3D Scatter Plot of Positions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    successful = df[df['Success'] == 1]
    failed = df[df['Success'] == 0]
    
    ax1.scatter(successful['Position X'], successful['Position Y'], successful['Position Z'],
                c='green', marker='o', label='Success', alpha=0.6, s=50)
    ax1.scatter(failed['Position X'], failed['Position Y'], failed['Position Z'],
                c='red', marker='x', label='Failure', alpha=0.6, s=50)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('3D Grasp Position Distribution')
    ax1.legend()
    
    # 2. XY Plane Heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    # Create grid for heatmap
    x_bins = np.linspace(df['Position X'].min(), df['Position X'].max(), 15)
    y_bins = np.linspace(df['Position Y'].min(), df['Position Y'].max(), 15)
    
    heatmap_data = np.zeros((len(y_bins)-1, len(x_bins)-1))
    count_data = np.zeros((len(y_bins)-1, len(x_bins)-1))
    
    for _, row in df.iterrows():
        x_idx = np.digitize(row['Position X'], x_bins) - 1
        y_idx = np.digitize(row['Position Y'], y_bins) - 1
        if 0 <= x_idx < len(x_bins)-1 and 0 <= y_idx < len(y_bins)-1:
            heatmap_data[y_idx, x_idx] += row['Success']
            count_data[y_idx, x_idx] += 1
    
    # Calculate success rate per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        success_rate = np.where(count_data > 0, heatmap_data / count_data, np.nan)
    
    im = ax2.imshow(success_rate, cmap='RdYlGn', aspect='auto', origin='lower',
                    extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], vmin=0, vmax=1)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Success Rate Heatmap (XY Plane)')
    plt.colorbar(im, ax=ax2, label='Success Rate')
    
    # 3. XZ Plane Heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    z_bins = np.linspace(df['Position Z'].min(), df['Position Z'].max(), 15)
    
    heatmap_xz = np.zeros((len(z_bins)-1, len(x_bins)-1))
    count_xz = np.zeros((len(z_bins)-1, len(x_bins)-1))
    
    for _, row in df.iterrows():
        x_idx = np.digitize(row['Position X'], x_bins) - 1
        z_idx = np.digitize(row['Position Z'], z_bins) - 1
        if 0 <= x_idx < len(x_bins)-1 and 0 <= z_idx < len(z_bins)-1:
            heatmap_xz[z_idx, x_idx] += row['Success']
            count_xz[z_idx, x_idx] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        success_rate_xz = np.where(count_xz > 0, heatmap_xz / count_xz, np.nan)
    
    im2 = ax3.imshow(success_rate_xz, cmap='RdYlGn', aspect='auto', origin='lower',
                     extent=[x_bins[0], x_bins[-1], z_bins[0], z_bins[-1]], vmin=0, vmax=1)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Success Rate Heatmap (XZ Plane)')
    plt.colorbar(im2, ax=ax3, label='Success Rate')
    
    # 4. Orientation Distribution (Yaw)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist([successful['Orientation Yaw'], failed['Orientation Yaw']],
             bins=20, label=['Success', 'Failure'], color=['green', 'red'], alpha=0.6)
    ax4.set_xlabel('Yaw Angle (rad)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Yaw Orientation Distribution')
    ax4.legend()
    
    # 5. Height (Z) Distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist([successful['Position Z'], failed['Position Z']],
             bins=20, label=['Success', 'Failure'], color=['green', 'red'], alpha=0.6)
    ax5.set_xlabel('Z Position (m)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Height Distribution')
    ax5.legend()
    
    # 6. Statistics Box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    TRAINING DATA STATISTICS
    ========================
    Total Samples: {len(df)}
    Successful Grasps: {df['Success'].sum()}
    Failed Grasps: {len(df) - df['Success'].sum()}
    Success Rate: {(df['Success'].sum() / len(df)) * 100:.2f}%
    
    POSITION RANGES
    ========================
    X: [{df['Position X'].min():.3f}, {df['Position X'].max():.3f}]
    Y: [{df['Position Y'].min():.3f}, {df['Position Y'].max():.3f}]
    Z: [{df['Position Z'].min():.3f}, {df['Position Z'].max():.3f}]
    
    ORIENTATION RANGES (rad)
    ========================
    Roll:  [{df['Orientation Roll'].min():.3f}, {df['Orientation Roll'].max():.3f}]
    Pitch: [{df['Orientation Pitch'].min():.3f}, {df['Orientation Pitch'].max():.3f}]
    Yaw:   [{df['Orientation Yaw'].min():.3f}, {df['Orientation Yaw'].max():.3f}]
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    images_folder = os.path.join(SCRIPT_DIR, "images")
    os.makedirs(images_folder, exist_ok=True)
    plot_file = os.path.join(images_folder, f"training_analysis_{gripper_type}_{object_type}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved plot to {plot_file}")
    
    plt.show()


def visualize_test_results(object_type, gripper_type="pr2"):
    """
    Visualize test results with:
    1. Confusion matrix
    2. Prediction accuracy by position
    3. Confidence distribution
    4. Error analysis
    """
    data_folder = os.path.join(SCRIPT_DIR, "data")
    test_file = os.path.join(data_folder, f"test_results_{gripper_type}_{object_type}.csv")
    
    if not os.path.exists(test_file):
        print(f"[ERROR] Test results file {test_file} not found.")
        print("[INFO] Please run testing phase first (Option 3 in main menu)")
        return
    
    print(f"[INFO] Loading test results from {test_file}")
    df = pd.read_csv(test_file)
    
    if len(df) == 0:
        print("[ERROR] No test data found in file.")
        return
    
    # Calculate metrics
    accuracy = (df['Match'].sum() / len(df)) * 100
    
    print(f"\n[INFO] Total Tests: {len(df)}")
    print(f"[INFO] Correct Predictions: {df['Match'].sum()}")
    print(f"[INFO] Prediction Accuracy: {accuracy:.2f}%")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Test Results Analysis: {gripper_type.upper()} - {object_type.capitalize()}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(2, 3, 1)
    cm = confusion_matrix(df['Actual Success'], df['Predicted Success'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix')
    ax1.set_xticklabels(['Failure', 'Success'])
    ax1.set_yticklabels(['Failure', 'Success'])
    
    # 2. Prediction Confidence Distribution
    ax2 = fig.add_subplot(2, 3, 2)
    correct = df[df['Match'] == True]
    incorrect = df[df['Match'] == False]
    
    ax2.hist([correct['Confidence'], incorrect['Confidence']],
             bins=20, label=['Correct', 'Incorrect'], color=['green', 'red'], alpha=0.6)
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    
    # 3. 3D Scatter of Test Positions
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    correct_pred = df[df['Match'] == True]
    incorrect_pred = df[df['Match'] == False]
    
    ax3.scatter(correct_pred['Position X'], correct_pred['Position Y'], correct_pred['Position Z'],
                c='green', marker='o', label='Correct', alpha=0.7, s=60)
    ax3.scatter(incorrect_pred['Position X'], incorrect_pred['Position Y'], incorrect_pred['Position Z'],
                c='red', marker='X', label='Incorrect', alpha=0.7, s=60)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_zlabel('Z Position (m)')
    ax3.set_title('Test Positions (Color = Prediction Correctness)')
    ax3.legend()
    
    # 4. True Positives, False Positives, etc.
    ax4 = fig.add_subplot(2, 3, 4)
    tp = len(df[(df['Predicted Success'] == 1) & (df['Actual Success'] == 1)])
    fp = len(df[(df['Predicted Success'] == 1) & (df['Actual Success'] == 0)])
    tn = len(df[(df['Predicted Success'] == 0) & (df['Actual Success'] == 0)])
    fn = len(df[(df['Predicted Success'] == 0) & (df['Actual Success'] == 1)])
    
    categories = ['True\nPositive', 'False\nPositive', 'True\nNegative', 'False\nNegative']
    values = [tp, fp, tn, fn]
    colors = ['green', 'orange', 'lightgreen', 'red']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Count')
    ax4.set_title('Prediction Breakdown')
    ax4.set_ylim(0, max(values) * 1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Actual vs Predicted Success Rate
    ax5 = fig.add_subplot(2, 3, 5)
    actual_success_rate = (df['Actual Success'].sum() / len(df)) * 100
    predicted_success_rate = (df['Predicted Success'].sum() / len(df)) * 100
    
    ax5.bar(['Actual', 'Predicted'], [actual_success_rate, predicted_success_rate],
            color=['blue', 'orange'], alpha=0.7)
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Actual vs Predicted Success Rates')
    ax5.set_ylim(0, 100)
    
    for i, v in enumerate([actual_success_rate, predicted_success_rate]):
        ax5.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 6. Statistics Box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Precompute confidence metrics to avoid complex expressions inside f-strings
    mean_conf = df['Confidence'].mean() if 'Confidence' in df.columns and len(df) > 0 else 0.0
    correct_mean = correct['Confidence'].mean() if 'Confidence' in correct.columns and len(correct) > 0 else 0.0
    incorrect_mean = incorrect['Confidence'].mean() if 'Confidence' in incorrect.columns and len(incorrect) > 0 else 0.0

    stats_text = f"""
    TEST RESULTS STATISTICS
    ========================
    Total Tests: {len(df)}
    Correct Predictions: {df['Match'].sum()}
    Prediction Accuracy: {accuracy:.2f}%

    CLASSIFICATION METRICS
    ========================
    True Positives:  {tp}
    False Positives: {fp}
    True Negatives:  {tn}
    False Negatives: {fn}

    Precision: {precision:.3f}
    Recall:    {recall:.3f}
    F1-Score:  {f1_score:.3f}

    CONFIDENCE METRICS
    ========================
    Mean Confidence: {mean_conf:.3f}
    Correct Mean:    {correct_mean:.3f}
    Incorrect Mean:  {incorrect_mean:.3f}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    images_folder = os.path.join(SCRIPT_DIR, "images")
    os.makedirs(images_folder, exist_ok=True)
    plot_file = os.path.join(images_folder, f"test_results_{gripper_type}_{object_type}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved plot to {plot_file}")
    
    plt.show()
    
    # Print classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(df['Actual Success'], df['Predicted Success'],
                                target_names=['Failure', 'Success']))
