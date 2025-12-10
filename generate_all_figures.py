"""
Regenerate all figures for LaTeX report with updated statistics
Based on improved test results (balanced confusion matrices)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = Path("latex_report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("REGENERATING ALL FIGURES WITH UPDATED STATISTICS")
print("="*80)


def load_training_data(config):
    """Load training data"""
    filepath = f"data/grasp_data_{config}.csv"
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    return None


def load_test_results(config):
    """Load test results"""
    filepath = f"data/test_results_{config}.csv"
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    return None


def plot_training_analysis(config, config_name):
    """Generate training analysis plots"""
    print(f"\n📊 Generating training analysis: {config_name}")
    
    df = load_training_data(config)
    if df is None:
        print(f"   ⚠️  Training data not found for {config}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training Data Analysis: {config_name}', fontsize=14, fontweight='bold')
    
    # Success distribution
    success_counts = df['Success'].value_counts()
    axes[0, 0].bar(['Failure', 'Success'], 
                   [success_counts.get(0, 0), success_counts.get(1, 0)],
                   color=['#ff6b6b', '#51cf66'])
    axes[0, 0].set_title('Success Distribution')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rate text
    total = len(df)
    successes = df['Success'].sum()
    success_rate = (successes / total) * 100
    axes[0, 0].text(0.5, 0.95, f'Total: {total} samples\nSuccess: {successes} ({success_rate:.1f}%)',
                    transform=axes[0, 0].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Delta Z distribution
    if 'Delta Z' in df.columns:
        axes[0, 1].hist(df[df['Success'] == 0]['Delta Z'], bins=30, alpha=0.6, 
                       color='red', label='Failure', density=True)
        axes[0, 1].hist(df[df['Success'] == 1]['Delta Z'], bins=30, alpha=0.6, 
                       color='green', label='Success', density=True)
        axes[0, 1].axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='5cm threshold')
        axes[0, 1].set_title('Lift Height Distribution')
        axes[0, 1].set_xlabel('Delta Z (m)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Position distribution
    axes[1, 0].scatter(df[df['Success'] == 0]['Position X'], 
                      df[df['Success'] == 0]['Position Y'],
                      c='red', alpha=0.3, s=10, label='Failure')
    axes[1, 0].scatter(df[df['Success'] == 1]['Position X'], 
                      df[df['Success'] == 1]['Position Y'],
                      c='green', alpha=0.5, s=10, label='Success')
    axes[1, 0].set_title('Spatial Distribution (Top View)')
    axes[1, 0].set_xlabel('X Position (m)')
    axes[1, 0].set_ylabel('Y Position (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Orientation distribution
    if 'Orientation Roll' in df.columns:
        axes[1, 1].scatter(df[df['Success'] == 0]['Orientation Roll'], 
                          df[df['Success'] == 0]['Orientation Pitch'],
                          c='red', alpha=0.3, s=10, label='Failure')
        axes[1, 1].scatter(df[df['Success'] == 1]['Orientation Roll'], 
                          df[df['Success'] == 1]['Orientation Pitch'],
                          c='green', alpha=0.5, s=10, label='Success')
        axes[1, 1].set_title('Orientation Distribution')
        axes[1, 1].set_xlabel('Roll (rad)')
        axes[1, 1].set_ylabel('Pitch (rad)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"training_analysis_{config}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_test_results(config, config_name):
    """Generate test results visualization"""
    print(f"\n📊 Generating test results: {config_name}")
    
    df = load_test_results(config)
    if df is None:
        print(f"   ⚠️  Test results not found for {config}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Test Results: {config_name}', fontsize=14, fontweight='bold')
    
    # Confusion matrix
    cm = confusion_matrix(df['Actual Success'], df['Predicted Success'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Failure', 'Success'],
                yticklabels=['Failure', 'Success'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.1f}%\nRecall: {recall:.1f}%\nPrecision: {precision:.1f}%\nF1: {f1:.3f}'
    axes[0].text(1.5, 0.5, metrics_text, transform=axes[0].transData,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, verticalalignment='center')
    
    # Spatial distribution
    correct = df['Match'] == True
    incorrect = df['Match'] == False
    
    axes[1].scatter(df[correct]['Position X'], df[correct]['Position Y'],
                   c='green', alpha=0.6, s=30, label='Correct', marker='o')
    axes[1].scatter(df[incorrect]['Position X'], df[incorrect]['Position Y'],
                   c='red', alpha=0.6, s=30, label='Incorrect', marker='x')
    axes[1].set_title('Prediction Results (Top View)')
    axes[1].set_xlabel('X Position (m)')
    axes[1].set_ylabel('Y Position (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"test_results_{config}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_roc_curve_single(config, config_name):
    """Generate ROC curve"""
    print(f"\n📊 Generating ROC curve: {config_name}")
    
    df = load_test_results(config)
    if df is None:
        print(f"   ⚠️  Test results not found for {config}")
        return
    
    if 'Confidence' not in df.columns:
        print(f"   ⚠️  No confidence scores for {config}")
        return
    
    fpr, tpr, thresholds = roc_curve(df['Actual Success'], df['Confidence'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: {config_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    output_path = OUTPUT_DIR / f"roc_curve_{config}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_confusion_matrices_all():
    """Generate combined confusion matrices for all configs"""
    print(f"\n📊 Generating combined confusion matrices")
    
    configs = [
        ('pr2_cuboid', 'PR2-Cuboid'),
        ('pr2_cylinder', 'PR2-Cylinder'),
        ('sdh_cuboid', 'SDH-Cuboid'),
        ('sdh_cylinder', 'SDH-Cylinder')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices: All Configurations', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    for idx, (config, name) in enumerate(configs):
        df = load_test_results(config)
        if df is None:
            continue
        
        cm = confusion_matrix(df['Actual Success'], df['Predicted Success'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Failure', 'Success'],
                    yticklabels=['Failure', 'Success'],
                    cbar=True)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        axes[idx].set_title(f'{name}\nAcc: {accuracy:.1f}%, F1: {f1:.3f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_dataset_sizes():
    """Generate dataset size comparison"""
    print(f"\n📊 Generating dataset sizes plot")
    
    configs = [
        ('pr2_cuboid', 'PR2-Cuboid'),
        ('pr2_cylinder', 'PR2-Cylinder'),
        ('sdh_cuboid', 'SDH-Cuboid'),
        ('sdh_cylinder', 'SDH-Cylinder')
    ]
    
    train_sizes = []
    test_sizes = []
    labels = []
    
    for config, name in configs:
        train_df = load_training_data(config)
        test_df = load_test_results(config)
        
        if train_df is not None and test_df is not None:
            train_sizes.append(len(train_df))
            test_sizes.append(len(test_df))
            labels.append(name)
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_sizes, width, label='Training', color='steelblue')
    bars2 = ax.bar(x + width/2, test_sizes, width, label='Testing', color='coral')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Dataset Sizes: Training vs Testing', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "dataset_sizes.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_training_distribution():
    """Generate overall training distribution"""
    print(f"\n📊 Generating training distribution plot")
    
    configs = [
        ('pr2_cuboid', 'PR2-Cuboid'),
        ('pr2_cylinder', 'PR2-Cylinder'),
        ('sdh_cuboid', 'SDH-Cuboid'),
        ('sdh_cylinder', 'SDH-Cylinder')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Lift Height Distributions (Training Data)', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    for idx, (config, name) in enumerate(configs):
        df = load_training_data(config)
        if df is None or 'Delta Z' not in df.columns:
            continue
        
        axes[idx].hist(df[df['Success'] == 0]['Delta Z'], bins=40, alpha=0.6,
                      color='red', label='Failure', density=True)
        axes[idx].hist(df[df['Success'] == 1]['Delta Z'], bins=40, alpha=0.6,
                      color='green', label='Success', density=True)
        axes[idx].axvline(x=0.05, color='black', linestyle='--', linewidth=2,
                         label='5cm threshold')
        
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Lift Height (m)')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics
        success_rate = df['Success'].mean() * 100
        avg_delta = df['Delta Z'].mean()
        axes[idx].text(0.95, 0.95, f'Success: {success_rate:.1f}%\nAvg Δz: {avg_delta:.3f}m',
                      transform=axes[idx].transAxes, ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "training_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_performance_comparison():
    """Generate performance comparison across configs"""
    print(f"\n📊 Generating performance comparison plot")
    
    configs = [
        ('pr2_cuboid', 'PR2\nCuboid'),
        ('pr2_cylinder', 'PR2\nCylinder'),
        ('sdh_cuboid', 'SDH\nCuboid'),
        ('sdh_cylinder', 'SDH\nCylinder')
    ]
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    labels = []
    
    for config, name in configs:
        df = load_test_results(config)
        if df is None:
            continue
        
        cm = confusion_matrix(df['Actual Success'], df['Predicted Success'])
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        accuracies.append(accuracy * 100)
        precisions.append(precision * 100)
        recalls.append(recall * 100)
        f1_scores.append(f1 * 100)
        labels.append(name)
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', color='coral')
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', color='lightgreen')
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='gold')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "performance_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_gripper_comparison():
    """Generate PR2 vs SDH comparison"""
    print(f"\n📊 Generating gripper comparison plot")
    
    pr2_configs = ['pr2_cuboid', 'pr2_cylinder']
    sdh_configs = ['sdh_cuboid', 'sdh_cylinder']
    
    pr2_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    sdh_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for config in pr2_configs:
        df = load_test_results(config)
        if df is None:
            continue
        
        cm = confusion_matrix(df['Actual Success'], df['Predicted Success'])
        tn, fp, fn, tp = cm.ravel()
        
        pr2_metrics['accuracy'].append((tp + tn) / (tp + tn + fp + fn))
        pr2_metrics['precision'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        pr2_metrics['recall'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        pr2_metrics['f1'].append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
    
    for config in sdh_configs:
        df = load_test_results(config)
        if df is None:
            continue
        
        cm = confusion_matrix(df['Actual Success'], df['Predicted Success'])
        tn, fp, fn, tp = cm.ravel()
        
        sdh_metrics['accuracy'].append((tp + tn) / (tp + tn + fp + fn))
        sdh_metrics['precision'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        sdh_metrics['recall'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        sdh_metrics['f1'].append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
    
    # Average metrics
    pr2_avg = {k: np.mean(v) * 100 for k, v in pr2_metrics.items()}
    sdh_avg = {k: np.mean(v) * 100 for k, v in sdh_metrics.items()}
    
    metrics = list(pr2_avg.keys())
    pr2_values = [pr2_avg[m] for m in metrics]
    sdh_values = [sdh_avg[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, pr2_values, width, label='PR2 (Parallel-jaw)', 
                   color='steelblue')
    bars2 = ax.bar(x + width/2, sdh_values, width, label='SDH (3-finger)', 
                   color='coral')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Gripper Type Comparison (Averaged)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "gripper_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def generate_statistics_json():
    """Generate statistics JSON file"""
    print(f"\n📊 Generating statistics.json")
    
    configs = [
        ('pr2_cuboid', 'PR2-Cuboid'),
        ('pr2_cylinder', 'PR2-Cylinder'),
        ('sdh_cuboid', 'SDH-Cuboid'),
        ('sdh_cylinder', 'SDH-Cylinder')
    ]
    
    stats = {}
    
    for config, name in configs:
        train_df = load_training_data(config)
        test_df = load_test_results(config)
        
        if train_df is not None and test_df is not None:
            cm = confusion_matrix(test_df['Actual Success'], test_df['Predicted Success'])
            tn, fp, fn, tp = cm.ravel()
            
            stats[config] = {
                'name': name,
                'training_samples': len(train_df),
                'training_success_rate': float(train_df['Success'].mean()),
                'test_samples': len(test_df),
                'test_accuracy': float((tp + tn) / (tp + tn + fp + fn)),
                'test_precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'test_recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'test_f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
                'confusion_matrix': {
                    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                }
            }
    
    output_path = OUTPUT_DIR / "statistics.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ Saved: {output_path}")


def main():
    """Main function to regenerate all figures"""
    
    configs = [
        ('pr2_cuboid', 'PR2-Cuboid'),
        ('pr2_cylinder', 'PR2-Cylinder'),
        ('sdh_cuboid', 'SDH-Cuboid'),
        ('sdh_cylinder', 'SDH-Cylinder')
    ]
    
    # Individual config plots
    for config, name in configs:
        plot_training_analysis(config, name)
        plot_test_results(config, name)
        plot_roc_curve_single(config, name)
    
    # Combined plots
    plot_confusion_matrices_all()
    plot_dataset_sizes()
    plot_training_distribution()
    plot_performance_comparison()
    plot_gripper_comparison()
    
    # Statistics
    generate_statistics_json()
    
    print("\n" + "="*80)
    print("✅ ALL FIGURES REGENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
