"""
Generate bar plots showing success/failure distribution across roll angles for training data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

OUTPUT_DIR = Path("latex_report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_training_roll_analysis():
    """Create bar plots showing success/failure by roll angle bins for training data"""
    
    configs = ['pr2_cuboid', 'pr2_cylinder', 'sdh_cuboid', 'sdh_cylinder']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        # Load training data
        df = pd.read_csv(f'data/grasp_data_{config}.csv')
        
        # Create roll bins (convert radians to degrees for better readability)
        df['Roll (deg)'] = np.degrees(df['Orientation Roll'])
        
        # Define bins: -180 to 180 degrees in 30-degree intervals
        bins = np.arange(-180, 181, 30)
        bin_labels = [f'{int(bins[i])}°' for i in range(len(bins)-1)]
        
        # Bin the data
        df['Roll Bin'] = pd.cut(df['Roll (deg)'], bins=bins, labels=bin_labels, include_lowest=True)
        
        # Separate success and failure
        success = df[df['Success'] == 1]
        failure = df[df['Success'] == 0]
        
        # Count samples in each bin
        success_counts = success.groupby('Roll Bin', observed=False).size()
        failure_counts = failure.groupby('Roll Bin', observed=False).size()
        
        # Ensure all bins are present
        all_bins = pd.CategoricalIndex(bin_labels, categories=bin_labels, ordered=True)
        success_counts = success_counts.reindex(all_bins, fill_value=0)
        failure_counts = failure_counts.reindex(all_bins, fill_value=0)
        
        # Create bar positions
        x = np.arange(len(bin_labels))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, failure_counts, width, label='Failure', 
                       color='#E74C3C', alpha=0.8, edgecolor='darkred', linewidth=0.5)
        bars2 = ax.bar(x + width/2, success_counts, width, label='Success', 
                       color='#2ECC71', alpha=0.8, edgecolor='darkgreen', linewidth=0.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=6)
        
        # Formatting
        ax.set_xlabel('Orientation Roll (degrees)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Frequency (Number of Samples)', fontsize=9, fontweight='bold')
        ax.set_title(f'{config.upper().replace("_", "-")}: Training Data Roll Distribution',
                    fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics box
        total_samples = len(df)
        success_rate = (len(success) / total_samples * 100)
        stats_text = (
            f'Total: {total_samples}\n'
            f'Success: {len(success)} ({success_rate:.1f}%)\n'
            f'Failure: {len(failure)} ({(100-success_rate):.1f}%)'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "training_roll_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_training_pitch_analysis():
    """Create bar plots showing success/failure by pitch angle bins for training data"""
    
    configs = ['pr2_cuboid', 'pr2_cylinder', 'sdh_cuboid', 'sdh_cylinder']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        # Load training data
        df = pd.read_csv(f'data/grasp_data_{config}.csv')
        
        # Create pitch bins (convert radians to degrees for better readability)
        df['Pitch (deg)'] = np.degrees(df['Orientation Pitch'])
        
        # Define bins: -90 to 90 degrees in 15-degree intervals
        bins = np.arange(-90, 91, 15)
        bin_labels = [f'{int(bins[i])}°' for i in range(len(bins)-1)]
        
        # Bin the data
        df['Pitch Bin'] = pd.cut(df['Pitch (deg)'], bins=bins, labels=bin_labels, include_lowest=True)
        
        # Separate success and failure
        success = df[df['Success'] == 1]
        failure = df[df['Success'] == 0]
        
        # Count samples in each bin
        success_counts = success.groupby('Pitch Bin', observed=False).size()
        failure_counts = failure.groupby('Pitch Bin', observed=False).size()
        
        # Ensure all bins are present
        all_bins = pd.CategoricalIndex(bin_labels, categories=bin_labels, ordered=True)
        success_counts = success_counts.reindex(all_bins, fill_value=0)
        failure_counts = failure_counts.reindex(all_bins, fill_value=0)
        
        # Create bar positions
        x = np.arange(len(bin_labels))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, failure_counts, width, label='Failure', 
                       color='#E74C3C', alpha=0.8, edgecolor='darkred', linewidth=0.5)
        bars2 = ax.bar(x + width/2, success_counts, width, label='Success', 
                       color='#2ECC71', alpha=0.8, edgecolor='darkgreen', linewidth=0.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=6)
        
        # Formatting
        ax.set_xlabel('Orientation Pitch (degrees)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Frequency (Number of Samples)', fontsize=9, fontweight='bold')
        ax.set_title(f'{config.upper().replace("_", "-")}: Training Data Pitch Distribution',
                    fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics box
        total_samples = len(df)
        success_rate = (len(success) / total_samples * 100)
        stats_text = (
            f'Total: {total_samples}\n'
            f'Success: {len(success)} ({success_rate:.1f}%)\n'
            f'Failure: {len(failure)} ({(100-success_rate):.1f}%)'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "training_pitch_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    """Generate all training orientation distribution figures"""
    print("="*80)
    print("GENERATING TRAINING ORIENTATION DISTRIBUTION FIGURES")
    print("="*80)
    
    print("\n1. Roll angle distribution...")
    create_training_roll_analysis()
    
    print("\n2. Pitch angle distribution...")
    create_training_pitch_analysis()
    
    print("\n" + "="*80)
    print("✅ TRAINING ORIENTATION DISTRIBUTION FIGURES GENERATED!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("📊 Files created:")
    print("   - training_roll_distribution.png")
    print("   - training_pitch_distribution.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
