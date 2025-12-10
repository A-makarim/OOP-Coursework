"""
Generate 3D scatter plots of training grasp data showing success/failure spatial distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

OUTPUT_DIR = Path("latex_report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define consistent axis limits for all plots (based on reasonable ranges)
AXIS_LIMITS = {
    'x': (-0.5, 0.5),
    'y': (-0.5, 0.5),
    'z': (0.0, 1.0),
    'roll': (-np.pi, np.pi),
    'pitch': (-1.6, 1.6),
    'yaw': (-np.pi, np.pi)
}

def create_3d_scatter_plot(config_name):
    """Create 3D scatter plot for a configuration showing success/failure distribution"""
    
    # Load training data
    df = pd.read_csv(f'data/grasp_data_{config_name}.csv')
    
    # Filter outliers (keep data within reasonable bounds)
    df = df[
        (df['Position X'] >= AXIS_LIMITS['x'][0]) & (df['Position X'] <= AXIS_LIMITS['x'][1]) &
        (df['Position Y'] >= AXIS_LIMITS['y'][0]) & (df['Position Y'] <= AXIS_LIMITS['y'][1]) &
        (df['Position Z'] >= AXIS_LIMITS['z'][0]) & (df['Position Z'] <= AXIS_LIMITS['z'][1])
    ]
    
    print(f"  {config_name}: {len(df)} samples after outlier filtering")
    
    # Separate success and failure
    success = df[df['Success'] == 1]
    failure = df[df['Success'] == 0]
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(failure['Position X'], failure['Position Y'], failure['Position Z'],
               c='red', marker='o', s=10, alpha=0.3, label='Failure')
    ax1.scatter(success['Position X'], success['Position Y'], success['Position Z'],
               c='green', marker='^', s=20, alpha=0.6, label='Success')
    ax1.set_xlabel('X Position (m)', fontsize=8)
    ax1.set_ylabel('Y Position (m)', fontsize=8)
    ax1.set_zlabel('Z Position (m)', fontsize=8)
    ax1.set_xlim(AXIS_LIMITS['x'])
    ax1.set_ylim(AXIS_LIMITS['y'])
    ax1.set_zlim(AXIS_LIMITS['z'])
    ax1.set_title(f'{config_name.upper().replace("_", "-")}: 3D Position Distribution', 
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.view_init(elev=20, azim=45)
    
    # XY plane view (top-down)
    ax2 = fig.add_subplot(222)
    ax2.scatter(failure['Position X'], failure['Position Y'],
               c='red', marker='o', s=10, alpha=0.3, label='Failure')
    ax2.scatter(success['Position X'], success['Position Y'],
               c='green', marker='^', s=20, alpha=0.6, label='Success')
    ax2.set_xlabel('X Position (m)', fontsize=8)
    ax2.set_ylabel('Y Position (m)', fontsize=8)
    ax2.set_xlim(AXIS_LIMITS['x'])
    ax2.set_ylim(AXIS_LIMITS['y'])
    ax2.set_title('Top View (XY Plane)', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.set_aspect('equal')
    
    # XZ plane view (side view)
    ax3 = fig.add_subplot(223)
    ax3.scatter(failure['Position X'], failure['Position Z'],
               c='red', marker='o', s=10, alpha=0.3, label='Failure')
    ax3.scatter(success['Position X'], success['Position Z'],
               c='green', marker='^', s=20, alpha=0.6, label='Success')
    ax3.set_xlabel('X Position (m)', fontsize=8)
    ax3.set_ylabel('Z Position (m)', fontsize=8)
    ax3.set_xlim(AXIS_LIMITS['x'])
    ax3.set_ylim(AXIS_LIMITS['z'])
    ax3.set_title('Side View (XZ Plane)', fontsize=9, fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.4, color='gray', linewidth=0.5, alpha=0.3, linestyle='--', label='Object center')
    ax3.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Orientation distribution (roll vs pitch)
    ax4 = fig.add_subplot(224)
    ax4.scatter(failure['Orientation Roll'], failure['Orientation Pitch'],
               c='red', marker='o', s=10, alpha=0.3, label='Failure')
    ax4.scatter(success['Orientation Roll'], success['Orientation Pitch'],
               c='green', marker='^', s=20, alpha=0.6, label='Success')
    ax4.set_xlabel('Roll (rad)', fontsize=8)
    ax4.set_ylabel('Pitch (rad)', fontsize=8)
    ax4.set_xlim(AXIS_LIMITS['roll'])
    ax4.set_ylim(AXIS_LIMITS['pitch'])
    ax4.set_title('Orientation Distribution', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax4.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Add statistics text
    stats_text = (
        f"Total: {len(df)} samples\n"
        f"Success: {len(success)} ({len(success)/len(df)*100:.1f}%)\n"
        f"Failure: {len(failure)} ({len(failure)/len(df)*100:.1f}%)\n"
        f"Avg Δz: {df['Delta Z'].mean():.3f}m"
    )
    fig.text(0.02, 0.98, stats_text, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = OUTPUT_DIR / f"3d_visualization_{config_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_combined_3d_plot():
    """Create combined 3D plot showing all configurations"""
    
    configs = ['pr2_cuboid', 'pr2_cylinder', 'sdh_cuboid', 'sdh_cylinder']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    fig = plt.figure(figsize=(14, 6))
    
    # Combined 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    
    for config, color in zip(configs, colors):
        df = pd.read_csv(f'data/grasp_data_{config}.csv')
        # Filter outliers
        df = df[
            (df['Position X'] >= AXIS_LIMITS['x'][0]) & (df['Position X'] <= AXIS_LIMITS['x'][1]) &
            (df['Position Y'] >= AXIS_LIMITS['y'][0]) & (df['Position Y'] <= AXIS_LIMITS['y'][1]) &
            (df['Position Z'] >= AXIS_LIMITS['z'][0]) & (df['Position Z'] <= AXIS_LIMITS['z'][1])
        ]
        success = df[df['Success'] == 1]
        
        ax1.scatter(success['Position X'], success['Position Y'], success['Position Z'],
                   c=color, marker='o', s=15, alpha=0.5, 
                   label=config.upper().replace('_', '-'))
    
    ax1.set_xlabel('X Position (m)', fontsize=9)
    ax1.set_ylabel('Y Position (m)', fontsize=9)
    ax1.set_zlabel('Z Position (m)', fontsize=9)
    ax1.set_xlim(AXIS_LIMITS['x'])
    ax1.set_ylim(AXIS_LIMITS['y'])
    ax1.set_zlim(AXIS_LIMITS['z'])
    ax1.set_title('Successful Grasps: All Configurations (3D)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.view_init(elev=20, azim=45)
    
    # Top view (XY plane)
    ax2 = fig.add_subplot(122)
    
    for config, color in zip(configs, colors):
        df = pd.read_csv(f'data/grasp_data_{config}.csv')
        # Filter outliers
        df = df[
            (df['Position X'] >= AXIS_LIMITS['x'][0]) & (df['Position X'] <= AXIS_LIMITS['x'][1]) &
            (df['Position Y'] >= AXIS_LIMITS['y'][0]) & (df['Position Y'] <= AXIS_LIMITS['y'][1])
        ]
        success = df[df['Success'] == 1]
        
        ax2.scatter(success['Position X'], success['Position Y'],
                   c=color, marker='o', s=15, alpha=0.5,
                   label=config.upper().replace('_', '-'))
    
    ax2.set_xlabel('X Position (m)', fontsize=9)
    ax2.set_ylabel('Y Position (m)', fontsize=9)
    ax2.set_xlim(AXIS_LIMITS['x'])
    ax2.set_ylim(AXIS_LIMITS['y'])
    ax2.set_title('Successful Grasps: Top View (XY Plane)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "3d_visualization_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_noise_visualization():
    """Create visualization showing effect of noise on sampling"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    configs = ['pr2_cuboid', 'pr2_cylinder', 'sdh_cuboid', 'sdh_cylinder']
    
    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        df = pd.read_csv(f'data/grasp_data_{config}.csv')
        
        # Filter outliers
        df = df[
            (df['Position X'] >= AXIS_LIMITS['x'][0]) & (df['Position X'] <= AXIS_LIMITS['x'][1]) &
            (df['Position Y'] >= AXIS_LIMITS['y'][0]) & (df['Position Y'] <= AXIS_LIMITS['y'][1]) &
            (df['Position Z'] >= AXIS_LIMITS['z'][0]) & (df['Position Z'] <= AXIS_LIMITS['z'][1])
        ]
        
        # Calculate radial distance from origin
        df['radial_dist'] = np.sqrt(df['Position X']**2 + df['Position Y']**2)
        
        # Separate by success
        success = df[df['Success'] == 1]
        failure = df[df['Success'] == 0]
        
        # Create 2D histogram
        ax.hist2d(df['radial_dist'], df['Position Z'], bins=30, cmap='YlOrRd', alpha=0.6, 
                 range=[[0, 0.7], AXIS_LIMITS['z']])
        ax.scatter(success['radial_dist'], success['Position Z'],
                  c='green', marker='^', s=10, alpha=0.4, label='Success')
        ax.scatter(failure['radial_dist'], failure['Position Z'],
                  c='red', marker='o', s=5, alpha=0.2, label='Failure')
        
        ax.set_xlabel('Radial Distance from Object (m)', fontsize=8)
        ax.set_ylabel('Z Position (m)', fontsize=8)
        ax.set_xlim(0, 0.7)
        ax.set_ylim(AXIS_LIMITS['z'])
        ax.set_title(f'{config.upper().replace("_", "-")}: Noise Distribution', 
                    fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Add std dev annotations
        r_std = df['radial_dist'].std()
        z_std = df['Position Z'].std()
        ax.text(0.02, 0.98, f'σ_r={r_std:.3f}m\nσ_z={z_std:.3f}m',
               transform=ax.transAxes, fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "noise_distribution_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    """Generate all 3D visualization figures"""
    print("="*80)
    print("GENERATING 3D VISUALIZATION FIGURES")
    print("="*80)
    
    configs = ['pr2_cuboid', 'pr2_cylinder', 'sdh_cuboid', 'sdh_cylinder']
    
    print("\n1. Individual configuration 3D plots...")
    for config in configs:
        create_3d_scatter_plot(config)
    
    print("\n2. Combined 3D plot...")
    create_combined_3d_plot()
    
    print("\n3. Noise distribution visualization...")
    create_noise_visualization()
    
    print("\n" + "="*80)
    print("✅ ALL 3D VISUALIZATION FIGURES GENERATED!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Total figures: 6 (4 individual + 1 combined + 1 noise)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
