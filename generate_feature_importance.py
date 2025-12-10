"""
Generate feature importance plots for all configurations
Requires trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = Path("latex_report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING FEATURE IMPORTANCE PLOTS")
print("="*80)


def load_or_train_model(config):
    """Load existing model or train new one"""
    model_path = f"models/grasp_model_{config}.pkl"
    scaler_path = f"models/scaler_{config}.pkl"
    data_path = f"data/grasp_data_{config}.csv"
    
    # Try to load existing model
    if Path(model_path).exists() and Path(scaler_path).exists():
        print(f"   Loading existing model for {config}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    
    # Train new model if data exists
    if not Path(data_path).exists():
        print(f"   ⚠️  No data found for {config}")
        return None, None
    
    print(f"   Training new model for {config}")
    df = pd.read_csv(data_path)
    
    # Extract features
    feature_cols = ['Position X', 'Position Y', 'Position Z', 
                   'Orientation Roll', 'Orientation Pitch', 'Orientation Yaw']
    X = df[feature_cols].values
    y = df['Success'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler


def plot_feature_importance(config, config_name):
    """Generate feature importance plot"""
    print(f"\n📊 Generating feature importance: {config_name}")
    
    model, scaler = load_or_train_model(config)
    if model is None:
        return
    
    # Get feature importances
    feature_names = ['Position X', 'Position Y', 'Position Z', 
                    'Orientation Roll', 'Orientation Pitch', 'Orientation Yaw']
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.bar(range(len(importances)), importances[indices], 
                   color=[colors[i] for i in indices])
    
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance: {config_name}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
              rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # Add summary text
    total_position = sum([importances[i] for i in range(3)])
    total_orientation = sum([importances[i] for i in range(3, 6)])
    summary_text = f'Position total: {total_position:.3f}\nOrientation total: {total_orientation:.3f}'
    plt.text(0.98, 0.98, summary_text,
            transform=plt.gca().transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"feature_importance_{config}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def main():
    """Main function"""
    
    configs = [
        ('pr2_cuboid', 'PR2-Cuboid'),
        ('pr2_cylinder', 'PR2-Cylinder'),
        ('sdh_cuboid', 'SDH-Cuboid'),
        ('sdh_cylinder', 'SDH-Cylinder')
    ]
    
    for config, name in configs:
        plot_feature_importance(config, name)
    
    print("\n" + "="*80)
    print("✅ ALL FEATURE IMPORTANCE PLOTS GENERATED!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
