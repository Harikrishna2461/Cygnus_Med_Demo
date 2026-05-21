"""
Comprehensive training metrics visualization for Medical LLM CPT.
Generates multiple meaningful plots to analyze training quality.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH = Path("logs") / "training_metrics.csv"
OUTPUT_DIR = Path("logs")
DPI = 150

# ── Load Data ──────────────────────────────────────────────────────────────────
print("Loading training metrics...")
df = pd.read_csv(CSV_PATH)
df = df.dropna()

if len(df) == 0:
    print("No metrics data found!")
    exit(1)

print(f"Loaded {len(df)} metric records")
print(f"Steps: {df['step'].min():.0f} → {df['step'].max():.0f}")

# Calculate derived metrics
df['loss_improvement'] = df['train_loss'].iloc[0] - df['train_loss']
df['eval_perplexity'] = np.exp(df['eval_loss'])
df['train_perplexity'] = np.exp(df['train_loss'])

# Smoothing for trend visualization
window = max(3, len(df) // 20)  # Smooth over ~5% of data
df['train_loss_smooth'] = uniform_filter1d(df['train_loss'], size=window, mode='nearest')
df['eval_loss_smooth'] = uniform_filter1d(df['eval_loss'], size=window, mode='nearest')

# ── Figure 1: Training & Validation Loss ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df['step'], df['train_loss'], 'o-', label='Training Loss',
        color='#2E86AB', alpha=0.6, markersize=4, linewidth=1.5)
ax.plot(df['step'], df['train_loss_smooth'], '-', label='Training Loss (Smoothed)',
        color='#2E86AB', linewidth=2.5)

ax.plot(df['step'], df['eval_loss'], 's-', label='Validation Loss',
        color='#A23B72', alpha=0.6, markersize=4, linewidth=1.5)
ax.plot(df['step'], df['eval_loss_smooth'], '-', label='Validation Loss (Smoothed)',
        color='#A23B72', linewidth=2.5)

ax.fill_between(df['step'], df['train_loss'], df['eval_loss'],
                alpha=0.1, color='gray', label='Train-Val Gap')

ax.set_xlabel('Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training vs Validation Loss Over Time', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_loss_curves.png", dpi=DPI, bbox_inches='tight')
print("✓ Saved: 01_loss_curves.png")
plt.close()

# ── Figure 2: Perplexity ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df['step'], df['train_perplexity'], 'o-', label='Training Perplexity',
        color='#06A77D', alpha=0.6, markersize=4, linewidth=1.5)
ax.plot(df['step'], df['eval_perplexity'], 's-', label='Validation Perplexity',
        color='#D62828', alpha=0.6, markersize=4, linewidth=1.5)

ax.set_xlabel('Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
ax.set_title('Model Perplexity Improvement', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_perplexity.png", dpi=DPI, bbox_inches='tight')
print("✓ Saved: 02_perplexity.png")
plt.close()

# ── Figure 3: Loss Improvement Rate ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

improvement_pct = (df['loss_improvement'] / df['train_loss'].iloc[0]) * 100

ax.fill_between(df['step'], 0, improvement_pct, alpha=0.3, color='#06A77D', label='Total Improvement')
ax.plot(df['step'], improvement_pct, 'o-', color='#06A77D', linewidth=2, markersize=5)

# Add annotation for final improvement
final_improvement = improvement_pct.iloc[-1]
ax.text(df['step'].iloc[-1], final_improvement, f'{final_improvement:.1f}%',
        fontsize=11, fontweight='bold', ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss Improvement (%)', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Loss Improvement Over Training', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_improvement_rate.png", dpi=DPI, bbox_inches='tight')
print("✓ Saved: 03_improvement_rate.png")
plt.close()

# ── Figure 4: Training Dynamics (2x2 Subplots) ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Raw losses
ax = axes[0, 0]
ax.plot(df['step'], df['train_loss'], 'o-', label='Train', color='#2E86AB', markersize=3)
ax.plot(df['step'], df['eval_loss'], 's-', label='Val', color='#A23B72', markersize=3)
ax.set_ylabel('Loss', fontweight='bold')
ax.set_title('Raw Loss Values', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Loss gap (overfitting indicator)
ax = axes[0, 1]
loss_gap = df['eval_loss'] - df['train_loss']
ax.fill_between(df['step'], 0, loss_gap, alpha=0.4, color='#F18F01')
ax.plot(df['step'], loss_gap, 'o-', color='#F18F01', markersize=3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax.set_ylabel('Loss Gap (Val - Train)', fontweight='bold')
ax.set_title('Overfitting Indicator', fontweight='bold')
ax.grid(True, alpha=0.3)

# Subplot 3: Perplexity ratio
ax = axes[1, 0]
ppl_ratio = df['eval_perplexity'] / df['train_perplexity']
ax.plot(df['step'], ppl_ratio, 'o-', color='#C1121F', markersize=3)
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (ratio=1)')
ax.set_xlabel('Step', fontweight='bold')
ax.set_ylabel('PPL Ratio (Val/Train)', fontweight='bold')
ax.set_title('Generalization Quality', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: Step-wise improvement
ax = axes[1, 1]
loss_delta = -df['train_loss'].diff()  # Negative because we want positive improvement
ax.bar(df['step'], loss_delta, color='#06A77D', alpha=0.6, width=20)
ax.axhline(y=0, color='k', linewidth=0.8)
ax.set_xlabel('Step', fontweight='bold')
ax.set_ylabel('Loss Improvement per Step', fontweight='bold')
ax.set_title('Training Progress Per Step', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Training Dynamics Overview', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_training_dynamics.png", dpi=DPI, bbox_inches='tight')
print("✓ Saved: 04_training_dynamics.png")
plt.close()

# ── Figure 5: Convergence Analysis ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate moving average of loss changes
window_size = max(2, len(df) // 10)
loss_velocity = df['eval_loss'].rolling(window=window_size).std()

ax2 = ax.twinx()

# Primary axis: Loss
ax.plot(df['step'], df['eval_loss'], 'o-', color='#A23B72',
        label='Validation Loss', linewidth=2, markersize=5)
ax.fill_between(df['step'], df['eval_loss'].min(), df['eval_loss'],
                alpha=0.15, color='#A23B72')

# Secondary axis: Stability (inverse of velocity)
ax2.plot(df['step'], loss_velocity, '--', color='#F4A261',
         label='Loss Stability', linewidth=2, alpha=0.7)

ax.set_xlabel('Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold', color='#A23B72')
ax2.set_ylabel('Loss Variance (Stability)', fontsize=12, fontweight='bold', color='#F4A261')

ax.tick_params(axis='y', labelcolor='#A23B72')
ax2.tick_params(axis='y', labelcolor='#F4A261')

ax.set_title('Convergence & Stability Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_convergence_stability.png", dpi=DPI, bbox_inches='tight')
print("✓ Saved: 05_convergence_stability.png")
plt.close()

# ── Summary Statistics ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

print(f"\nLoss Metrics:")
print(f"  Initial train loss:     {df['train_loss'].iloc[0]:.4f}")
print(f"  Final train loss:       {df['train_loss'].iloc[-1]:.4f}")
print(f"  Training improvement:   {(df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]):.4f} ({final_improvement:.2f}%)")

print(f"\nValidation Metrics:")
print(f"  Best val loss:          {df['eval_loss'].min():.4f} (step {df.loc[df['eval_loss'].idxmin(), 'step']:.0f})")
print(f"  Final val loss:         {df['eval_loss'].iloc[-1]:.4f}")
print(f"  Val loss stability:     σ = {df['eval_loss'].std():.4f}")

print(f"\nPerplexity:")
print(f"  Initial train PPL:      {df['train_perplexity'].iloc[0]:.2f}")
print(f"  Final train PPL:        {df['train_perplexity'].iloc[-1]:.2f}")
print(f"  Final val PPL:          {df['eval_perplexity'].iloc[-1]:.2f}")

print(f"\nOverfitting Analysis:")
loss_gap_final = df['eval_loss'].iloc[-1] - df['train_loss'].iloc[-1]
print(f"  Final train-val gap:    {loss_gap_final:.4f}")
print(f"  Mean train-val gap:     {(df['eval_loss'] - df['train_loss']).mean():.4f}")
print(f"  Status:                 {'✓ Healthy' if loss_gap_final < 0.5 else '⚠ Watch for overfitting'}")

print(f"\nTotal Steps:             {len(df)}")
print(f"Evaluation Frequency:    Every {int(df['step'].diff().mode()[0])} steps")

print("="*70)
print("✓ All plots saved to: logs/")
print("="*70)
