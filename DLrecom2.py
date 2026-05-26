import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------------
# Example data (replace with your actual results from training)
# -------------------------------------------------------------------
# Validation RMSE recorded at each epoch (30 epochs) for NCF
val_rmse_per_epoch = [
    1.104, 0.995, 0.966, 0.950, 0.939,
    0.931, 0.925, 0.921, 0.918, 0.916,
    0.914, 0.913, 0.912, 0.911, 0.911,
    0.912, 0.910, 0.909, 0.908, 0.907,
    0.906, 0.905, 0.904, 0.903, 0.902,
    0.901, 0.900, 0.899, 0.899, 0.898
]

# Final test metrics (replace with your actual numbers)
ncf_test_rmse = 0.876
svd_test_rmse = 0.918
ncf_hit_rate = 0.72
svd_hit_rate = 0.65

# -------------------------------------------------------------------
# 1. RMSE vs. Epoch Curve
# -------------------------------------------------------------------
epochs = range(1, len(val_rmse_per_epoch) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_rmse_per_epoch, marker='o', linestyle='-', linewidth=2, markersize=4, color='royalblue')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation RMSE', fontsize=12)
plt.title('NCF (NeuMF) Learning Curve on MovieLens 100K', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('rmse_vs_epoch.png', dpi=300)
plt.show()

# -------------------------------------------------------------------
# 2. Comparison Bar Chart (RMSE & Hit Rate)
# -------------------------------------------------------------------
metrics = ['RMSE', 'Hit Rate@10']
ncf_scores = [ncf_test_rmse, ncf_hit_rate]
svd_scores = [svd_test_rmse, svd_hit_rate]

x = np.arange(len(metrics))  # label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, ncf_scores, width, label='NCF (NeuMF)', color='darkorange')
bars2 = ax.bar(x + width/2, svd_scores, width, label='Matrix Factorization (SVD)', color='steelblue')

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Comparison: NCF vs. Matrix Factorization', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('comparison_bar_chart.png', dpi=300)
plt.show()

# -------------------------------------------------------------------
# 3. Table of Per‑Epoch Validation RMSE (as printed table & optional image)
# -------------------------------------------------------------------
# Print as formatted table
print("\nPer‑Epoch Validation RMSE for NCF (30 epochs):")
print("-" * 40)
print("Epoch | Validation RMSE")
print("-" * 40)
for i, rmse in enumerate(val_rmse_per_epoch, start=1):
    print(f"{i:5d} | {rmse:.4f}")

# Optional: save table as image using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')
# Create a list of rows for the table
table_data = [['Epoch', 'Validation RMSE']] + [[i, f"{rmse:.4f}"] for i, rmse in enumerate(val_rmse_per_epoch, start=1)]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)
plt.title('NCF Validation RMSE per Epoch', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('per_epoch_rmse_table.png', dpi=300, bbox_inches='tight')
plt.show()