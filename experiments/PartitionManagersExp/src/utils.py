import matplotlib.pyplot as plt
import torch
import numpy as np

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    return data

def plot_predictions(y_true, y_pred, title, outpath):
    y_true_np = to_numpy(y_true)
    y_pred_np = to_numpy(y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_np, y_pred_np, alpha=0.6, s=15, edgecolors='k', linewidth=0.5)
    
    # Línea ideal
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()