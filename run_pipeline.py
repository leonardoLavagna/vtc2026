import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.hyperparams import EPOCHS, LR
from config.gate_options import gate_options
from config.paths import LOG_DIR, PLOT_DIR, FILES_DIR
from core.data_loader import load_dataset
from core.utils import curvature
from core.meta_opt import meta_optimize


# Load data
df, target_v = load_dataset()
times = df["time"].values
# Build meta-grid
meta_grid = [
    {
        "n_qubits": nq,
        "depth": d,
        "gate_set": g,
        "feature_order": fo,
    }
    for nq in [2, 4]
    for d in [1, 2, 3]
    for g in gate_options
    for fo in [1, 2, 3]
]
dyn_params = {"target_v": target_v}
log_file = os.path.join(LOG_DIR, "training_log.txt")
open(log_file, "w").close()
# Run meta-optimization
best_model, best_score = meta_optimize(
    meta_grid, times, curvature, dyn_params,
    epochs=EPOCHS, lr=LR, log_file=log_file
)
# Evaluate best model
T = torch.tensor(times, dtype=torch.float32)
outputs = best_model.forward_sequence(T).detach().numpy()
# Hybrid velocity extraction
if outputs.shape[1] == 4:
    v_pred = outputs[:, 1]
else:
    v_pred = outputs[:, 0]
v_true = target_v.numpy()
# Save predictions
results_df = pd.DataFrame({
    "time": times,
    "v_true": v_true,
    "v_pred": v_pred
})
# If 4-qubit model, also save u, a, x
if outputs.shape[1] == 4:
    results_df["u_pred"] = outputs[:, 0]
    results_df["a_pred"] = outputs[:, 2]
    results_df["x_pred"] = outputs[:, 3]
# Save file
predictions_path = os.path.join(FILES_DIR, "best_model_predictions.csv")
results_df.to_csv(predictions_path, index=False)
print("Saved predictions to:", predictions_path)
# Plot 
plt.figure(figsize=(12, 4))
plt.plot(times, v_true, label="True velocity", linewidth=2)
plt.plot(times, v_pred, "--", label="Predicted velocity", linewidth=2)
plt.legend()
plt.grid(True)
plot_path = os.path.join(PLOT_DIR, "velocity_curve.png")
plt.savefig(plot_path)
plt.show()
print("\nSaved plot to:", plot_path)
print("Saved logs to:", log_file)
