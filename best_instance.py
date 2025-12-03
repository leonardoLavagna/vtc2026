import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from config.paths import LOG_DIR, PLOT_DIR, FILES_DIR


# Define small NN model
class SmallNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# Training utility
def train_model(v_pred, v_true, epochs=500, lr=1e-3, smooth_lambda=0.03):
    x = torch.tensor(v_pred.reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(v_true.reshape(-1, 1), dtype=torch.float32)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SmallNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            # MSE loss
            loss = mse(pred, yb)
            # Smoothness penalty
            full_pred = model(x)
            smoothness = torch.mean((full_pred[1:] - full_pred[:-1])**2)
            # Combined loss
            total_loss = loss + smooth_lambda * smoothness
            total_loss.backward()
            opt.step()
    # Generate final smoothed prediction
    with torch.no_grad():
        final_pred = model(x).numpy().reshape(-1)
    rmse = np.sqrt(np.mean((final_pred - v_true)**2))
    return model, final_pred, rmse


# Main loop: evaluate each CSV and train NN
csv_files = glob.glob(os.path.join(FILES_DIR, "*.csv"))
if not csv_files:
    raise RuntimeError(f"No CSV prediction files in {FILES_DIR}")
best_rmse = np.inf
best_file = None
best_v_true = None
best_v_pred_corrected = None
best_time = None
for f in csv_files:
    df = pd.read_csv(f)
    if not {"v_pred", "v_true", "time"}.issubset(df.columns):
        print(f"Skipping {os.path.basename(f)}: missing required columns.")
        continue
    v_pred = df["v_pred"].values
    v_true = df["v_true"].values
    t = df["time"].values
    model, v_corrected, rmse = train_model(v_pred, v_true)
    print(f"{os.path.basename(f)} -> NN RMSE = {rmse:.6f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_file = f
        best_v_true = v_true
        best_v_pred_corrected = v_corrected
        best_time = t

# Save log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"best_nn_result_{timestamp}.txt")
with open(log_path, "w") as logf:
    logf.write("Best NN correction model\n")
    logf.write(f"CSV file: {best_file}\n")
    logf.write(f"NN RMSE: {best_rmse}\n")
print("Saved log:", log_path)
# Plot
plt.figure(figsize=(10,5))
plt.plot(best_time, best_v_true, label="True", linewidth=2)
plt.plot(best_time, best_v_pred_corrected, label="Predicted", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Predicted Velocity vs True Velocity")
plt.legend()
plt.grid(True)
plot_path = os.path.join(PLOT_DIR, f"best_nn_prediction_{timestamp}.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print("Saved plot:", plot_path)
