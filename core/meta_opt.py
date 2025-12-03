import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from core.model_components import QuantumDynamicsModel
from core.training import train_inner
from config.paths import PLOT_DIR, FILES_DIR


def meta_optimize(meta_grid, times, curvature, dyn_params, epochs, lr, log_file):
    best_model = None
    best_score = float("inf")
    # Start logging
    with open(log_file, "a") as f:
        f.write("Starting meta-optimization\n")
    # Loop over all configurations
    for idx, cfg in enumerate(meta_grid):
        msg = f"\n[CONFIG {idx}] Trying config: {cfg}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        # Build model
        model = QuantumDynamicsModel(cfg)
        # Train this configuration
        score = train_inner(
            model, times, curvature, dyn_params,
            epochs=epochs, lr=lr, log_file=log_file
        )
        # Evaluate trajectory
        T = torch.tensor(times, dtype=torch.float32)
        outputs = model.forward_sequence(T).detach().numpy()
        # Hybrid velocity extraction
        if outputs.shape[1] == 1:
            v_pred = outputs[:, 0]     
        else:
            v_pred = outputs[:, 1]      
        v_true = dyn_params["target_v"].numpy()
        prediction_file = os.path.join(FILES_DIR, f"config_{idx}_predictions.csv")
        df_out = {
            "time": times,
            "v_true": v_true,
            "v_pred": v_pred,
        }
        # Also save u, a, x for 4-qubit models
        if outputs.shape[1] == 4:
            df_out["u_pred"] = outputs[:, 0]
            df_out["a_pred"] = outputs[:, 2]
            df_out["x_pred"] = outputs[:, 3]
        df_out = pd.DataFrame(df_out)
        df_out.to_csv(prediction_file, index=False)
        print(f"Saved predictions: {prediction_file}")
        with open(log_file, "a") as f:
            f.write(f"Saved predictions: {prediction_file}\n")
        plt.figure(figsize=(12, 4))
        plt.plot(times, v_true, label="True", linewidth=2)
        plt.plot(times, v_pred, "--", label="Predicted", linewidth=2)
        plt.title(f"Velocity - Config {idx}: {cfg}")
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.legend()
        plt.grid(True)
        plot_name = f"velocity_config_{idx}.png"
        plot_path = os.path.join(PLOT_DIR, plot_name)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        with open(log_file, "a") as f:
            f.write(f"Saved plot: {plot_path}\n")
        # Update best model
        if score < best_score:
            best_score = score
            best_model = model

    return best_model, best_score
