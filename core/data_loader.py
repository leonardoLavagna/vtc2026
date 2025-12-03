import pandas as pd
import torch
from config.paths import DATA_PATH

def load_dataset(t_min=0, t_max=25):
    df = pd.read_csv(DATA_PATH)
    df = df[(df["time"] >= t_min) & (df["time"] <= t_max)].reset_index(drop=True)
    true_v = df["velocity"].values
    return df, torch.tensor(true_v, dtype=torch.float32)
