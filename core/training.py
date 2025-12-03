import torch
import torch.optim as optim
from core.loss_functions import loss_fn


def train_inner(model, times, curvature_fn, dyn_params, epochs, lr, log_file):
    opt = optim.Adam(model.parameters(), lr=lr)
    T = torch.tensor(times, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        outs = model.forward_sequence(T)
        loss = loss_fn(outs, curvature_fn, dyn_params)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            msg = f"[EPOCH {ep}] loss = {loss.item():.6f}"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
    return loss.item()
