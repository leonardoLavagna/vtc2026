import torch
from config.hyperparams import (LAMBDA_BACK, LAMBDA_SMOOTH, LAMBDA_MATCH, LAMBDA_VMAX,
                                A_NU_MAX, A_MAX, U_MAX, V_MAX, LAMBDA_MONO)


def loss_fn(outputs, curvature_fn, params):
    channels = outputs.shape[1]
    target_v = params["target_v"].to(outputs.device, outputs.dtype)
    # CASE 1 (2 qubits)
    if channels < 4:
        v = outputs[:, 0]   
        # smoothness of velocity
        smoothness = LAMBDA_SMOOTH * torch.mean((v[1:] - v[:-1])**2)
        # match true velocity
        match_loss = LAMBDA_MATCH * torch.mean((v - target_v)**2)
        # forbid backward motion
        no_back = LAMBDA_BACK * torch.mean(torch.relu(-v))
        # enforce physical max speed
        too_fast = LAMBDA_VMAX * torch.mean(torch.relu(v - V_MAX))
        dv = v[1:] - v[:-1]
        monotonicity = LAMBDA_MONO * torch.mean(torch.relu(-dv))
        return smoothness + match_loss + no_back + too_fast + monotonicity

    # CASE 2 
    u = outputs[:, 0]
    v = outputs[:, 1]
    a = outputs[:, 2]
    x = outputs[:, 3]
    # kinematic residuals
    dv = torch.gradient(v)[0]
    dx = torch.gradient(x)[0]
    da = torch.gradient(a)[0]
    rx = torch.mean((dx - v)**2)
    rv = torch.mean((dv - a)**2)
    ra = torch.mean((da - u)**2)
    # constraints
    c_lat  = torch.mean(torch.relu(torch.abs(v * curvature_fn(x)) - A_NU_MAX))
    c_acc  = torch.mean(torch.relu(torch.abs(a) - A_MAX))
    c_jerk = torch.mean(torch.relu(torch.abs(u) - U_MAX))
    # positivity of velocity
    no_back = LAMBDA_BACK * torch.mean(torch.relu(-v))
    # max velocity
    too_fast = LAMBDA_VMAX * torch.mean(torch.relu(v - V_MAX))
    # smoothness
    smoothness = LAMBDA_SMOOTH * torch.mean((v[1:] - v[:-1])**2)
    # match true velocity
    match_loss = LAMBDA_MATCH * torch.mean((v - target_v)**2)
    dv_step = v[1:] - v[:-1]
    monotonicity = LAMBDA_MONO * torch.mean(torch.relu(-dv_step))
    return (rx + rv + ra +
            c_lat + c_acc + c_jerk +
            no_back + too_fast +
            smoothness + match_loss +
            monotonicity)
