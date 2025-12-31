import numpy as np
import torch

def load_pendulum(model_type="hnn"):
    data = np.load("data/pendulum_dataset.npz")
    
    if model_type in ["hnn","lnn"]:
        theta = data["theta"]
        omega = data["omega"]
        X = np.stack([np.sin(theta), np.cos(theta), omega], axis=1)
        dx = data["dx"]  
    elif model_type in ["mlp", "vector_field"]:
        X = np.stack([data["theta"], data["omega"]], axis=1)
        dx = data["dx"]
    else:
        raise ValueError("Unknown model_type")

    return torch.tensor(X, dtype=torch.float32), torch.tensor(dx, dtype=torch.float32)
