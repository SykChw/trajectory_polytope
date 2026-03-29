import numpy as np
import torch

def boundary_sample(model, t, n_samples=15000, threshold=0.1, device="cpu"):
    pts = np.random.uniform(-1, 1, size=(n_samples, 3))

    t_col = np.full((n_samples, 1), t * 2 - 1)
    inputs = np.hstack([pts, t_col])

    with torch.no_grad():
        logits = model(torch.tensor(inputs, dtype=torch.float32).to(device))
        scores = logits.cpu().numpy()

    boundary_mask = np.abs(scores[:, 0]) < threshold
    inside_mask = scores[:, 0] > 0

    return pts[boundary_mask], pts[inside_mask]