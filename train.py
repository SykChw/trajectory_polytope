import torch
import torch.nn as nn
import torch.optim as optim

from model import TrajectoryNet
from dataloader import get_dataloader


# DEFINE YOUR FUNCTION HERE
def f(x, y, z, t):
    cx = torch.sin(2 * torch.pi * t)
    cy = torch.cos(2 * torch.pi * t)
    cz = 0.5 * torch.sin(4 * torch.pi * t)

    r = 0.3 + 0.1 * torch.sin(2 * torch.pi * t)

    return r**2 - ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)


def estimate_pos_weight(loader, device):
    pos = 0
    total = 0

    for X, y in loader:
        pos += y.sum().item()
        total += y.numel()

    neg = total - pos
    return torch.tensor([neg / (pos + 1e-6)], device=device)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TrajectoryNet().to(device)
    loader = get_dataloader(f)

    pos_weight = estimate_pos_weight(loader, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(25):
        total_loss = 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "trajectory_model.pt")


if __name__ == "__main__":
    train()