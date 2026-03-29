import torch
from model import TrajectoryNet

def load_model(path, device):
    model = TrajectoryNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model