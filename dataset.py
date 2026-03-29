import torch
from torch.utils.data import Dataset

class FunctionalTrajectoryDataset(Dataset):
    def __init__(self, f, n_samples=100000, boundary_ratio=0.3):
        self.f = f
        self.n = n_samples
        self.boundary_ratio = boundary_ratio

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Sample time in [0,1]
        t = torch.rand(1)

        # Normalize time to [-1,1]
        t_norm = t * 2 - 1

        # Sample spatial point
        if torch.rand(1) < self.boundary_ratio:
            # boundary-focused sampling
            x = torch.rand(1) * 2 - 1
            y = torch.rand(1) * 2 - 1
            z = torch.rand(1) * 2 - 1

            noise = torch.randn(3) * 0.05
            x, y, z = x + noise[0], y + noise[1], z + noise[2]
        else:
            x = torch.rand(1) * 2 - 1
            y = torch.rand(1) * 2 - 1
            z = torch.rand(1) * 2 - 1

        # Evaluate function
        val = self.f(x, y, z, t)

        label = (val >= 0).float()

        inp = torch.cat([x, y, z, t_norm], dim=0)

        return inp, label
