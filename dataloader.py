from torch.utils.data import DataLoader
from dataset import FunctionalTrajectoryDataset

def get_dataloader(f, batch_size=512, n_samples=100000):
    dataset = FunctionalTrajectoryDataset(f, n_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
