# datasets/client_dataset.py

from torch.utils.data import DataLoader, Subset


def get_client_loader(dataset, indices, batch_size=64, shuffle=True):
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    return loader