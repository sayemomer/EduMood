import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import KFold
import numpy as np

def custom_loader_kfold(batch_size, data_path='./Dataset/Train', n_splits=10):
    # Transforms
    normalize = transforms.Normalize(mean=[0.5836, 0.4212, 0.3323],
                                     std=[0.2325, 0.1985, 0.1722])
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        normalize
    ])

    # Load dataset using ImageFolder
    master_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # KFold splitting
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # Creating DataLoaders for each fold
    fold_data_loaders = []
    for train_idx, val_idx in kfold.split(np.arange(len(master_dataset))):
        # Subset the dataset
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        # Create DataLoaders
        train_loader = DataLoader(
            master_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(
            master_dataset, batch_size=batch_size, sampler=val_subsampler)

        fold_data_loaders.append((train_loader, val_loader))

    return fold_data_loaders


