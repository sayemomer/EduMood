# dataloader.py

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


def custom_loader(batch_size, shuffle_test=False, data_path='./Dataset/Train'):
    # Add the necessary transforms
    normalize = transforms.Normalize(mean=[0.5836, 0.4212, 0.3323],
                                     std=[0.2325, 0.1985, 0.1722])
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, 4),
        # Adjust this if your images are a different size
        transforms.Resize((48, 48)),
        # transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        normalize
    ])

    # Load your dataset using ImageFolder
    master_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Calculate the sizes of the splits
    total_size = len(master_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    # size of the training set
    print("train size=", train_size)
    print("validation size=", val_size)

    # Use random_split to create datasets for training, testing, and validation
    train_dataset, val_dataset = random_split(
        master_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
