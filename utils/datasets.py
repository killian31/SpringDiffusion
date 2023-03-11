import torch
import torchvision
from torchvision import transforms


# Load dataset
def load_flowers():
    return torchvision.datasets.Flowers102(root="../data", download=True)


def load_transformed_flowers(img_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.Flowers102(
        root=".", download=True, transform=data_transform
    )

    test = torchvision.datasets.Flowers102(
        root=".", download=True, transform=data_transform, split="test"
    )
    return torch.utils.data.ConcatDataset([train, test])
