import torch
import torchvision


# Load dataset
def load_flowers():
    return torchvision.datasets.Flowers102(root="../data", download=True)
