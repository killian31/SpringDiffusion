import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append("./../")
from models.UNet import SimpleUnet
from scripts.train import train
from utils.datasets import load_transformed_flowers
from utils.util import linear_beta_schedule

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    img_size = 128
    data = load_transformed_flowers(img_size)
    batch_size = 128
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    T = 1000
    betas = linear_beta_schedule(timesteps=T)
    model = SimpleUnet()
    if os.path.exists("./weights/weights.pt"):
        model.load_state_dict(torch.load("./weights/weights.pt"))
    else:
        os.makedirs("./weights", exist_ok=True)
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 100
    model, losses = train(
        optimizer, epochs, device, dataloader, batch_size, T, model, img_size, betas
    )
