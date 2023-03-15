import argparse
import os
import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append("./../")
from models.UNet import SimpleUnet
from scripts.train import train
from utils.datasets import load_transformed_flowers
from utils.util import linear_beta_schedule

parser = argparse.ArgumentParser()

parser.add_argument("--img-size", type=int, default=128, help="size of output images")
parser.add_argument("--batch-size", type=int, default=32, help="size of the batches")
parser.add_argument(
    "--sampling-steps", type=int, default=1000, help="number of sampling steps"
)
parser.add_argument(
    "--epochs", type=int, default=20, help="number of epochs to train for"
)
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="model checkpoint to resume training from",
)
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    args = parser.parse_args()
    data = load_transformed_flowers(args.img_size)
    dataloader = DataLoader(
        data, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    betas = linear_beta_schedule(timesteps=args.sampling_steps)
    model = SimpleUnet()
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        os.makedirs("./weights", exist_ok=True)
    optimizer = Adam(model.parameters(), lr=args.lr)
    model, losses = train(
        optimizer,
        args.epochs,
        device,
        dataloader,
        args.batch_size,
        args.sampling_steps,
        model,
        args.img_size,
        betas,
    )
