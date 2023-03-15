import argparse
import os
import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append("./../")
from models.UNet import SimpleUnet
from scripts.sample_image import sample_save_image
from scripts.train import train
from utils.datasets import load_transformed_flowers
from utils.util import linear_beta_schedule

parser = argparse.ArgumentParser()

parser.add_argument("--img-size", type=int, default=128, help="size of output images")
parser.add_argument(
    "--sampling-steps", type=int, default=1000, help="number of sampling steps"
)
parser.add_argument(
    "--model-in-file",
    type=str,
    default=None,
    help="model weights",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="image_sample",
    help="image directory in which to save image",
)
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    args = parser.parse_args()
    betas = linear_beta_schedule(timesteps=args.sampling_steps)
    model = SimpleUnet()
    if args.model_in_file is not None:
        if device == "cuda":
            model.load_state_dict(torch.load(args.model_in_file))
        else:
            model.load_state_dict(
                torch.load(args.model_in_file, map_location=torch.device("cpu"))
            )
    else:
        print("No model weights provided")

    sample_save_image(
        model, betas, args.output_dir, args.img_size, device, args.sampling_steps
    )
