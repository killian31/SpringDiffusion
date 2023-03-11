import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.util import compute_alphas, get_index_from_list


def sample_timestep(x, t, model, betas):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    with torch.no_grad():
        (
            _,
            _,
            _,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
        ) = compute_alphas(betas)
        betas_t = get_index_from_list(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


def sample_save_image(model, betas, img_file, img_size, device, T):
    # Sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)

    if not os.path.exists("images"):
        os.makedirs("images")

    num_images = 10
    stepsize = int(T / num_images)
    print("Saving sample images...")
    for i in tqdm(range(0, T)[::-1]):
        with torch.no_grad():
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, t, model, betas)
            if i % stepsize == 0:
                img_np = np.transpose(img.detach().cpu().squeeze().numpy(), (1, 2, 0))
                cv2.imwrite(f"images/{img_file}_{i}.png", img_np * 255)
